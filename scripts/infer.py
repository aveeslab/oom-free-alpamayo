"""Run Alpamayo-R1 inference using a saved residency config.

Usage:
    python scripts/infer.py --config config.json
    python scripts/infer.py --config config.json --num-iterations 5
    python scripts/infer.py --config config.json --input data.json --output traj.json

The script:
    1. Loads the config produced by scripts/profile.py.
    2. Loads Alpamayo-R1, moves non-VLM essentials to GPU.
    3. Loads resident VLM layers (per config) directly to GPU.
    4. Pins remaining (offload) VLM layers to host pinned memory and
       installs DoubleBufHook for asynchronous prefetch via DFB.
    5. Runs inference, prints timing, and optionally writes output trajectories.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

_PKG_ROOT = Path(__file__).resolve().parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from alpamayo_memopt import DoubleBufHook, load_config


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run Alpamayo-R1 inference with saved residency config."
    )
    p.add_argument("--config", "-c", type=Path, required=True,
                   help="Path to config.json produced by scripts/profile.py")
    p.add_argument("--num-iterations", "-n", type=int, default=1,
                   help="Number of inference iterations (default: 1)")
    p.add_argument("--warmup", type=int, default=1,
                   help="Number of warmup iterations (untimed). Default: 1.")
    p.add_argument("--input", type=Path, default=None,
                   help=("Optional input JSON. If omitted, the standard Alpamayo "
                         "benchmark sample is used."))
    p.add_argument("--output", "-o", type=Path, default=None,
                   help="Optional output path for predicted trajectories (JSON).")
    p.add_argument("--device", type=int, default=0, help="CUDA device index.")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────
# Model glue (Alpamayo-specific)
# ─────────────────────────────────────────────────────────────────────

def _load_alpamayo():
    """Load Alpamayo-R1 onto CPU and return (model, vlm_layers)."""
    try:
        from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
    except ImportError as e:
        raise ImportError(
            "alpamayo_r1 package not found. Install Alpamayo-R1 first."
        ) from e

    model = AlpamayoR1.from_pretrained(
        "nvidia/Alpamayo-R1-10B",
        dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    model.eval()
    vlm_layers = model.vlm.model.language_model.layers
    return model, vlm_layers


def _setup_non_vlm_essentials(model) -> None:
    """Move non-VLM-layer essentials (embed, norm, lm_head, ViT, Expert
    non-layers, diffusion) to GPU."""
    lm = model.vlm.model.language_model
    visual = model.vlm.model.visual

    lm.embed_tokens.to("cuda")
    lm.norm.to("cuda")
    lm.rotary_emb.to("cuda")
    model.vlm.lm_head.to("cuda")

    for attr in ("action_in_proj", "action_out_proj", "diffusion"):
        if hasattr(model, attr):
            getattr(model, attr).to("cuda")

    for n, m in visual.named_children():
        if n != "blocks":
            m.to("cuda")
    for b in visual.blocks:
        b.to("cuda")

    for n, child in model.expert.named_children():
        if n != "layers":
            child.to("cuda")
    for n, p in model.expert.named_parameters():
        if "layers." not in n and p.device.type == "cpu":
            p.data = p.data.to("cuda")
    for n, b in model.expert.named_buffers():
        if "layers." not in n and b.device.type == "cpu":
            b.data = b.data.to("cuda")


def _prepare_default_inputs(model):
    """Prepare the standard Alpamayo benchmark sample as inference input."""
    try:
        from alpamayo_r1 import helper
        from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
        import physical_ai_av
    except ImportError as e:
        raise ImportError(
            "Alpamayo dataset utilities not available."
        ) from e

    avdi = physical_ai_av.PhysicalAIAVDatasetInterface()
    data = load_physical_aiavdataset(
        "030c760c-ae38-49aa-9ad8-f5650a545d26", t0_us=5_100_000, avdi=avdi
    )
    messages = helper.create_message(data["image_frames"].flatten(0, 1))
    processor = helper.get_processor(model.tokenizer)
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False,
        continue_final_message=True, return_dict=True, return_tensors="pt",
    )
    return helper.to_device({
        "tokenized_data": inputs,
        "ego_history_xyz": data["ego_history_xyz"],
        "ego_history_rot": data["ego_history_rot"],
    }, "cuda")


def _serialize_output(out, path: Path) -> None:
    """Best-effort JSON serialization of inference output."""
    def _to_jsonable(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().tolist()
        if isinstance(x, dict):
            return {k: _to_jsonable(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [_to_jsonable(v) for v in x]
        return x

    path.write_text(json.dumps(_to_jsonable(out), indent=2))


# ─────────────────────────────────────────────────────────────────────
# Inference pipeline
# ─────────────────────────────────────────────────────────────────────

def main() -> int:
    args = parse_args()
    print("=" * 60)
    print("Alpamayo Memory Optimizer — Inference")
    print("=" * 60)

    # 1. Load config
    print(f"\n[1] Loading config: {args.config}")
    config = load_config(args.config)
    print(f"    GPU expected         : {config.system.gpu_name}")
    print(f"    VRAM budget          : {config.system.vram_budget_gb:.2f} GB")
    print(f"    Resident layers      : {config.residency.num_resident} "
          f"of {config.model.vlm_layers}")
    print(f"    Predicted time       : {config.predicted_performance.inference_time_s:.3f} s")

    # GPU sanity check
    actual_gpu = torch.cuda.get_device_properties(args.device).name
    if actual_gpu != config.system.gpu_name:
        print(f"    [!] Current GPU '{actual_gpu}' differs from config "
              f"'{config.system.gpu_name}'. Performance may vary.")

    # 2. Load model
    print("\n[2] Loading Alpamayo-R1-10B (CPU)...")
    model, vlm_layers = _load_alpamayo()
    if len(vlm_layers) != config.model.vlm_layers:
        print(f"    [!] VLM layer count mismatch: model has {len(vlm_layers)}, "
              f"config expects {config.model.vlm_layers}", file=sys.stderr)
        return 2

    # 3. Setup non-VLM essentials on GPU
    print("\n[3] Setting up non-VLM components on GPU...")
    _setup_non_vlm_essentials(model)

    # 4. Move resident VLM layers to GPU
    resident_indices = set(config.residency.resident_indices)
    offload_indices = sorted(set(range(len(vlm_layers))) - resident_indices)
    print(f"\n[4] Loading {len(resident_indices)} resident VLM layers to GPU...")
    for i in resident_indices:
        vlm_layers[i].to("cuda")

    # 5. Pin offload layers + DoubleBufHook
    print(f"\n[5] Pinning {len(offload_indices)} offload VLM layers + "
          "installing DoubleBufHook...")
    hook = DoubleBufHook(auto_restart=True)
    hook.pin(vlm_layers, offload_indices)
    hook.allocate(hook.max_elements())
    hook.register(vlm_layers, offload_indices)
    torch.cuda.synchronize()
    peak_after_setup_gb = torch.cuda.memory_allocated(args.device) / (1024 ** 3)
    print(f"    VRAM after setup     : {peak_after_setup_gb:.2f} GB "
          f"(budget {config.system.vram_budget_gb:.2f} GB)")
    if peak_after_setup_gb > config.system.vram_budget_gb:
        print(f"    [!] Allocated VRAM exceeds the budget set in config.",
              file=sys.stderr)

    # 6. Prepare inputs
    print("\n[6] Preparing inputs...")
    if args.input is not None:
        raise NotImplementedError(
            "Custom --input not yet supported. Run without --input to use the "
            "default Alpamayo benchmark sample."
        )
    inputs = _prepare_default_inputs(model)

    # 7. Run inference (warmup + timed iterations)
    print(f"\n[7] Running inference: {args.warmup} warmup + "
          f"{args.num_iterations} timed iteration(s)...")

    @torch.no_grad()
    def _run_once():
        hook.start()
        out = model.sample_trajectories_from_data_with_vlm_rollout(**inputs)
        hook.reset()
        return out

    for _ in range(args.warmup):
        _run_once()
    torch.cuda.synchronize()

    times = []
    last_out = None
    for i in range(args.num_iterations):
        t0 = time.perf_counter()
        last_out = _run_once()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)
        print(f"    iter {i + 1}: {times[-1]:.3f} s")

    if times:
        mean_s = sum(times) / len(times)
        std_s = (sum((t - mean_s) ** 2 for t in times) / len(times)) ** 0.5
        print(f"\n    Mean inference time  : {mean_s:.3f} s "
              f"(std {std_s:.3f}, n={len(times)})")
        print(f"    Predicted (config)   : "
              f"{config.predicted_performance.inference_time_s:.3f} s")
        print(f"    Δ vs predicted       : "
              f"{(mean_s - config.predicted_performance.inference_time_s) * 1000:+.1f} ms")

    # 8. Save output if requested
    if args.output is not None and last_out is not None:
        print(f"\n[8] Saving output to {args.output}...")
        _serialize_output(last_out, args.output)

    # Cleanup
    hook.remove()

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
