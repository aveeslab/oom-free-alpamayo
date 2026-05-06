"""Profile system specs + Alpamayo model layers, predict optimal residency,
and save a config JSON consumable by scripts/infer.py.

Usage:
    python scripts/profile.py --output config.json
    python scripts/profile.py --vram-budget 12.0 --output config.json
    python scripts/profile.py --baseline-time 14.52 --output config.json

The script:
    1. Detects CPU DRAM total and verifies it can hold the full model weights.
    2. Detects GPU VRAM total. If --vram-budget is omitted, defaults to
       (total - 1.0 GB) safety reserve.
    3. Loads Alpamayo-R1, moves non-VLM essentials to GPU, measures
       non_vlm_overhead_gb (VRAM consumed before any VLM layer is loaded).
    4. Runs Sequential Demand Layering (Nr = 0, all VLM layers offloaded)
       once with DoubleBufHook to measure full-offload inference time and
       per-layer DMA timing.
    5. Computes max possible resident layer count, applies a conservative
       margin of -2, and selects resident indices via interleaved placement
       (paper Eq. 8: first layer included, last layer excluded).
    6. Predicts the inference time at the chosen resident count using the
       linear residency-benefit model.
    7. Writes the config JSON.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# Ensure the package is importable when run as a script.
_PKG_ROOT = Path(__file__).resolve().parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from alpamayo_memopt import DoubleBufHook
from alpamayo_memopt import config as cfg
from alpamayo_memopt import predictor, profiler


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=("Profile system + Alpamayo model and produce a residency "
                     "config for memory-efficient inference.")
    )
    p.add_argument("--output", "-o", type=Path, default=Path("config.json"),
                   help="Output config JSON path (default: ./config.json)")
    p.add_argument("--vram-budget", type=float, default=None,
                   help=("Allowed VRAM budget in GB. Default: total VRAM - 1 GB. "
                         "Must be ≤ total VRAM."))
    p.add_argument("--baseline-time", type=float, default=14.52,
                   help=("Reference baseline inference time in seconds for "
                         "speedup reporting. Default: 14.52 (Accelerate offloading "
                         "on RTX 5070 Ti, paper)."))
    p.add_argument("--decode-tokens", type=int, default=21,
                   help="Expected number of VLM decode tokens (default: 21).")
    p.add_argument("--margin", type=int, default=2,
                   help="Conservative resident-count margin (default: 2).")
    p.add_argument("--device", type=int, default=0, help="CUDA device index.")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────
# Model loading hooks (Alpamayo-specific glue)
# ─────────────────────────────────────────────────────────────────────

def _load_alpamayo():
    """Load Alpamayo-R1 onto CPU, return (model, vlm_layers).

    Requires the user to have installed NVIDIA Alpamayo-R1 separately.
    """
    try:
        from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
    except ImportError as e:
        raise ImportError(
            "alpamayo_r1 package not found. Install Alpamayo-R1 first. "
            "See README for instructions."
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
    non-layers, diffusion) to the GPU. VLM layers stay on CPU until the hook
    pins and prefetches them."""
    lm = model.vlm.model.language_model
    visual = model.vlm.model.visual

    lm.embed_tokens.to("cuda")
    lm.norm.to("cuda")
    lm.rotary_emb.to("cuda")
    model.vlm.lm_head.to("cuda")

    for attr in ("action_in_proj", "action_out_proj", "diffusion"):
        if hasattr(model, attr):
            getattr(model, attr).to("cuda")

    # Vision: non-block components to GPU
    for n, m in visual.named_children():
        if n != "blocks":
            m.to("cuda")
    for b in visual.blocks:
        b.to("cuda")  # ViT layers fully resident (small footprint)

    # Expert: non-layer components to GPU; layers handled separately
    for n, child in model.expert.named_children():
        if n != "layers":
            child.to("cuda")
    for n, p in model.expert.named_parameters():
        if "layers." not in n and p.device.type == "cpu":
            p.data = p.data.to("cuda")
    for n, b in model.expert.named_buffers():
        if "layers." not in n and b.device.type == "cpu":
            b.data = b.data.to("cuda")


def _prepare_sample_inputs(model):
    """Prepare a sample input matching the standard benchmark."""
    try:
        from alpamayo_r1 import helper
        from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
        import physical_ai_av
    except ImportError as e:
        raise ImportError(
            "Alpamayo dataset utilities not available. Install Alpamayo-R1 "
            "with its dataset dependencies."
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


# ─────────────────────────────────────────────────────────────────────
# Profiling pipeline
# ─────────────────────────────────────────────────────────────────────

def main() -> int:
    args = parse_args()
    print("=" * 60)
    print("Alpamayo Memory Optimizer — Profiler")
    print("=" * 60)

    # 1. System detection
    print("\n[1] Detecting system specifications...")
    cpu_dram_gb = profiler.detect_cpu_dram_gb()
    gpu_info = profiler.detect_gpu_info(args.device)
    print(f"    CPU DRAM total : {cpu_dram_gb:.2f} GB")
    print(f"    GPU            : {gpu_info['name']}")
    print(f"    VRAM total     : {gpu_info['vram_total_gb']:.2f} GB")

    vram_total_gb = gpu_info["vram_total_gb"]
    vram_budget_gb = args.vram_budget if args.vram_budget else vram_total_gb - 1.0
    if vram_budget_gb > vram_total_gb:
        print(f"    [!] --vram-budget {vram_budget_gb:.2f} GB exceeds total "
              f"{vram_total_gb:.2f} GB", file=sys.stderr)
        return 2
    print(f"    VRAM budget    : {vram_budget_gb:.2f} GB")

    # 2. Load model, verify CPU memory
    print("\n[2] Loading Alpamayo-R1-10B (CPU)...")
    model, vlm_layers = _load_alpamayo()
    weights_total_gb = profiler.get_model_weight_size_gb(model)
    print(f"    Model weights total : {weights_total_gb:.2f} GB")
    profiler.verify_cpu_can_hold_weights(model, cpu_dram_gb)
    print("    CPU DRAM check       : OK")

    vlm_layer_size_mb = profiler.get_vlm_layer_size_mb(vlm_layers)
    print(f"    VLM layer count      : {len(vlm_layers)}")
    print(f"    VLM layer size       : {vlm_layer_size_mb:.2f} MB")

    # 3. Move non-VLM essentials to GPU; measure non_vlm_overhead
    print("\n[3] Setting up non-VLM components on GPU...")
    profiler.measure_current_vram_gb(args.device)
    torch.cuda.reset_peak_memory_stats(args.device)
    _setup_non_vlm_essentials(model)
    torch.cuda.synchronize()
    non_vlm_vram_gb = profiler.measure_current_vram_gb(args.device)
    print(f"    Non-VLM VRAM         : {non_vlm_vram_gb:.2f} GB")

    # 4. Pin all VLM layers and run sequential DL once (Nr = 0)
    print("\n[4] Sequential Demand Layering (Nr=0) profiling...")
    hook = DoubleBufHook(auto_restart=True)
    all_indices = list(range(len(vlm_layers)))
    hook.pin(vlm_layers, all_indices)
    hook.allocate(hook.max_elements())
    hook.register(vlm_layers, all_indices)

    inputs = _prepare_sample_inputs(model)

    @torch.no_grad()
    def _full_offload_inference():
        hook.start()
        # Two iterations: first as warmup, second timed
        _ = model.sample_trajectories_from_data_with_vlm_rollout(**inputs)
        hook.reset()

    # Warmup
    _full_offload_inference()

    # Timed
    timing = profiler.run_sequential_dl_profile(_full_offload_inference)
    full_offload_time_s = timing["full_offload_time_s"]
    print(f"    Full-offload time    : {full_offload_time_s:.3f} s")
    print(f"    Peak VRAM            : {timing['peak_vram_gb']:.2f} GB")

    # Per-layer DMA estimate (homogeneous layer assumption)
    pcie_bw_gbps = profiler.measure_h2d_bandwidth_gbps(
        size_mb=int(vlm_layer_size_mb), num_trials=5, warmup=2
    )
    vlm_layer_dma_ms = (vlm_layer_size_mb / 1024) / pcie_bw_gbps * 1000
    print(f"    H2D PCIe BW          : {pcie_bw_gbps:.2f} GB/s")
    print(f"    VLM layer DMA        : {vlm_layer_dma_ms:.2f} ms")

    # 5. Residency planning
    print("\n[5] Residency planning...")
    max_possible = profiler.compute_max_resident(
        vram_budget_gb=vram_budget_gb,
        vlm_layer_size_mb=vlm_layer_size_mb,
        non_vlm_overhead_gb=non_vlm_vram_gb,
    )
    num_resident = profiler.apply_conservative_margin(
        max_possible, margin=args.margin, minimum=1
    )
    indices = profiler.interleaved_placement(num_resident, len(vlm_layers))
    print(f"    Max possible         : {max_possible}")
    print(f"    Conservative (-{args.margin})       : {num_resident}")
    print(f"    Resident indices     : {indices}")

    # 6. Predict inference time
    predicted_s = predictor.predict_inference_time(
        num_resident=num_resident,
        full_offload_time_s=full_offload_time_s,
        vlm_layer_dma_ms=vlm_layer_dma_ms,
        num_decode_tokens=args.decode_tokens,
    )
    speedup = args.baseline_time / predicted_s if predicted_s > 0 else float("inf")
    print(f"    Predicted time       : {predicted_s:.3f} s")
    print(f"    Speedup vs baseline  : {speedup:.2f}× (baseline {args.baseline_time:.2f} s)")

    # 7. Build config & save
    print("\n[6] Saving config...")
    config = cfg.Config(
        system=cfg.SystemConfig(
            gpu_name=gpu_info["name"],
            vram_total_gb=vram_total_gb,
            vram_budget_gb=vram_budget_gb,
            cpu_dram_total_gb=cpu_dram_gb,
        ),
        model=cfg.ModelConfig(
            name="Alpamayo-R1-10B",
            weights_total_gb=weights_total_gb,
            vlm_layers=len(vlm_layers),
            vlm_layer_size_mb=vlm_layer_size_mb,
        ),
        profiling=cfg.ProfilingConfig(
            vlm_layer_dma_ms=vlm_layer_dma_ms,
            vlm_layer_exe_ms=0.0,  # not measured separately in this pass
            vit_layer_dma_ms=0.0,
            vit_layer_exe_ms=0.0,
            non_vlm_overhead_gb=non_vlm_vram_gb,
        ),
        residency=cfg.ResidencyConfig(
            max_possible=max_possible,
            num_resident=num_resident,
            resident_indices=indices,
        ),
        predicted_performance=cfg.PredictedPerformance(
            inference_time_s=predicted_s,
            vit_resident=True,
            expert_resident=False,
        ),
    )
    cfg.save_config(config, args.output)
    print(f"    Config saved to      : {args.output}")

    # Cleanup
    hook.remove()

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
