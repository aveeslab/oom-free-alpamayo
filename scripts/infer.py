"""Run Alpamayo 1.5 inference using a saved residency config.

Usage:
    python scripts/infer.py --config config.json
    python scripts/infer.py --config config.json --num-iterations 5
    python scripts/infer.py --config config.json --input data.json --output traj.json

The script:
    1. Loads the config produced by scripts/profile.py.
    2. Loads Alpamayo 1.5, moves non-VLM modules to GPU.
    3. Loads resident VLM layers (per config) directly to GPU.
    4. Pins remaining VLM layers to host pinned memory and
       installs DoubleBufHook for asynchronous prefetch via DFB.
    5. Runs inference, prints timing, and optionally writes output trajectories.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

_THIS_DIR = str(Path(__file__).resolve().parent)
_PKG_ROOT = str(Path(__file__).resolve().parent.parent)
# Remove the script's own directory from sys.path to avoid namespace-package
# discovery clashes with transformers' lazy loader (see scripts/profile.py).
sys.path[:] = [p for p in sys.path if p != _THIS_DIR]
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from alpamayo_memopt import DoubleBufHook, load_config
from alpamayo_memopt import alpamayo15


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run Alpamayo 1.5 inference with saved residency config."
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
                   help="Optional output path for trajectories (JSON).")
    p.add_argument("--device", type=int, default=0, help="CUDA device index.")
    p.add_argument("--alpamayo-src", type=Path, default=None,
                   help=("Path to Alpamayo 1.5 src directory. Defaults to the "
                         "ALPAMAYO15_SRC env var, then the source path saved "
                         "in config, then an installed alpamayo1_5 package."))
    p.add_argument("--model-id", default=None,
                   help=("Optional model id/path override. Defaults to "
                         "ALPAMAYO15_MODEL_ID, then the config model name."))
    p.add_argument("--model-cache-dir", type=Path, default=None,
                   help=("Optional HuggingFace/Transformers cache_dir. Defaults to "
                         "ALPAMAYO15_MODEL_CACHE_DIR, then the config value."))
    p.add_argument("--model-revision", default=None,
                   help=("Optional model revision passed to from_pretrained. Defaults "
                         "to ALPAMAYO15_MODEL_REVISION, then the config value."))
    p.add_argument("--local-files-only", action=argparse.BooleanOptionalAction,
                   default=None,
                   help=("Pass local_files_only to from_pretrained. Defaults to "
                         "ALPAMAYO15_LOCAL_FILES_ONLY, then the config value."))
    p.add_argument("--attn-implementation", default=None,
                   help="Optional Transformers attention implementation override, e.g. eager.")
    p.add_argument("--clip-id", default=None,
                   help=("physical_ai_av clip id. Defaults to ALPAMAYO15_CLIP_ID, "
                         "then the config value."))
    p.add_argument("--t0-us", type=int, default=None,
                   help=("physical_ai_av timestamp. Defaults to ALPAMAYO15_T0_US, "
                         "then the config value."))
    p.add_argument("--dataset-revision", dest="dataset_revisions", action="append",
                   default=None,
                   help=("physical_ai_av revision candidate. Can be repeated or "
                         "comma-separated. Defaults to ALPAMAYO15_DATASET_REVISIONS, "
                         "then the config value."))
    p.add_argument("--num-traj-samples", type=int, default=1,
                   help="Trajectory samples per inference (default: 1).")
    p.add_argument("--max-generation-length", type=int, default=256,
                   help="VLM max_new_tokens per inference (default: 256).")
    return p.parse_args()


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
    torch.cuda.set_device(args.device)
    device = f"cuda:{args.device}"
    print("=" * 60)
    print("Alpamayo 1.5 Memory Optimizer - Inference")
    print("=" * 60)

    # 1. Load config
    print(f"\n[1] Loading config: {args.config}")
    config = load_config(args.config)
    print(f"    GPU expected         : {config.system.gpu_name}")
    print(f"    VRAM budget          : {config.system.vram_budget_gb:.2f} GB")
    print(f"    Resident layers      : {config.residency.num_resident} "
          f"of {config.model.vlm_layers}")

    # GPU sanity check
    actual_gpu = torch.cuda.get_device_properties(args.device).name
    if actual_gpu != config.system.gpu_name:
        print(f"    [!] Current GPU '{actual_gpu}' differs from config "
              f"'{config.system.gpu_name}'. Performance may vary.")

    # 2. Load model
    config_source_path = config.model.alpamayo_source_path or None
    if config_source_path and not Path(config_source_path).expanduser().exists():
        print(f"    [!] Config source path not found, ignoring: {config_source_path}")
        config_source_path = None
    alpamayo_src = alpamayo15.resolve_alpamayo15_source_path(
        args.alpamayo_src,
        config_source_path=config_source_path,
    )
    model_id = (
        args.model_id
        or os.environ.get(alpamayo15.ENV_MODEL_ID)
        or config.model.name
        or alpamayo15.FALLBACK_MODEL_ID
    )
    model_cache_dir = (
        args.model_cache_dir
        or alpamayo15.DEFAULT_MODEL_CACHE_DIR
        or (Path(config.model.model_cache_dir).expanduser()
            if config.model.model_cache_dir else None)
    )
    model_revision = (
        args.model_revision
        or os.environ.get(alpamayo15.ENV_MODEL_REVISION)
        or config.model.model_revision
        or None
    )
    if args.local_files_only is not None:
        local_files_only = args.local_files_only
    elif os.environ.get(alpamayo15.ENV_LOCAL_FILES_ONLY) is not None:
        local_files_only = alpamayo15.DEFAULT_LOCAL_FILES_ONLY
    else:
        local_files_only = config.model.local_files_only
    attn_implementation = (
        args.attn_implementation
        or os.environ.get(alpamayo15.ENV_ATTN_IMPLEMENTATION)
        or config.model.attn_implementation
        or None
    )
    clip_id = (
        args.clip_id
        or os.environ.get(alpamayo15.ENV_CLIP_ID)
        or config.model.clip_id
        or alpamayo15.FALLBACK_CLIP_ID
    )
    if args.t0_us is not None:
        t0_us = args.t0_us
    elif os.environ.get(alpamayo15.ENV_T0_US) is not None:
        t0_us = alpamayo15.DEFAULT_T0_US
    else:
        t0_us = config.model.t0_us or alpamayo15.FALLBACK_T0_US
    if (
        args.dataset_revisions is not None
        or os.environ.get(alpamayo15.ENV_DATASET_REVISIONS)
        or os.environ.get(alpamayo15.ENV_DATASET_REVISION)
    ):
        dataset_revisions = alpamayo15.resolve_dataset_revisions(args.dataset_revisions)
    elif config.model.dataset_revisions:
        dataset_revisions = tuple(config.model.dataset_revisions)
    else:
        dataset_revisions = alpamayo15.resolve_dataset_revisions(())

    print("\n[2] Loading Alpamayo 1.5 (CPU)...")
    model, vlm_layers, expert_layers = alpamayo15.load_alpamayo15(
        source_path=alpamayo_src,
        model_id=model_id,
        attn_implementation=attn_implementation,
        cache_dir=model_cache_dir,
        revision=model_revision,
        local_files_only=local_files_only,
    )
    if len(vlm_layers) != config.model.vlm_layers:
        print(f"    [!] VLM layer count mismatch: model has {len(vlm_layers)}, "
              f"config expects {config.model.vlm_layers}", file=sys.stderr)
        return 2
    if config.model.expert_layers and len(expert_layers) != config.model.expert_layers:
        print(f"    [!] Expert layer count mismatch: model has {len(expert_layers)}, "
              f"config expects {config.model.expert_layers}", file=sys.stderr)
        return 2

    # 3. Setup non-VLM modules on GPU
    print("\n[3] Setting up non-VLM components on GPU...")
    alpamayo15.setup_non_layer_components(model, device=device)

    # 4. Move resident VLM layers to GPU
    resident_indices = set(config.residency.resident_indices)
    offload_indices = sorted(set(range(len(vlm_layers))) - resident_indices)
    print(f"\n[4] Loading {len(resident_indices)} resident VLM layers to GPU...")
    for i in resident_indices:
        vlm_layers[i].to(device)

    # 5. Pin offload layers + DoubleBufHook
    print(f"\n[5] Pinning {len(offload_indices)} offload VLM layers + "
          "installing DoubleBufHook...")
    vlm_hook = DoubleBufHook(auto_restart=True, device=device)
    vlm_hook.pin(vlm_layers, offload_indices)
    vlm_hook.allocate(vlm_hook.max_elements())
    vlm_hook.register(vlm_layers, offload_indices)
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
    inputs = alpamayo15.prepare_default_inputs(
        model,
        alpamayo_src,
        device=device,
        clip_id=clip_id,
        t0_us=t0_us,
        dataset_revisions=dataset_revisions,
    )

    # 7. Run inference (warmup + timed iterations)
    print(f"\n[7] Running inference: {args.warmup} warmup + "
          f"{args.num_iterations} timed iteration(s)...")

    @torch.no_grad()
    def _run_once():
        vlm_hook.start()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model.sample_trajectories_from_data_with_vlm_rollout(
                data=inputs,
                num_traj_samples=args.num_traj_samples,
                max_generation_length=args.max_generation_length,
            )
        vlm_hook.reset()
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

    # 8. Save output if requested
    if args.output is not None and last_out is not None:
        print(f"\n[8] Saving output to {args.output}...")
        _serialize_output(last_out, args.output)

    # Cleanup
    vlm_hook.remove()

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
