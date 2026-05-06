"""Profile system specs + Alpamayo model layers and save a config JSON.

Usage:
    python scripts/profile.py --output config.json
    python scripts/profile.py --vram-budget 12.0 --output config.json

The script:
    1. Detects CPU DRAM total and verifies it can hold the full model weights.
    2. Detects GPU VRAM total. If --vram-budget is omitted, defaults to
       (total - 1.0 GB) safety reserve.
    3. Loads Alpamayo 1.5, moves non-VLM modules to GPU, measures
       non_vlm_overhead_gb (VRAM consumed before VLM layers are loaded).
    4. Runs Sequential Demand Layering (Nr = 0, all VLM layers offloaded)
       once with DoubleBufHook to measure full-offload inference time and
       peak VRAM.
    5. Computes max possible resident layer count, applies a conservative
       margin of -2, and selects resident indices via interleaved placement
       (paper Eq. 8: first layer included, last layer excluded).
    6. Writes the config JSON.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# Ensure the package is importable when run as a script.
_THIS_DIR = str(Path(__file__).resolve().parent)
_PKG_ROOT = str(Path(__file__).resolve().parent.parent)
# Remove the script's own directory from sys.path; otherwise namespace-package
# discovery via this directory can collide with the transformers lazy loader
# and trigger spurious "cannot import name 'GenerationMixin'" errors.
sys.path[:] = [p for p in sys.path if p != _THIS_DIR]
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from alpamayo_memopt import DoubleBufHook
from alpamayo_memopt import alpamayo15
from alpamayo_memopt import config as cfg
from alpamayo_memopt import profiler


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
    p.add_argument("--margin", type=int, default=2,
                   help="Conservative resident-count margin (default: 2).")
    p.add_argument("--device", type=int, default=0, help="CUDA device index.")
    p.add_argument("--alpamayo-src", type=Path, default=None,
                   help=("Path to Alpamayo 1.5 src directory. Defaults to "
                         "ALPAMAYO15_SRC, then an installed alpamayo1_5 package."))
    p.add_argument("--model-id", default=alpamayo15.DEFAULT_MODEL_ID,
                   help=("Model id/path passed to from_pretrained "
                         f"(default: {alpamayo15.DEFAULT_MODEL_ID})"))
    p.add_argument("--model-cache-dir", type=Path, default=alpamayo15.DEFAULT_MODEL_CACHE_DIR,
                   help=("Optional HuggingFace/Transformers cache_dir. Defaults to "
                         "ALPAMAYO15_MODEL_CACHE_DIR if set."))
    p.add_argument("--model-revision", default=alpamayo15.DEFAULT_MODEL_REVISION,
                   help=("Optional model revision passed to from_pretrained. Defaults "
                         "to ALPAMAYO15_MODEL_REVISION if set."))
    p.add_argument("--local-files-only", action=argparse.BooleanOptionalAction,
                   default=alpamayo15.DEFAULT_LOCAL_FILES_ONLY,
                   help=("Pass local_files_only to from_pretrained. Defaults to "
                         "ALPAMAYO15_LOCAL_FILES_ONLY."))
    p.add_argument("--attn-implementation", default=alpamayo15.DEFAULT_ATTN_IMPLEMENTATION,
                   help="Optional Transformers attention implementation override, e.g. eager.")
    p.add_argument("--clip-id", default=alpamayo15.DEFAULT_CLIP_ID,
                   help=("physical_ai_av clip id used for profiling input. Defaults "
                         "to ALPAMAYO15_CLIP_ID."))
    p.add_argument("--t0-us", type=int, default=alpamayo15.DEFAULT_T0_US,
                   help=("physical_ai_av timestamp used for profiling input. Defaults "
                         "to ALPAMAYO15_T0_US."))
    p.add_argument("--dataset-revision", dest="dataset_revisions", action="append",
                   default=None,
                   help=("physical_ai_av revision candidate. Can be repeated or "
                         "comma-separated. Defaults to ALPAMAYO15_DATASET_REVISIONS."))
    p.add_argument("--num-traj-samples", type=int, default=1,
                   help="Trajectory samples used while profiling (default: 1).")
    p.add_argument("--max-generation-length", type=int, default=256,
                   help="VLM max_new_tokens used while profiling (default: 256).")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────
# Profiling pipeline
# ─────────────────────────────────────────────────────────────────────

def main() -> int:
    args = parse_args()
    print("=" * 60)
    print("Alpamayo 1.5 Memory Optimizer - Profiler")
    print("=" * 60)

    # 1. System detection
    print("\n[1] Detecting system specifications...")
    cpu_dram_gb = profiler.detect_cpu_dram_gb()
    gpu_info = profiler.detect_gpu_info(args.device)
    torch.cuda.set_device(args.device)
    device = f"cuda:{args.device}"
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
    print("\n[2] Loading Alpamayo 1.5 (CPU)...")
    resolved_source_path = alpamayo15.ensure_alpamayo15_importable(args.alpamayo_src)
    dataset_revisions = alpamayo15.resolve_dataset_revisions(args.dataset_revisions)
    model, vlm_layers, expert_layers = alpamayo15.load_alpamayo15(
        source_path=resolved_source_path,
        model_id=args.model_id,
        attn_implementation=args.attn_implementation,
        cache_dir=args.model_cache_dir,
        revision=args.model_revision,
        local_files_only=args.local_files_only,
    )
    weights_total_gb = profiler.get_model_weight_size_gb(model)
    print(f"    Model weights total : {weights_total_gb:.2f} GB")
    profiler.verify_cpu_can_hold_weights(model, cpu_dram_gb)
    print("    CPU DRAM check       : OK")

    vlm_layer_size_mb = profiler.get_vlm_layer_size_mb(vlm_layers)
    expert_layer_size_mb = profiler.get_vlm_layer_size_mb(expert_layers)
    print(f"    VLM layer count      : {len(vlm_layers)}")
    print(f"    VLM layer size       : {vlm_layer_size_mb:.2f} MB")
    print(f"    Expert layer count   : {len(expert_layers)}")
    print(f"    Expert layer size    : {expert_layer_size_mb:.2f} MB")

    # 3. Move non-VLM modules to GPU; measure non-VLM overhead
    print("\n[3] Setting up non-VLM components on GPU...")
    profiler.measure_current_vram_gb(args.device)
    torch.cuda.reset_peak_memory_stats(args.device)
    alpamayo15.setup_non_layer_components(model, device=device)
    torch.cuda.synchronize()
    non_vlm_vram_gb = profiler.measure_current_vram_gb(args.device)
    print(f"    Non-VLM VRAM         : {non_vlm_vram_gb:.2f} GB")

    # 4. Pin all VLM layers and run sequential DL once (Nr = 0)
    print("\n[4] Sequential Demand Layering (Nr=0) profiling...")
    vlm_hook = DoubleBufHook(auto_restart=True, device=device)
    all_vlm_indices = list(range(len(vlm_layers)))
    vlm_hook.pin(vlm_layers, all_vlm_indices)
    vlm_hook.allocate(vlm_hook.max_elements())
    vlm_hook.register(vlm_layers, all_vlm_indices)

    inputs = alpamayo15.prepare_default_inputs(
        model,
        resolved_source_path,
        device=device,
        clip_id=args.clip_id,
        t0_us=args.t0_us,
        dataset_revisions=dataset_revisions,
    )

    @torch.no_grad()
    def _full_offload_inference():
        vlm_hook.start()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _ = model.sample_trajectories_from_data_with_vlm_rollout(
                data=inputs,
                num_traj_samples=args.num_traj_samples,
                max_generation_length=args.max_generation_length,
            )
        vlm_hook.reset()

    # Warmup
    _full_offload_inference()

    # Timed
    timing = profiler.run_sequential_dl_profile(_full_offload_inference, args.device)
    full_offload_time_s = timing["full_offload_time_s"]
    print(f"    Full-offload time    : {full_offload_time_s:.3f} s")
    print(f"    Peak VRAM            : {timing['peak_vram_gb']:.2f} GB")

    # 5. Residency planning
    print("\n[5] Residency planning...")
    max_possible_raw = profiler.compute_max_resident(
        vram_budget_gb=vram_budget_gb,
        vlm_layer_size_mb=vlm_layer_size_mb,
        non_vlm_overhead_gb=non_vlm_vram_gb,
    )
    max_possible = min(max_possible_raw, max(len(vlm_layers) - 1, 0))
    minimum_resident = 1 if max_possible > 0 else 0
    num_resident = min(
        profiler.apply_conservative_margin(
            max_possible, margin=args.margin, minimum=minimum_resident
        ),
        max_possible,
    )
    indices = profiler.interleaved_placement(num_resident, len(vlm_layers))
    print(f"    Max possible         : {max_possible}")
    print(f"    Conservative (-{args.margin})       : {num_resident}")
    print(f"    Resident indices     : {indices}")

    # 6. Build config & save
    print("\n[6] Saving config...")
    config = cfg.Config(
        system=cfg.SystemConfig(
            gpu_name=gpu_info["name"],
            vram_total_gb=vram_total_gb,
            vram_budget_gb=vram_budget_gb,
            cpu_dram_total_gb=cpu_dram_gb,
        ),
        model=cfg.ModelConfig(
            name=args.model_id,
            weights_total_gb=weights_total_gb,
            vlm_layers=len(vlm_layers),
            vlm_layer_size_mb=vlm_layer_size_mb,
            expert_layers=len(expert_layers),
            expert_layer_size_mb=expert_layer_size_mb,
            alpamayo_source_path=str(resolved_source_path) if resolved_source_path else "",
            model_cache_dir=str(Path(args.model_cache_dir).expanduser())
            if args.model_cache_dir else "",
            model_revision=args.model_revision or "",
            attn_implementation=args.attn_implementation or "",
            local_files_only=args.local_files_only,
            clip_id=args.clip_id,
            t0_us=args.t0_us,
            dataset_revisions=list(dataset_revisions),
        ),
        profiling=cfg.ProfilingConfig(
            vlm_layer_dma_ms=0.0,
            vlm_layer_exe_ms=0.0,  # not measured separately in this pass
            vit_layer_dma_ms=0.0,
            vit_layer_exe_ms=0.0,
            non_vlm_overhead_gb=non_vlm_vram_gb,
        ),
        residency=cfg.ResidencyConfig(
            max_possible=max_possible,
            num_resident=num_resident,
            resident_indices=indices,
            expert_max_possible=0,
            expert_num_resident=0,
            expert_resident_indices=[],
        ),
    )
    cfg.save_config(config, args.output)
    print(f"    Config saved to      : {args.output}")

    # Cleanup
    vlm_hook.remove()

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
