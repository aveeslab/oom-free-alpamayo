"""Profile the host + an Alpamayo model and emit a residency config.json.

One framework, two models. Pick the model with ``--model {r1,r15}`` (default:
r15). The memory-optimization pipeline is shared; only model loading and
input preparation differ per version.

Usage:
    python scripts/profile.py                              # Alpamayo 1.5 (default)
    python scripts/profile.py --model r1                   # Alpamayo-R1
    python scripts/profile.py --vram-budget 12.0 -o config.json
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
# Drop the script's own dir so the transformers lazy loader doesn't trip on it.
sys.path[:] = [p for p in sys.path if p != str(_HERE)]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from alpamayo_memopt import config as cfg          # noqa: E402
from alpamayo_memopt import gpu, profiler          # noqa: E402
from alpamayo_memopt.models import (               # noqa: E402
    ADAPTER_CHOICES, DEFAULT_KIND, TriHookPipeline, get_adapter,
)


def build_parser() -> tuple[argparse.ArgumentParser, object]:
    """Two-stage parse: resolve --model, then add that adapter's options."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--model", choices=ADAPTER_CHOICES, default=DEFAULT_KIND)
    pre_args, _ = pre.parse_known_args()
    adapter = get_adapter(pre_args.model)

    p = argparse.ArgumentParser(
        description="Profile system + Alpamayo model -> residency config.")
    p.add_argument("--model", choices=ADAPTER_CHOICES, default=DEFAULT_KIND,
                   help=f"Alpamayo version to profile (default: {DEFAULT_KIND}).")
    p.add_argument("--output", "-o", type=Path, default=Path("config.json"),
                   help="Output config JSON path (default: ./config.json).")
    p.add_argument("--vram-budget", type=float, default=None,
                   help="Allowed VRAM in GB. Default: total VRAM - 1 GB.")
    p.add_argument("--margin", type=int, default=2,
                   help="Conservative resident-count margin (default: 2).")
    p.add_argument("--device", type=int, default=0, help="CUDA device index.")
    p.add_argument("--num-traj-samples", type=int, default=None,
                   help="Trajectory samples while profiling (default: model-specific).")
    p.add_argument("--max-generation-length", type=int, default=None,
                   help="VLM max_new_tokens while profiling (default: model-specific).")
    p.add_argument("--lock-clock", action=argparse.BooleanOptionalAction, default=True,
                   help="Lock the GPU graphics clock for reproducible timing "
                        "(default: on; needs sudo).")
    p.add_argument("--max-clock", type=int, default=None,
                   help="Graphics clock to lock in MHz (default: max supported).")

    adapter.add_args(p)
    return p, adapter


def main() -> int:
    parser, adapter = build_parser()
    args = parser.parse_args()
    if args.num_traj_samples is None:
        args.num_traj_samples = adapter.default_num_traj_samples
    if args.max_generation_length is None:
        args.max_generation_length = adapter.default_max_generation_length

    torch.cuda.set_device(args.device)
    device = f"cuda:{args.device}"

    # Quiet HuggingFace shard-loading bars / advisory logs for clean output.
    try:
        from transformers.utils import logging as _hf
        _hf.set_verbosity_error(); _hf.disable_progress_bar()
    except Exception:
        pass

    bar = "═" * 60
    print(bar)
    print(f"  oom-free-alpamayo · Profiler · {adapter.display_name}")
    print(bar)

    # [1/5] System
    cpu_dram_gb = profiler.detect_cpu_dram_gb()
    gpu_info = profiler.detect_gpu_info(args.device)
    vram_total_gb = gpu_info["vram_total_gb"]
    vram_budget_gb = args.vram_budget if args.vram_budget else vram_total_gb - 1.0
    print("\n[1/5] System")
    print(f"      GPU         : {gpu_info['name']} ({vram_total_gb:.2f} GB)")
    if vram_budget_gb > vram_total_gb:
        print(f"      [!] --vram-budget {vram_budget_gb:.2f} GB exceeds total "
              f"{vram_total_gb:.2f} GB", file=sys.stderr)
        return 2
    print(f"      VRAM budget : {vram_budget_gb:.2f} GB")
    print(f"      CPU DRAM    : {cpu_dram_gb:.2f} GB")

    if args.lock_clock:
        gpu.lock_gpu_clock(args.max_clock, args.device)

    # [2/5] Model
    torch.cuda.empty_cache(); gc.collect()
    loaded = adapter.load(args)
    n_vlm = len(loaded.vlm_layers)
    vlm_layer_size_mb = profiler.get_vlm_layer_size_mb(loaded.vlm_layers)
    vit_size_mb = (profiler.get_vlm_layer_size_mb(loaded.vit_blocks)
                   if loaded.vit_blocks else 0.0)
    exp_size_mb = (profiler.get_vlm_layer_size_mb(loaded.expert_layers)
                   if loaded.expert_layers else 0.0)
    profiler.verify_cpu_can_hold_weights(loaded.model, cpu_dram_gb)
    print(f"\n[2/5] Model · {adapter.display_name}")
    print(f"      Weights     : {loaded.weights_total_gb:.2f} GB  (fits CPU DRAM)")
    print(f"      VLM         : {n_vlm} layers × {vlm_layer_size_mb:.1f} MB"
          f" = {n_vlm * vlm_layer_size_mb / 1024:.2f} GB")
    print(f"      ViT         : {len(loaded.vit_blocks)} blocks × {vit_size_mb:.1f} MB"
          f" = {len(loaded.vit_blocks) * vit_size_mb / 1024:.2f} GB")
    print(f"      Expert      : {len(loaded.expert_layers)} layers × {exp_size_mb:.1f} MB"
          f" = {len(loaded.expert_layers) * exp_size_mb / 1024:.2f} GB")

    # [3/5] Profiling — essentials to GPU, then sequential demand layering (Nr=0).
    adapter.setup_essentials(loaded, device)
    pipeline = TriHookPipeline(loaded.vlm_layers, loaded.vit_blocks,
                               loaded.expert_layers, vlm_resident=[], device=device)
    inputs = adapter.prepare_inputs(loaded, args, device)

    def run_once():
        pipeline.start_iteration()
        with torch.no_grad():
            adapter.run(loaded, inputs, args)
        torch.cuda.synchronize(args.device)

    print("\n[3/5] Profiling · sequential demand layering (Nr=0)")
    run_once()  # warmup
    timing = profiler.run_sequential_dl_profile(run_once, args.device)
    full_offload_time_s = timing["full_offload_time_s"]
    peak_vram_gb = timing["peak_vram_gb"]
    print(f"      Full-offload time : {full_offload_time_s:.2f} s")
    print(f"      Peak VRAM         : {peak_vram_gb:.2f} GB")
    pipeline.remove()
    del inputs
    torch.cuda.empty_cache(); gc.collect()

    # [4/5] Residency plan. Making k layers resident adds k*layer to the
    # Nr=0 peak, so max k = (budget - peak) / layer_size.
    available_gb = max(0.0, vram_budget_gb - peak_vram_gb)
    max_possible = min(int(available_gb * 1024 / vlm_layer_size_mb), n_vlm - 1)
    minimum = 1 if max_possible > 0 else 0
    num_resident = max(minimum, max_possible - args.margin)
    indices = profiler.interleaved_placement(num_resident, n_vlm)
    shown = ", ".join(str(i) for i in indices[:12]) + (", …" if len(indices) > 12 else "")
    print("\n[4/5] Residency plan")
    print(f"      Resident VLM : {num_resident} / {n_vlm}  "
          f"(max fit {max_possible}, margin −{args.margin})")
    print(f"      Indices      : [{shown}]")

    # [5/5] Save config
    model_cfg = cfg.ModelConfig(
        name=loaded.settings.get("model_id", adapter.default_model_id),
        kind=adapter.kind,
        weights_total_gb=loaded.weights_total_gb,
        vlm_layers=n_vlm,
        vlm_layer_size_mb=vlm_layer_size_mb,
        expert_layers=len(loaded.expert_layers),
        expert_layer_size_mb=exp_size_mb,
        **adapter.extra_model_config(loaded, args),
    )
    config = cfg.Config(
        system=cfg.SystemConfig(
            gpu_name=gpu_info["name"], vram_total_gb=vram_total_gb,
            vram_budget_gb=vram_budget_gb, cpu_dram_total_gb=cpu_dram_gb,
        ),
        model=model_cfg,
        profiling=cfg.ProfilingConfig(
            vlm_layer_dma_ms=0.0, vlm_layer_exe_ms=0.0,
            vit_layer_dma_ms=0.0, vit_layer_exe_ms=0.0,
            non_vlm_overhead_gb=peak_vram_gb,
        ),
        residency=cfg.ResidencyConfig(
            max_possible=max_possible, num_resident=num_resident,
            resident_indices=indices,
        ),
    )
    cfg.save_config(config, args.output)
    print(f"\n[5/5] Saved → {args.output}")

    print("\n" + "─" * 60)
    print(f"  ✓ Ready · resident {num_resident}/{n_vlm} VLM layers · "
          f"full-offload {full_offload_time_s:.2f} s")
    print(f"    Next: python scripts/infer.py --config {args.output}")
    print("─" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
