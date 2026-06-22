"""Run Alpamayo inference using a saved residency config.

The model is taken from ``--model`` or, if omitted, from the ``kind`` field of
the config produced by ``scripts/profile.py``.

Usage:
    python scripts/infer.py --config config.json
    python scripts/infer.py --config config.json --num-iterations 5
    python scripts/infer.py --config config.json -o trajectory.json
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path[:] = [p for p in sys.path if p != str(_HERE)]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from alpamayo_memopt import gpu, load_config        # noqa: E402
from alpamayo_memopt.models import (                 # noqa: E402
    ADAPTER_CHOICES, DEFAULT_KIND, TriHookPipeline, get_adapter,
)


def _serialize_output(out, path: Path) -> None:
    """Best-effort JSON serialization of inference output."""
    def to_jsonable(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().tolist()
        if isinstance(x, dict):
            return {k: to_jsonable(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [to_jsonable(v) for v in x]
        return x
    path.write_text(json.dumps(to_jsonable(out), indent=2))


def build_parser_and_config():
    """Resolve model + config first, then add the adapter's options."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--model", choices=ADAPTER_CHOICES, default=None)
    pre.add_argument("--config", "-c", type=Path, default=Path("config.json"))
    pre_args, _ = pre.parse_known_args()

    if not pre_args.config.exists():
        print(f"[!] Config not found: {pre_args.config}. Run scripts/profile.py "
              "first.", file=sys.stderr)
        sys.exit(2)
    config = load_config(pre_args.config)
    kind = pre_args.model or config.model.kind or DEFAULT_KIND
    adapter = get_adapter(kind)

    p = argparse.ArgumentParser(
        description="Run Alpamayo inference with a saved residency config.")
    p.add_argument("--model", choices=ADAPTER_CHOICES, default=kind,
                   help="Alpamayo version (default: config's kind).")
    p.add_argument("--config", "-c", type=Path, default=Path("config.json"),
                   help="Path to config.json from scripts/profile.py.")
    p.add_argument("--num-iterations", "-n", type=int, default=1,
                   help="Timed inference iterations (default: 1).")
    p.add_argument("--warmup", type=int, default=1,
                   help="Untimed warmup iterations (default: 1).")
    p.add_argument("--output", "-o", type=Path, default=None,
                   help="Optional path to save predicted trajectories (JSON).")
    p.add_argument("--input", type=Path, default=None,
                   help="Optional input JSON (default: standard benchmark sample).")
    p.add_argument("--device", type=int, default=0, help="CUDA device index.")
    p.add_argument("--num-traj-samples", type=int, default=None,
                   help="Trajectory samples per inference (default: model-specific).")
    p.add_argument("--max-generation-length", type=int, default=None,
                   help="VLM max_new_tokens per inference (default: model-specific).")
    p.add_argument("--lock-clock", action=argparse.BooleanOptionalAction, default=True,
                   help="Lock the GPU graphics clock (default: on; needs sudo).")
    p.add_argument("--max-clock", type=int, default=None,
                   help="Graphics clock to lock in MHz (default: max supported).")
    adapter.add_args(p)
    return p, adapter, config


def main() -> int:
    parser, adapter, config = build_parser_and_config()
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
    print(f"  oom-free-alpamayo · Inference · {adapter.display_name}")
    print(bar)

    # [1/4] Config
    budget = config.system.vram_budget_gb
    print(f"\n[1/4] Config · {args.config}")
    print(f"      Model    : {adapter.display_name} ({adapter.kind})")
    print(f"      GPU      : {config.system.gpu_name}")
    print(f"      Resident : {config.residency.num_resident} / "
          f"{config.model.vlm_layers} VLM layers")
    actual_gpu = torch.cuda.get_device_properties(args.device).name
    if actual_gpu != config.system.gpu_name:
        print(f"      [!] Current GPU '{actual_gpu}' differs from config "
              f"'{config.system.gpu_name}'; performance may vary.")
    if args.input is not None:
        print("      [!] Custom --input not supported yet; using the default sample.",
              file=sys.stderr)

    if args.lock_clock:
        gpu.lock_gpu_clock(args.max_clock, args.device)

    # [2/4] Load model + install demand-layering pipeline
    print("\n[2/4] Loading model + installing demand-layering pipeline...")
    torch.cuda.empty_cache(); gc.collect()
    loaded = adapter.load(args, config)
    if len(loaded.vlm_layers) != config.model.vlm_layers:
        print(f"      [!] VLM layer count mismatch: model has "
              f"{len(loaded.vlm_layers)}, config expects {config.model.vlm_layers}",
              file=sys.stderr)
        return 2
    adapter.setup_essentials(loaded, device)
    resident = list(config.residency.resident_indices)
    for i in resident:
        loaded.vlm_layers[i].to(device)
    pipeline = TriHookPipeline(loaded.vlm_layers, loaded.vit_blocks,
                               loaded.expert_layers, resident, device=device)
    inputs = adapter.prepare_inputs(loaded, args, device)
    torch.cuda.synchronize(args.device)
    vram_after = torch.cuda.memory_allocated(args.device) / (1024 ** 3)
    print(f"      VRAM in use : {vram_after:.2f} GB / {budget:.2f} GB budget  "
          f"[{'✓' if vram_after <= budget else '!'}]")
    if vram_after > budget:
        print("      [!] Allocated VRAM exceeds the configured budget.",
              file=sys.stderr)

    # [3/4] Run
    print(f"\n[3/4] Running · {args.warmup} warmup + {args.num_iterations} timed")

    def run_once():
        pipeline.start_iteration()
        with torch.no_grad():
            out = adapter.run(loaded, inputs, args)
        torch.cuda.synchronize(args.device)
        return out

    for _ in range(args.warmup):
        run_once()
    torch.cuda.reset_peak_memory_stats(args.device)
    torch.cuda.synchronize(args.device)

    times = []
    last_out = None
    for i in range(args.num_iterations):
        t0 = time.perf_counter()
        last_out = run_once()
        times.append(time.perf_counter() - t0)
        print(f"      iter {i + 1}: {times[-1]:.3f} s")

    # [4/4] Optional output
    if args.output is not None and last_out is not None:
        _serialize_output(last_out, args.output)
        print(f"\n[4/4] Trajectories → {args.output}")

    pipeline.remove()
    if times:
        mean_s = sum(times) / len(times)
        std_s = (sum((t - mean_s) ** 2 for t in times) / len(times)) ** 0.5
        peak_gb = torch.cuda.max_memory_allocated(args.device) / (1024 ** 3)
        print("\n" + "─" * 60)
        print(f"  ✓ {mean_s:.3f} s / inference  (std {std_s:.3f}, n={len(times)}) · "
              f"peak {peak_gb:.2f} GB / {budget:.2f} GB")
        print("─" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
