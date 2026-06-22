<div align="center">

# 🏔️ oom-free-alpamayo

**Run NVIDIA Alpamayo 1.5 on a 12 GB GPU — no quantization, no pruning, no accuracy loss.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org)
[![IEEE RTCSA 2026](https://img.shields.io/badge/IEEE%20RTCSA-2026-success.svg)](#-citation)
[![arXiv](https://img.shields.io/badge/arXiv-2605.11678-b31b1b.svg)](https://arxiv.org/abs/2605.11678)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/aveeslab/oom-free-alpamayo/pulls)

*OOM-free inference for the NVIDIA Alpamayo 1.5 Vision-Language-Action (VLA) model via CPU–GPU memory swapping.*

[**Quick Start**](#-quick-start) · [**How It Works**](#-how-it-works) · [**Benchmarks**](#-benchmarks) · [**Citation**](#-citation)

</div>

---

## 🚀 Overview

Alpamayo-1.5-10B needs **~21.5 GB of VRAM** — out of reach for consumer GPUs. `oom-free-alpamayo` fits it onto **12–16 GB cards** using **CPU–GPU memory swapping at layer granularity**, so the model runs *bit-for-bit identically* to the full-VRAM baseline. Alpamayo-R1 is also supported through the same CLI.

```bash
pip install -e .
python scripts/profile.py -o config.json          # Alpamayo 1.5 (default), one-time per machine
python scripts/infer.py   -c config.json          # run inference

python scripts/profile.py --model r1 -o r1.json   # or target Alpamayo-R1
```

> ✅ **Accepted at IEEE RTCSA 2026** — *OOM-Free Alpamayo via CPU-GPU Memory Swapping for Vision-Language-Action Models*. [[arXiv]](https://arxiv.org/abs/2605.11678)

---

## 📊 Benchmarks

Measured on **Alpamayo-R1-10B** (the paper's evaluation target). Stock inference runs out of memory on both GPUs; `oom-free-alpamayo` runs the **identical** model within budget.

| Platform | VRAM | Stock | **Ours** | Status |
|---|---:|---:|---:|:--:|
| RTX 5070 Ti | 16 GB | OOM ❌ | **4.09 s** | ✅ |
| RTX 3080 Ti | 12 GB | OOM ❌ | **15.46 s** | ✅ |

*Up to **3.55× speedup over Accelerate offloading** while keeping full precision. Alpamayo 1.5 uses the same pipeline and is the CLI default; its numbers will be added as they are finalized.*

---

## ✨ How It Works

Three coordinated optimizations turn a 21.5 GB model into a 12 GB-friendly one:

| # | Technique | What it does |
|:-:|---|---|
| 1️⃣ | **Sequential Demand Layering** | Drops VRAM granularity from *model-level* (21.5 GB) to *layer-level* by streaming layer weights on demand. |
| 2️⃣ | **Pipelined Demand Layering** | A two-slot GPU buffer + a dedicated prefetch CUDA stream hide H2D parameter transfer behind layer compute. |
| 3️⃣ | **GPU-Resident Layer Decision Policy** | Permanently keeps the most beneficial VLM layers on-GPU via *interleaved residency placement* (paper Eq. 8), killing the residual transfer cost that pipelining alone can't hide. |

> 🎯 A one-time **offline profiling run** measures each module's residency benefit and picks the resident-layer count + placement that fit your VRAM budget — automatically, per machine.

---

## 📦 Requirements

- NVIDIA GPU with **≥ 12 GB VRAM** (tested: RTX 3080 Ti 12 GB, RTX 5070 Ti 16 GB)
- CUDA 12.x · Python ≥ 3.10 · PyTorch ≥ 2.0
- CPU DRAM large enough to hold full model weights (**≥ 22 GB** recommended)
- **NVIDIA Alpamayo source + 10B weights**, installed separately (see below):
  - **Alpamayo 1.5** (default) — source reachable via `ALPAMAYO15_SRC` / `--alpamayo-src` / an installed `alpamayo1_5` package
  - **Alpamayo-R1** — the `alpamayo_r1` package + `nvidia/Alpamayo-R1-10B` weights

This repo **depends on but does not redistribute** the Alpamayo source or weights.

---

## ⚡ Quick Start

### 1. Install the Alpamayo model you target (external dependency)

<details open>
<summary><b>Alpamayo 1.5 (default)</b></summary>

Make the Alpamayo 1.5 source importable and copy the env template:

```bash
cp env.example .env        # then edit: ALPAMAYO15_SRC, model cache, etc.
# or: export ALPAMAYO15_SRC=/path/to/alpamayo1.5/src
```
</details>

<details>
<summary><b>Alpamayo-R1</b></summary>

Follow NVIDIA's instructions to install the `alpamayo_r1` package and download the `nvidia/Alpamayo-R1-10B` weights. No env vars needed.
</details>

### 2. Install this framework

```bash
git clone https://github.com/aveeslab/oom-free-alpamayo.git
cd oom-free-alpamayo
pip install -e .          # add ".[dev]" for pytest + ruff
```

### 3. Profile once → infer many times

```bash
# Step 1 — profile your machine, emit config.json (one-time)
python scripts/profile.py --output config.json            # Alpamayo 1.5 (default)
python scripts/profile.py --model r1 --output config.json # or Alpamayo-R1

# Step 2 — run inference (model auto-detected from config)
python scripts/infer.py --config config.json
python scripts/infer.py --config config.json --num-iterations 5   # timing stats
```

> 🔒 Both commands **lock the GPU graphics clock by default** for reproducible timing (needs `sudo`; you'll be prompted for your machine's password). Add `--no-lock-clock` to skip it.

<details>
<summary><b>📋 Example profiler output</b></summary>

```text
============================================================
Alpamayo 1.5 Memory Optimizer - Profiler
============================================================

[1] Detecting system specifications...
    CPU DRAM total : 32.00 GB
    GPU            : NVIDIA GeForce RTX 5070 Ti
    VRAM total     : 16.30 GB
    VRAM budget    : 15.30 GB

[2] Loading Alpamayo 1.5 (CPU)...
    Model weights total : 21.52 GB
    CPU DRAM check      : OK
    VLM layers          : 36 x 348.39 MB = 12.25 GB
    ViT blocks          : 27 x 25.18 MB = 0.66 GB
    Expert layers       : 6 x 86.30 MB = 0.51 GB

[3] Moving non-layer essentials to GPU...

[4] Sequential Demand Layering (Nr=0) profiling...
    Full-offload time   : 14.520 s
    Peak VRAM (Nr=0)    : 3.84 GB

[5] Residency planning...
    Max possible        : 31
    Conservative (-2)    : 29
    Resident indices    : [0, 1, 3, 5, 6, ...]

[6] Saving config...
    Config saved to     : config.json

Done.
```

</details>

### Common CLI options

Shared by both `profile.py` and `infer.py`:

| Option | Default | Description |
|---|---|---|
| `--model` | `r15` | Alpamayo version: `r15` (1.5) or `r1` |
| `--device` | `0` | CUDA device index |
| `--lock-clock` / `--no-lock-clock` | on | Lock GPU graphics clock for reproducible timing (`sudo`) |
| `--max-clock` | max supported | Graphics clock to lock, in MHz |

**`profile.py`** also: `--output/-o` (`config.json`), `--vram-budget` (`total − 1 GB`), `--margin` (`2`).
**`infer.py`** also: `--config/-c` (`config.json`), `--num-iterations/-n` (`1`), `--warmup` (`1`), `--output/-o` (save trajectories).

**Model-specific (Alpamayo 1.5):** `--alpamayo-src`, `--model-id`, `--model-cache-dir`, `--model-revision`, `--attn-implementation`, `--clip-id`, `--t0-us`, `--dataset-revision` — each also reads its `ALPAMAYO15_*` env var (see `env.example`). Run `python scripts/profile.py --model r15 --help` for the full list.

---

## 🧩 Programmatic Use

```python
from alpamayo_memopt import get_adapter, load_config, TriHookPipeline

config = load_config("config.json")
adapter = get_adapter(config.model.kind)        # "r15" or "r1"

loaded = adapter.load(args, config)             # model on CPU + VLM/ViT/Expert layers
adapter.setup_essentials(loaded, "cuda")        # always-resident modules → GPU

resident = config.residency.resident_indices
for i in resident:                              # keep planned layers on the GPU
    loaded.vlm_layers[i].to("cuda")

pipeline = TriHookPipeline(                      # stream the rest on demand
    loaded.vlm_layers, loaded.vit_blocks, loaded.expert_layers,
    vlm_resident=resident, device="cuda",
)
inputs = adapter.prepare_inputs(loaded, args, "cuda")
pipeline.start_iteration()
out = adapter.run(loaded, inputs, args)
pipeline.remove()
```

`args` is an `argparse.Namespace` of the model knobs — see `scripts/profile.py` and `scripts/infer.py` for the canonical end-to-end flow.

---

## 🗂️ Project Structure

```text
oom-free-alpamayo/
├── alpamayo_memopt/        # Python package
│   ├── hook.py             # DoubleBufHook — pipelined prefetch w/ double flat buffer
│   ├── models/             # per-version adapters + shared pipeline
│   │   ├── base.py         #   ModelAdapter interface + TriHookPipeline (VLM+ViT+Expert)
│   │   ├── r1.py           #   Alpamayo-R1 adapter
│   │   └── r15.py          #   Alpamayo 1.5 adapter
│   ├── profiler.py         # system / model / bandwidth profiling primitives
│   ├── gpu.py              # optional GPU graphics-clock locking
│   └── config.py           # Config dataclass + JSON I/O
├── scripts/
│   ├── profile.py          # CLI: profiling → config.json
│   └── infer.py            # CLI: inference using config.json
├── run.sh                  # convenience launcher for infer.py
├── env.example             # ALPAMAYO15_* environment template
├── test_inference.py       # full-GPU baseline (Apache-2.0, derived from NVIDIA)
├── pyproject.toml          # package metadata (PEP 621)
├── LICENSE                 # MIT
└── NOTICE                  # third-party attribution (Alpamayo, Apache 2.0)
```

---

## 📄 Citation

```bibtex
@inproceedings{roh2026alpamayo,
  title     = {OOM-Free Alpamayo via CPU-GPU Memory Swapping for
               Vision-Language-Action Models},
  author    = {Roh, Seungwoo and Kim, Huiyeong and Kim, Jong-Chan},
  booktitle = {Proc. 32nd IEEE Int. Conf. on Embedded and Real-Time
               Computing Systems and Applications (RTCSA)},
  year      = {2026},
  eprint        = {2605.11678},
  archivePrefix = {arXiv},
}
```

## 📜 License

Released under the **MIT License** (see [`LICENSE`](LICENSE)). This work depends on, but does not redistribute, NVIDIA Alpamayo source (Apache 2.0) and the Alpamayo-10B weights (NVIDIA's separate model license). See [`NOTICE`](NOTICE).

## 🙏 Acknowledgements

- **AVEES Lab**, Graduate School of Automobile and Mobility, Kookmin University
- **NVIDIA NVlabs** for the Alpamayo models
