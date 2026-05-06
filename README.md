# OOM-free Alpamayo 1.5

Memory-efficient inference adapter for **NVIDIA Alpamayo 1.5** using the
existing Alpamayo 1.5 source tree or an installed `alpamayo1_5` package.

This directory is a copy of the small `oom-free-alpamayo` framework, ported so
that model code and model weights are not copied here. At runtime it imports the
Alpamayo 1.5 source path and loads `nvidia/Alpamayo-1.5-10B` through
`from_pretrained`.

## What Changed

- Uses `alpamayo1_5.models.alpamayo1_5.Alpamayo1_5`.
- Defaults to `nvidia/Alpamayo-1.5-10B`.
- Uses the Alpamayo 1.5 example dataset/input formatting, including camera ids.
- Applies demand-layering hooks to `model.vlm.model.language_model.layers`.
- Keeps non-VLM modules, including the Expert module, on GPU.
- Keeps Expert layer count/size in `config.json` as metadata only.
- Removes inference-time prediction and speedup reporting from the runtime
  output and config schema.
- Provides a platform-editable `run.sh` launcher.

## How It Works

This project wraps an existing Alpamayo 1.5 installation instead of vendoring
model code or weights. The runtime flow is:

1. Import `alpamayo1_5` from `--alpamayo-src`, `ALPAMAYO15_SRC`, or the active
   Python environment.
2. Load `Alpamayo1_5.from_pretrained(...)` on CPU.
3. Move non-VLM modules, including the Expert module, to GPU.
4. Keep selected VLM transformer layers resident on GPU.
5. Keep the remaining VLM layers in CPU pinned memory.
6. Use `DoubleBufHook` to prefetch offloaded VLM layers into a double GPU
   buffer immediately before each layer forward.
7. Run the original Alpamayo 1.5
   `sample_trajectories_from_data_with_vlm_rollout(...)` method.

The two user-facing scripts match that flow:

- `scripts/profile.py` measures the current machine and writes a residency
  config.
- `scripts/infer.py` loads that config and runs OOM-free inference.
- `run.sh` is a thin launcher around `scripts/infer.py`.

Run `scripts/profile.py` again on each target platform. GPU VRAM, PCIe
bandwidth, attention peak memory, and CPU memory differ across machines, so a
config generated on one platform is not guaranteed to fit another one.

The config is still required, but it is no longer used for predicted inference
time. It stores the actual runtime placement and platform settings: VRAM
budget, VLM resident layer indices, model/source path, and input defaults.

## Install

From this directory:

```bash
pip install -e .
```

Alpamayo 1.5 dependencies still need to be installed in the same environment.
No local model/source paths are required in this repository. Set them per
machine with environment variables or command-line options:

```bash
export ALPAMAYO15_SRC=/path/to/alpamayo1.5/src
export ALPAMAYO15_MODEL_ID=nvidia/Alpamayo-1.5-10B
export ALPAMAYO15_MODEL_CACHE_DIR=/path/to/hf/cache
export ALPAMAYO15_ATTN_IMPLEMENTATION=eager
export ALPAMAYO15_CLIP_ID=030c760c-ae38-49aa-9ad8-f5650a545d26
export ALPAMAYO15_T0_US=5100000
export ALPAMAYO15_DATASET_REVISIONS=2ae73f49ffd2b5db43b404201beb7b92889f7afc,37a7cc2c868d684d0456b5412a7ec5d18597a96a
```

or per command:

```bash
python3 scripts/profile.py --alpamayo-src /path/to/alpamayo1.5/src
```

If `alpamayo1_5` is already importable in the active Python environment,
`--alpamayo-src`/`ALPAMAYO15_SRC` can be omitted.

## Apply on a New Platform

1. Make Alpamayo 1.5 available in the Python environment.

```bash
cd /path/to/oom-free-alpamayo1_5
pip install -e .
```

2. Set platform-specific paths and defaults.

```bash
export ALPAMAYO15_SRC=/path/to/alpamayo1.5/src
export ALPAMAYO15_MODEL_ID=nvidia/Alpamayo-1.5-10B
export ALPAMAYO15_MODEL_CACHE_DIR=/path/to/hf/cache
export ALPAMAYO15_ATTN_IMPLEMENTATION=eager
```

`env.example` contains the full set of optional variables, including dataset
clip id, timestamp, dataset revisions, model revision, and local-only loading.

3. Profile the target GPU and write a config.

```bash
python3 scripts/profile.py \
  --output config.json \
  --margin 12
```

Use a larger `--margin` when inference still OOMs during attention or rollout.
The margin reduces the number of resident VLM layers and leaves more VRAM for
temporary tensors.

4. Edit `run.sh` for the target platform.

Open `run.sh` and replace every required `please set this` value:

```bash
PYTHON_BIN="${PYTHON_BIN:-please set this}"
ALPAMAYO_SRC="${ALPAMAYO_SRC:-${ALPAMAYO15_SRC:-please set this}}"
CONFIG="${CONFIG:-please set this}"
```

For example:

```bash
PYTHON_BIN="${PYTHON_BIN:-/path/to/venv/bin/python}"
ALPAMAYO_SRC="${ALPAMAYO_SRC:-/path/to/alpamayo1.5/src}"
CONFIG="${CONFIG:-config.json}"
```

If any required value is still `please set this`, `run.sh` exits before running
inference and prints which value must be set. The same values can also be
overridden without editing the file:

```bash
PYTHON_BIN=/path/to/venv/bin/python \
ALPAMAYO_SRC=/path/to/alpamayo1.5/src \
CONFIG=config.json \
./run.sh
```

`ALPAMAYO_SRC` has priority over `ALPAMAYO15_SRC`. If `ALPAMAYO15_SRC` is
already exported, `run.sh` can use it without duplicating the source path.

5. Run OOM-free inference.

```bash
./run.sh
```

The launcher defaults to:

```bash
ATTN_IMPLEMENTATION=eager
DEVICE=0
WARMUP=0
NUM_ITERATIONS=1
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

Override these per run as needed:

```bash
NUM_ITERATIONS=10 ./run.sh
CONFIG=config.margin12.json ./run.sh
DEVICE=1 ./run.sh
```

6. Optional: run the full-GPU baseline to confirm the original path OOMs.

```bash
python3 test_inference.py
```

`test_inference.py` intentionally uses the original full-GPU model path. It is
for baseline/OOM checks, not the OOM-free path.

## Profile

```bash
python3 scripts/profile.py --output config.json
```

Useful options:

```bash
python3 scripts/profile.py \
  --output config.json \
  --vram-budget 15.0 \
  --attn-implementation eager \
  --model-cache-dir /path/to/hf/cache \
  --num-traj-samples 1 \
  --max-generation-length 256
```

The planner only applies residency to VLM layers. Expert layers are not part of
the residency/offload planner.

## Infer

Recommended launcher:

```bash
./run.sh
```

Direct invocation:

```bash
python3 scripts/infer.py --config config.json
```

Multiple timed iterations:

```bash
python3 scripts/infer.py --config config.json --num-iterations 5
```

Save trajectories:

```bash
python3 scripts/infer.py --config config.json --output trajectory.json
```

Inference output reports measured iteration times only. Predicted inference
time, predicted-vs-actual delta, and speedup-vs-baseline reporting have been
removed.

## Notes

- This repo does not copy Alpamayo 1.5 source, datasets, notebooks, or model
  weights.
- Full inference still requires CUDA, HuggingFace access to the gated model and
  dataset, and enough CPU DRAM to hold the model weights.
