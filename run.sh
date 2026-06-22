#!/usr/bin/env bash
# Convenience launcher for scripts/infer.py. Edit the values below or override
# them via environment variables, e.g.  CONFIG=config.json ./run.sh
set -euo pipefail

cd "$(dirname "$0")"

# Required.
PYTHON_BIN="${PYTHON_BIN:-please set this}"
CONFIG="${CONFIG:-please set this}"

# Model: r15 (default) or r1.
MODEL="${MODEL:-r15}"

# Alpamayo 1.5 source path (only needed for --model r15 if alpamayo1_5 is not
# already importable). Ignored for r1.
ALPAMAYO_SRC="${ALPAMAYO_SRC:-${ALPAMAYO15_SRC:-}}"

# Usually-safe defaults.
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-eager}"
DEVICE="${DEVICE:-0}"
WARMUP="${WARMUP:-1}"
NUM_ITERATIONS="${NUM_ITERATIONS:-1}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

require_set() {
  if [[ -z "$2" || "$2" == "please set this" ]]; then
    echo "[run.sh] Set ${1} in run.sh or export it before running." >&2
    exit 2
  fi
}

require_set "PYTHON_BIN" "$PYTHON_BIN"
require_set "CONFIG" "$CONFIG"

[[ -x "$PYTHON_BIN" ]] || { echo "[run.sh] PYTHON_BIN not executable: $PYTHON_BIN" >&2; exit 2; }
[[ -f "$CONFIG" ]]     || { echo "[run.sh] CONFIG not found: $CONFIG" >&2; exit 2; }

export PYTORCH_CUDA_ALLOC_CONF

ARGS=(--model "$MODEL" --config "$CONFIG" --device "$DEVICE"
      --warmup "$WARMUP" --num-iterations "$NUM_ITERATIONS")

if [[ "$MODEL" == "r15" ]]; then
  ARGS+=(--attn-implementation "$ATTN_IMPLEMENTATION")
  [[ -n "$ALPAMAYO_SRC" ]] && ARGS+=(--alpamayo-src "$ALPAMAYO_SRC")
fi

exec "$PYTHON_BIN" scripts/infer.py "${ARGS[@]}" "$@"
