#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Edit these values for each platform, or override them with environment vars.
PYTHON_BIN="${PYTHON_BIN:-please set this}"
ALPAMAYO_SRC="${ALPAMAYO_SRC:-${ALPAMAYO15_SRC:-please set this}}"
CONFIG="${CONFIG:-please set this}"

# Usually safe defaults.
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-eager}"
DEVICE="${DEVICE:-0}"
WARMUP="${WARMUP:-0}"
NUM_ITERATIONS="${NUM_ITERATIONS:-1}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

require_set() {
  local name="$1"
  local value="$2"
  if [[ -z "$value" || "$value" == "please set this" ]]; then
    echo "[run.sh] Set ${name} in run.sh or export it before running." >&2
    exit 2
  fi
}

require_set "PYTHON_BIN" "$PYTHON_BIN"
require_set "ALPAMAYO_SRC" "$ALPAMAYO_SRC"
require_set "CONFIG" "$CONFIG"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[run.sh] PYTHON_BIN is not executable: $PYTHON_BIN" >&2
  exit 2
fi

if [[ ! -d "$ALPAMAYO_SRC" ]]; then
  echo "[run.sh] ALPAMAYO_SRC directory not found: $ALPAMAYO_SRC" >&2
  exit 2
fi

if [[ ! -f "$CONFIG" ]]; then
  echo "[run.sh] CONFIG file not found: $CONFIG" >&2
  exit 2
fi

export PYTORCH_CUDA_ALLOC_CONF

exec "$PYTHON_BIN" scripts/infer.py \
  --config "$CONFIG" \
  --alpamayo-src "$ALPAMAYO_SRC" \
  --attn-implementation "$ATTN_IMPLEMENTATION" \
  --device "$DEVICE" \
  --warmup "$WARMUP" \
  --num-iterations "$NUM_ITERATIONS" \
  "$@"
