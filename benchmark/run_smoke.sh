#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OUT=${OUT:-"$ROOT/reproduced-results/smoke"}
mkdir -p "$OUT" "$ROOT/.torch_extensions"

export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-10.3}
export MAX_JOBS=${MAX_JOBS:-4}
export TORCH_EXTENSIONS_DIR="$ROOT/.torch_extensions"
export TOPK_DISPATCH_SOURCE="$ROOT/benchmark/persistent_topk_extension.cu"
export MAIN_TOPK_INCLUDE="$ROOT/kernels/main-903da60"
export EXACT_TOPK_INCLUDE="$ROOT/kernels/pr-exact-b8f88c1"
export CANDIDATE_TOPK_INCLUDE="$ROOT/kernels/sampled-adaptive-v7"
export CANDIDATE_NAME=sampled_v7
export EXTENSION_SUFFIX=smoke

python3 "$ROOT/benchmark/compare_production_dispatch.py" \
  --run 1 \
  --distribution narrow \
  --widths 4096 32768 32769 131072 \
  --rows 1 32 33 64 \
  --topks 512 1024 2048 \
  --warmups 5 \
  --samples 10 \
  --launches 10 \
  --out "$OUT/smoke.jsonl" \
  2>&1 | tee "$OUT/smoke.console.log"

