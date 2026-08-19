#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OUT=${OUT:-"$ROOT/reproduced-results"}
RAW="$OUT/raw"
ANALYSIS="$OUT/analysis"

mkdir -p "$RAW" "$ANALYSIS" "$ROOT/.torch_extensions"

export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-10.3}
export MAX_JOBS=${MAX_JOBS:-4}
export TORCH_EXTENSIONS_DIR="$ROOT/.torch_extensions"
export TOPK_DISPATCH_SOURCE="$ROOT/benchmark/persistent_topk_extension.cu"
export MAIN_TOPK_INCLUDE="$ROOT/kernels/main-903da60"
export EXACT_TOPK_INCLUDE="$ROOT/kernels/pr-exact-b8f88c1"
export CANDIDATE_TOPK_INCLUDE="$ROOT/kernels/sampled-adaptive-v7"
export CANDIDATE_NAME=sampled_v7
export EXTENSION_SUFFIX=production_dispatch

widths=(4096 12288 32767 32768 32769 65536 131072 250000)
rows=(1 8 16 32 33 64 128)
topks=(512 1024 2048)
distributions=(narrow normal wide_uniform)

for run in 1 2 3; do
  for distribution in "${distributions[@]}"; do
    name="run${run}_${distribution}"
    python3 "$ROOT/benchmark/compare_production_dispatch.py" \
      --run "$run" \
      --distribution "$distribution" \
      --widths "${widths[@]}" \
      --rows "${rows[@]}" \
      --topks "${topks[@]}" \
      --warmups 20 \
      --samples 60 \
      --launches 20 \
      --out "$RAW/${name}.jsonl" \
      2>&1 | tee "$RAW/${name}.console.log"
  done
done

python3 "$ROOT/benchmark/render_results.py" \
  --input-dir "$RAW" \
  --source-root "$ROOT" \
  --output-dir "$ANALYSIS"

python3 "$ROOT/benchmark/render_reviewer_table.py" \
  --csv "$ANALYSIS/PRODUCTION_DISPATCH_3RUN_CELLS.csv" \
  --out "$ANALYSIS/REVIEWER_STYLE_TABLE.txt"

tar -C "$OUT" -czf "$OUT/raw-3run.tgz" raw
sha256sum "$OUT/raw-3run.tgz"

