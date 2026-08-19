# Persistent TopK overflow experiment

This repository preserves an experimental sampled-adaptive alternative explored
while reviewing [vLLM PR #52149](https://github.com/vllm-project/vllm/pull/52149).
Thanks to [Roberto Lopez Castro](https://github.com/LopezCastroRoberto) for the
review that motivated the broader performance matrix.

The candidate is **not proposed for merge into PR #52149**. It is correct on
every tested case and substantially faster than the PR's exact fallback on
narrow-overflow inputs, but it adds substantial traversal and dispatch
complexity and regresses parts of the normal persistent path.

## Result

Three independent runs were made on one isolated NVIDIA B300 SXM6 AC. Each
cell used 20 warmups, 60 timing samples, and 20 launches per sample. All three
implementations used the same production dispatcher:

```text
rows > 32 and shared memory >= 128 KiB -> FilteredTopK
otherwise                             -> persistent_topk
```

Percentages below are geometric means of paired kernel-time ratios. Negative
means sampled-adaptive v7 is faster.

| Distribution / dispatch | Cases | MAIN bad rows | PR Exact bad rows | v7 bad rows | v7 vs MAIN | v7 vs PR Exact |
|---|---:|---:|---:|---:|---:|---:|
| Narrow overflow, all | 168 | 18,252 | 0 | 0 | +4.05% | **-34.66%** |
| Narrow, persistent (`rows <= 32`) | 96 | 2,052 | 0 | 0 | +4.41% | **-19.19%** |
| Narrow, FilteredTopK (`rows >= 33`) | 72 | 16,200 | 0 | 0 | +3.56% | **-50.79%** |
| Normal, persistent | 96 | 0 | 0 | 0 | +4.11% | +2.88% |
| Normal, FilteredTopK | 72 | 0 | 0 | 0 | **-2.99%** | +0.51% |
| Wide uniform, persistent | 96 | 0 | 0 | 0 | +4.33% | +2.68% |
| Wide uniform, FilteredTopK | 72 | 0 | 0 | 0 | **-2.83%** | +0.50% |

The strongest stable, correctness-equivalent win over MAIN was
`wide_uniform, width=65,536, rows=64, k=2,048`: **-16.66% / -15.72% /
-17.86%** over the three runs. The worst normal-input regression against PR
Exact was **+11.27%** at `normal, width=12,288, rows=1, k=2,048`.

MAIN is incorrect on narrow-overflow inputs, so its timing there is only a
shipping-baseline reference. The fair fixed-vs-fixed comparison is v7 against
PR Exact.

## Reviewer-style tables

The complete output is intentionally available in several forms:

- [Reviewer-style MAIN comparison](results/REVIEWER_STYLE_TABLE.txt): grouped
  by input distribution and width, with `rows` down the left and
  `k=512/1024/2048` across the page. Each group shows only `MAIN`, `v7`, and
  `v7 vs MAIN`, matching the review table format.
- [Full three-run table](results/FULL_3RUN_TABLE.md): every cell with all three
  run values.
- [CSV](results/CELLS.csv): 504 cells with raw per-run medians, paired ranges,
  and correctness counters.
- [Summary](results/SUMMARY.md) and [validation manifest](results/MANIFEST.json).
- [Raw logs](results/raw-3run.tgz), SHA256
  `6403a7b868d4a069868f6e5ece6d23116d1a2cc9de72d300da86dbce131e142a`.

## Reproduce

Requirements: Linux, an NVIDIA B300-class GPU, NVIDIA Container Toolkit, and
enough local storage for PyTorch CUDA extension builds.

```bash
git clone https://github.com/xijiaat/persistent-topk-overflow-experiment.git
cd persistent-topk-overflow-experiment

docker build -t persistent-topk-overflow-experiment .
docker run --rm --gpus '"device=0"' --ipc=host \
  --ulimit memlock=-1:-1 \
  -v "$PWD:/workspace" \
  persistent-topk-overflow-experiment \
  ./benchmark/run_three_runs.sh
```

Outputs are written to `reproduced-results/`. A single smaller smoke run can
be launched with:

```bash
docker run --rm --gpus '"device=0"' --ipc=host \
  --ulimit memlock=-1:-1 \
  -v "$PWD:/workspace" \
  persistent-topk-overflow-experiment \
  ./benchmark/run_smoke.sh
```

## Compared sources

- MAIN: PR base `903da60`
- PR Exact: PR head `b8f88c1`
- Candidate: sampled-adaptive v7
- PyTorch: `2.13.0+cu130`
- vLLM package: `0.27.0`

The benchmark wrapper queries device properties on each call, unlike the
static production cache. This overhead is identical for all variants, so the
paired relative comparison is valid; the absolute microseconds should not be
treated as exact full-engine latency.

## Maintenance cost

Compared with MAIN, the candidate changes the two kernel headers by
`+1,384/-39` lines (`1,907 -> 3,252`, net **+70.5%**, 38 diff hunks). Compared
with PR Exact it is `+1,069/-5` lines. It adds three exact-selection paths,
touches persistent, medium/decode, and FilteredTopK paths, and introduces
several workload thresholds. This is useful evidence and a working prototype,
but not a small or easy-to-review production patch.
# Persistent TopK overflow experiment

This repository preserves an experimental sampled-adaptive alternative explored
while reviewing [vLLM PR #52149](https://github.com/vllm-project/vllm/pull/52149).
Thanks to [Roberto Lopez Castro](https://github.com/LopezCastroRoberto) for the
review that motivated the broader performance matrix.

The candidate is **not proposed for merge into PR #52149**. It is correct on
every tested case and substantially faster than the PR's exact fallback on
narrow-overflow inputs, but it adds substantial traversal and dispatch
complexity and regresses parts of the normal persistent path.

## Result

Three independent runs were made on one isolated NVIDIA B300 SXM6 AC. Each
cell used 20 warmups, 60 timing samples, and 20 launches per sample. All three
implementations used the same production dispatcher:

```text
rows > 32 and shared memory >= 128 KiB -> FilteredTopK
otherwise                             -> persistent_topk
```

Percentages below are geometric means of paired kernel-time ratios. Negative
means sampled-adaptive v7 is faster.

| Distribution / dispatch | Cases | MAIN bad rows | PR Exact bad rows | v7 bad rows | v7 vs MAIN | v7 vs PR Exact |
|---|---:|---:|---:|---:|---:|---:|
| Narrow overflow, all | 168 | 18,252 | 0 | 0 | +4.05% | **-34.66%** |
| Narrow, persistent (`rows <= 32`) | 96 | 2,052 | 0 | 0 | +4.41% | **-19.19%** |
| Narrow, FilteredTopK (`rows >= 33`) | 72 | 16,200 | 0 | 0 | +3.56% | **-50.79%** |
| Normal, persistent | 96 | 0 | 0 | 0 | +4.11% | +2.88% |
| Normal, FilteredTopK | 72 | 0 | 0 | 0 | **-2.99%** | +0.51% |
| Wide uniform, persistent | 96 | 0 | 0 | 0 | +4.33% | +2.68% |
| Wide uniform, FilteredTopK | 72 | 0 | 0 | 0 | **-2.83%** | +0.50% |

The strongest stable, correctness-equivalent win over MAIN was
`wide_uniform, width=65,536, rows=64, k=2,048`: **-16.66% / -15.72% /
-17.86%** over the three runs. The worst normal-input regression against PR
Exact was **+11.27%** at `normal, width=12,288, rows=1, k=2,048`.

MAIN is incorrect on narrow-overflow inputs, so its timing there is only a
shipping-baseline reference. The fair fixed-vs-fixed comparison is v7 against
PR Exact.

## Reviewer-style tables

The complete output is intentionally available in several forms:

- [Reviewer-style MAIN comparison](results/REVIEWER_STYLE_TABLE.txt): grouped
  by input distribution and width, with `rows` down the left and
  `k=512/1024/2048` across the page. Each group shows only `MAIN`, `v7`, and
  `v7 vs MAIN`, matching the review table format.
- [Full three-run table](results/FULL_3RUN_TABLE.md): every cell with all three
  run values.
- [CSV](results/CELLS.csv): 504 cells with raw per-run medians, paired ranges,
  and correctness counters.
- [Summary](results/SUMMARY.md) and [validation manifest](results/MANIFEST.json).
- [Raw logs](results/raw-3run.tgz), SHA256
  `6403a7b868d4a069868f6e5ece6d23116d1a2cc9de72d300da86dbce131e142a`.

## Reproduce

Requirements: Linux, an NVIDIA B300-class GPU, NVIDIA Container Toolkit, and
enough local storage for PyTorch CUDA extension builds.

```bash
git clone https://github.com/xijiaat/persistent-topk-overflow-experiment.git
cd persistent-topk-overflow-experiment

docker build -t persistent-topk-overflow-experiment .
docker run --rm --gpus '"device=0"' --ipc=host \
  --ulimit memlock=-1:-1 \
  -v "$PWD:/workspace" \
  persistent-topk-overflow-experiment \
  ./benchmark/run_three_runs.sh
```

Outputs are written to `reproduced-results/`. A single smaller smoke run can
be launched with:

```bash
docker run --rm --gpus '"device=0"' --ipc=host \
  --ulimit memlock=-1:-1 \
  -v "$PWD:/workspace" \
  persistent-topk-overflow-experiment \
  ./benchmark/run_smoke.sh
```

## Compared sources

- MAIN: PR base `903da60`
- PR Exact: PR head `b8f88c1`
- Candidate: sampled-adaptive v7
- PyTorch: `2.13.0+cu130`
- vLLM package: `0.27.0`

The benchmark wrapper queries device properties on each call, unlike the
static production cache. This overhead is identical for all variants, so the
paired relative comparison is valid; the absolute microseconds should not be
treated as exact full-engine latency.

## Maintenance cost

Compared with MAIN, the candidate changes the two kernel headers by
`+1,384/-39` lines (`1,907 -> 3,252`, net **+70.5%**, 38 diff hunks). Compared
with PR Exact it is `+1,069/-5` lines. It adds three exact-selection paths,
touches persistent, medium/decode, and FilteredTopK paths, and introduces
several workload thresholds. This is useful evidence and a working prototype,
but not a small or easy-to-review production patch.
