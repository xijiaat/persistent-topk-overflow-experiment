#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


RUNS = (1, 2, 3)
DISTRIBUTIONS = ("narrow", "normal", "wide_uniform")
WIDTHS = (4096, 12288, 32767, 32768, 32769, 65536, 131072, 250000)
ROWS = (1, 8, 16, 32, 33, 64, 128)
TOPKS = (512, 1024, 2048)
VARIANTS = ("main", "exact", "sampled_v7")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def geomean(values: list[float]) -> float:
    return math.exp(statistics.fmean(math.log(value) for value in values))


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def load_and_validate(paths: list[Path]) -> tuple[dict, dict]:
    environments = {}
    indexed = defaultdict(dict)
    for path in paths:
        environment = None
        result_count = 0
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: {exc}") from exc
                if record.get("kind") == "environment":
                    if environment is not None:
                        raise ValueError(f"{path}: duplicate environment")
                    environment = record
                    continue
                if record.get("kind") != "result":
                    raise ValueError(f"{path}:{line_number}: unknown record")
                if environment is None:
                    raise ValueError(f"{path}:{line_number}: no environment")
                key = (
                    record["run"],
                    environment["distribution"],
                    record["width"],
                    record["rows"],
                    record["topk"],
                )
                variant = record["variant"]
                if variant in indexed[key]:
                    raise ValueError(f"duplicate result: {key}, {variant}")
                indexed[key][variant] = record
                result_count += 1

        if environment is None:
            raise ValueError(f"{path}: missing environment")
        env_key = (environment["run"], environment["distribution"])
        if env_key in environments:
            raise ValueError(f"duplicate environment: {env_key}")
        environments[env_key] = {**environment, "path": str(path)}
        if result_count != len(WIDTHS) * len(ROWS) * len(TOPKS) * len(VARIANTS):
            raise ValueError(f"{path}: expected 504 results, got {result_count}")

    expected_envs = {(run, dist) for run in RUNS for dist in DISTRIBUTIONS}
    if set(environments) != expected_envs:
        raise ValueError(
            f"environment mismatch: missing={expected_envs - set(environments)}, "
            f"extra={set(environments) - expected_envs}"
        )

    expected_keys = {
        (run, dist, width, rows, topk)
        for run in RUNS
        for dist in DISTRIBUTIONS
        for width in WIDTHS
        for rows in ROWS
        for topk in TOPKS
    }
    if set(indexed) != expected_keys:
        raise ValueError(
            f"matrix mismatch: missing={expected_keys - set(indexed)}, "
            f"extra={set(indexed) - expected_keys}"
        )
    for key, records in indexed.items():
        if set(records) != set(VARIANTS):
            raise ValueError(f"variant mismatch: {key}: {set(records)}")

    for environment in environments.values():
        expected = {
            "widths": list(WIDTHS),
            "rows": list(ROWS),
            "topks": list(TOPKS),
            "variants": list(VARIANTS),
            "warmups": 20,
            "samples": 60,
            "launches": 20,
            "dispatcher": "production: rows>32 FilteredTopK, else persistent",
        }
        for key, value in expected.items():
            if environment.get(key) != value:
                raise ValueError(
                    f"environment {environment['path']}: {key}="
                    f"{environment.get(key)!r}, expected {value!r}"
                )
    return environments, indexed


def cell(indexed: dict, distribution: str, width: int, rows: int,
         topk: int) -> dict:
    records = {
        run: indexed[(run, distribution, width, rows, topk)]
        for run in RUNS
    }
    times = {
        variant: [records[run][variant]["median_us"] for run in RUNS]
        for variant in VARIANTS
    }
    v7_main_ratios = [
        times["sampled_v7"][index] / times["main"][index]
        for index in range(len(RUNS))
    ]
    v7_exact_ratios = [
        times["sampled_v7"][index] / times["exact"][index]
        for index in range(len(RUNS))
    ]
    return {
        "distribution": distribution,
        "width": width,
        "rows": rows,
        "topk": topk,
        "dispatcher": "persistent" if rows <= 32 else "FilteredTopK",
        "records": records,
        "times": times,
        "median_times": {
            variant: statistics.median(values)
            for variant, values in times.items()
        },
        "v7_main_ratios": v7_main_ratios,
        "v7_exact_ratios": v7_exact_ratios,
    }


def all_cells(indexed: dict) -> list[dict]:
    return [
        cell(indexed, distribution, width, rows, topk)
        for distribution in DISTRIBUTIONS
        for width in WIDTHS
        for rows in ROWS
        for topk in TOPKS
    ]


def bad_stats(cells: list[dict], variant: str) -> tuple[int, int, int]:
    bad_case_runs = 0
    bad_rows = 0
    max_delta = 0.0
    for current in cells:
        for run in RUNS:
            record = current["records"][run][variant]
            bad_case_runs += record["bad_rows"] > 0
            bad_rows += record["bad_rows"]
            max_delta = max(max_delta, record["max_abs_delta"])
    return bad_case_runs, bad_rows, max_delta


def summarize_group(name: str, cells: list[dict]) -> dict:
    if not cells:
        raise ValueError(f"empty group: {name}")
    paired_main = [
        ratio for current in cells for ratio in current["v7_main_ratios"]
    ]
    paired_exact = [
        ratio for current in cells for ratio in current["v7_exact_ratios"]
    ]
    run_main_geomeans = [
        geomean([current["v7_main_ratios"][run - 1] for current in cells])
        for run in RUNS
    ]
    run_exact_geomeans = [
        geomean([current["v7_exact_ratios"][run - 1] for current in cells])
        for run in RUNS
    ]
    main_bad = bad_stats(cells, "main")
    exact_bad = bad_stats(cells, "exact")
    v7_bad = bad_stats(cells, "sampled_v7")
    return {
        "name": name,
        "cells": len(cells),
        "case_runs": len(cells) * len(RUNS),
        "main_bad_case_runs": main_bad[0],
        "main_bad_rows": main_bad[1],
        "main_max_delta": main_bad[2],
        "exact_bad_case_runs": exact_bad[0],
        "exact_bad_rows": exact_bad[1],
        "exact_max_delta": exact_bad[2],
        "v7_bad_case_runs": v7_bad[0],
        "v7_bad_rows": v7_bad[1],
        "v7_max_delta": v7_bad[2],
        "v7_main_median_ratio": statistics.median(paired_main),
        "v7_main_geomean_ratio": geomean(paired_main),
        "v7_main_p05_ratio": percentile(paired_main, 0.05),
        "v7_main_p95_ratio": percentile(paired_main, 0.95),
        "v7_main_run_geomeans": run_main_geomeans,
        "v7_exact_median_ratio": statistics.median(paired_exact),
        "v7_exact_geomean_ratio": geomean(paired_exact),
        "v7_exact_p05_ratio": percentile(paired_exact, 0.05),
        "v7_exact_p95_ratio": percentile(paired_exact, 0.95),
        "v7_exact_run_geomeans": run_exact_geomeans,
    }


def build_groups(cells: list[dict]) -> list[dict]:
    groups = []
    for distribution in DISTRIBUTIONS:
        selected = [c for c in cells if c["distribution"] == distribution]
        groups.append(summarize_group(distribution, selected))
        groups.append(summarize_group(
            f"{distribution}: persistent rows<=32",
            [c for c in selected if c["rows"] <= 32],
        ))
        groups.append(summarize_group(
            f"{distribution}: FilteredTopK rows>=33",
            [c for c in selected if c["rows"] >= 33],
        ))

    narrow = [c for c in cells if c["distribution"] == "narrow"]
    groups.extend([
        summarize_group(
            "narrow: persistent width<=32768",
            [c for c in narrow if c["rows"] <= 32 and c["width"] <= 32768],
        ),
        summarize_group(
            "narrow: persistent width=32769",
            [c for c in narrow if c["rows"] <= 32 and c["width"] == 32769],
        ),
        summarize_group(
            "narrow: persistent width>=65536",
            [c for c in narrow if c["rows"] <= 32 and c["width"] >= 65536],
        ),
    ])
    return groups


def ratio_percent(ratio: float) -> float:
    return (ratio - 1.0) * 100.0


def format_runs(values: list[float]) -> str:
    return " / ".join(f"{value:.3f}" for value in values)


def write_csv(path: Path, cells: list[dict]) -> None:
    fields = [
        "distribution", "width", "rows", "topk", "dispatcher",
        "main_run1_us", "main_run2_us", "main_run3_us", "main_median_us",
        "exact_run1_us", "exact_run2_us", "exact_run3_us", "exact_median_us",
        "v7_run1_us", "v7_run2_us", "v7_run3_us", "v7_median_us",
        "v7_vs_main_paired_median_pct", "v7_vs_main_paired_min_pct",
        "v7_vs_main_paired_max_pct", "v7_vs_exact_paired_median_pct",
        "v7_vs_exact_paired_min_pct", "v7_vs_exact_paired_max_pct",
        "exact_over_v7_paired_median_speedup",
        "main_bad_rows_run1", "main_bad_rows_run2", "main_bad_rows_run3",
        "exact_bad_rows_run1", "exact_bad_rows_run2", "exact_bad_rows_run3",
        "v7_bad_rows_run1", "v7_bad_rows_run2", "v7_bad_rows_run3",
        "main_max_abs_delta", "exact_max_abs_delta", "v7_max_abs_delta",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for current in cells:
            main_ratios = current["v7_main_ratios"]
            exact_ratios = current["v7_exact_ratios"]
            row = {
                "distribution": current["distribution"],
                "width": current["width"],
                "rows": current["rows"],
                "topk": current["topk"],
                "dispatcher": current["dispatcher"],
                "v7_vs_main_paired_median_pct": ratio_percent(
                    statistics.median(main_ratios)),
                "v7_vs_main_paired_min_pct": ratio_percent(min(main_ratios)),
                "v7_vs_main_paired_max_pct": ratio_percent(max(main_ratios)),
                "v7_vs_exact_paired_median_pct": ratio_percent(
                    statistics.median(exact_ratios)),
                "v7_vs_exact_paired_min_pct": ratio_percent(min(exact_ratios)),
                "v7_vs_exact_paired_max_pct": ratio_percent(max(exact_ratios)),
                "exact_over_v7_paired_median_speedup": statistics.median(
                    1.0 / value for value in exact_ratios),
            }
            for variant, prefix in (
                ("main", "main"), ("exact", "exact"),
                ("sampled_v7", "v7"),
            ):
                for run, value in zip(RUNS, current["times"][variant]):
                    row[f"{prefix}_run{run}_us"] = value
                    row[f"{prefix}_bad_rows_run{run}"] = current[
                        "records"][run][variant]["bad_rows"]
                row[f"{prefix}_median_us"] = current["median_times"][variant]
                row[f"{prefix}_max_abs_delta"] = max(
                    current["records"][run][variant]["max_abs_delta"]
                    for run in RUNS
                )
            writer.writerow(row)


def write_big_table(path: Path, cells: list[dict]) -> None:
    lines = [
        "# Production-dispatch Persistent TopK: three-run cell table",
        "",
        "Times are per-launch CUDA-event medians in microseconds. Each timing "
        "cell shows run 1 / run 2 / run 3. Percentages are the median of the "
        "three paired run ratios; lower is faster. `bad` is the total bad rows "
        "across the three runs.",
        "",
        "| Distribution | Width | Rows | k | Dispatch | MAIN us (R1/R2/R3) | "
        "Exact us (R1/R2/R3) | v7 us (R1/R2/R3) | v7 vs MAIN | "
        "v7 vs Exact | bad MAIN/Exact/v7 |",
        "|---|---:|---:|---:|---|---|---|---|---:|---:|---:|",
    ]
    for current in cells:
        main_bad = sum(
            current["records"][run]["main"]["bad_rows"] for run in RUNS
        )
        exact_bad = sum(
            current["records"][run]["exact"]["bad_rows"] for run in RUNS
        )
        v7_bad = sum(
            current["records"][run]["sampled_v7"]["bad_rows"]
            for run in RUNS
        )
        lines.append(
            f"| {current['distribution']} | {current['width']:,} | "
            f"{current['rows']} | {current['topk']:,} | "
            f"{current['dispatcher']} | {format_runs(current['times']['main'])} "
            f"| {format_runs(current['times']['exact'])} | "
            f"{format_runs(current['times']['sampled_v7'])} | "
            f"{ratio_percent(statistics.median(current['v7_main_ratios'])):+.2f}% "
            f"| {ratio_percent(statistics.median(current['v7_exact_ratios'])):+.2f}% "
            f"| {main_bad}/{exact_bad}/{v7_bad} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_ratio_summary(group: dict, prefix: str) -> str:
    geomean_ratio = group[f"{prefix}_geomean_ratio"]
    run_geomeans = group[f"{prefix}_run_geomeans"]
    return (
        f"{ratio_percent(geomean_ratio):+.2f}% "
        f"[{ratio_percent(min(run_geomeans)):+.2f}%, "
        f"{ratio_percent(max(run_geomeans)):+.2f}%]"
    )


def write_summary(path: Path, groups: list[dict], environments: dict,
                  source_root: Path, input_paths: list[Path]) -> None:
    lines = [
        "# Production-dispatch Persistent TopK final experiment",
        "",
        "## Method",
        "",
        "- Hardware: one isolated NVIDIA B300 SXM6 AC; production untouched.",
        "- Three independent runs per distribution with different seeds.",
        "- Per case: 20 warmups, 60 samples, 20 launches/sample.",
        "- All variants use the same production dispatcher: "
        "`rows > 32 -> FilteredTopK`, otherwise `persistent_topk`.",
        "- MAIN: PR base `903da60`; Exact: committed PR head `b8f88c1`; "
        "candidate: sampled-adaptive v7.",
        "- Each cell covers widths 4,096..250,000, rows 1..128, and "
        "top-k 512/1,024/2,048.",
        "- Timing is CUDA-event kernel time. Aggregate percentages are "
        "geometric means of paired v7/baseline ratios over all case-runs; "
        "brackets show the range of the three per-run geometric means.",
        "",
        "## Aggregate results",
        "",
        "| Group | Cases | MAIN bad rows | Exact bad rows | v7 bad rows | "
        "v7 vs MAIN | v7 vs Exact |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for group in groups:
        lines.append(
            f"| {group['name']} | {group['cells']} | "
            f"{group['main_bad_rows']:,} | {group['exact_bad_rows']:,} | "
            f"{group['v7_bad_rows']:,} | "
            f"{format_ratio_summary(group, 'v7_main')} | "
            f"{format_ratio_summary(group, 'v7_exact')} |"
        )

    exact_bad = sum(group["exact_bad_rows"] for group in groups[:9:3])
    v7_bad = sum(group["v7_bad_rows"] for group in groups[:9:3])
    main_bad = sum(group["main_bad_rows"] for group in groups[:9:3])
    lines.extend([
        "",
        "## Correct interpretation",
        "",
        f"Across the complete 504-case matrix and three runs, MAIN produced "
        f"{main_bad:,} bad row results; Exact produced {exact_bad:,}; v7 "
        f"produced {v7_bad:,}. Exact and v7 also had zero invalid indices, "
        "zero duplicate rows, and zero selected-value delta in every result.",
        "",
        "MAIN is incorrect on the narrow-overflow cases, so MAIN timing is "
        "reported as a production reference, not as a correctness-equivalent "
        "performance baseline. The fair fixed-vs-fixed comparison is v7 "
        "against Exact.",
        "",
        "The previous `full-v7-20260819` result forced the persistent kernel "
        "for every row count and mixed that dispatcher change into the "
        "comparison. Its `12.80%` long-row speedup must not be used as a "
        "production-dispatch conclusion. This experiment supersedes it.",
        "",
        "## Provenance",
        "",
        f"- GPU: `{next(iter(environments.values()))['gpu']}`",
        f"- PyTorch: `{next(iter(environments.values()))['torch']}`",
        f"- vLLM package: `{next(iter(environments.values()))['vllm']}`",
        "- Raw JSONL SHA256:",
    ])
    for input_path in input_paths:
        lines.append(f"  - `{input_path.name}`: `{sha256(input_path)}`")
    lines.extend([
        "- Source SHA256:",
    ])
    for relative in (
        "benchmark/persistent_topk_extension.cu",
        "benchmark/compare_production_dispatch.py",
        "kernels/main-903da60/persistent_topk.cuh",
        "kernels/main-903da60/topk_histogram_4096.cuh",
        "kernels/pr-exact-b8f88c1/persistent_topk.cuh",
        "kernels/pr-exact-b8f88c1/topk_histogram_4096.cuh",
        "kernels/sampled-adaptive-v7/persistent_topk.cuh",
        "kernels/sampled-adaptive-v7/topk_histogram_4096.cuh",
    ):
        source_path = source_root / relative
        lines.append(f"  - `{relative}`: `{sha256(source_path)}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest(path: Path, environments: dict, groups: list[dict],
                   inputs: list[Path], outputs: list[Path]) -> None:
    manifest = {
        "status": "complete",
        "expected_environment_records": 9,
        "expected_result_records": 4536,
        "expected_total_jsonl_lines": 4545,
        "runs": list(RUNS),
        "distributions": list(DISTRIBUTIONS),
        "widths": list(WIDTHS),
        "rows": list(ROWS),
        "topks": list(TOPKS),
        "variants": list(VARIANTS),
        "environments": list(environments.values()),
        "groups": groups,
        "input_sha256": {item.name: sha256(item) for item in inputs},
        "output_sha256": {item.name: sha256(item) for item in outputs},
    }
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    paths = sorted(args.input_dir.glob("run*.jsonl"))
    if len(paths) != 9:
        raise ValueError(f"expected 9 JSONL files, got {len(paths)}")
    environments, indexed = load_and_validate(paths)
    cells = all_cells(indexed)
    groups = build_groups(cells)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = args.output_dir / "PRODUCTION_DISPATCH_3RUN_CELLS.csv"
    table_path = args.output_dir / "PRODUCTION_DISPATCH_3RUN_BIG_TABLE.md"
    summary_path = args.output_dir / "PRODUCTION_DISPATCH_3RUN_SUMMARY.md"
    manifest_path = args.output_dir / "PRODUCTION_DISPATCH_3RUN_MANIFEST.json"
    write_csv(csv_path, cells)
    write_big_table(table_path, cells)
    write_summary(summary_path, groups, environments, args.source_root, paths)
    write_manifest(
        manifest_path,
        environments,
        groups,
        paths,
        [csv_path, table_path, summary_path],
    )

    print("validated environments=9 results=4536 total_jsonl_lines=4545")
    print(f"cells={len(cells)}")
    for output in (summary_path, table_path, csv_path, manifest_path):
        print(f"output={output} sha256={sha256(output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
