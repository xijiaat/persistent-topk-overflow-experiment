#!/usr/bin/env python3
import argparse
import csv
from collections import defaultdict
from pathlib import Path


TOPKS = (512, 1024, 2048)


def value(row: dict, key: str) -> float:
    return float(row[key])


def cell(row: dict) -> str:
    main = value(row, "main_median_us")
    v7 = value(row, "v7_median_us")
    vs_main = value(row, "v7_vs_main_paired_median_pct")
    return f"{main:8.3f} {v7:8.3f} {vs_main:+8.1f}%"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    grouped = defaultdict(dict)
    with args.csv.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            key = (row["distribution"], int(row["width"]), int(row["rows"]))
            grouped[key][int(row["topk"])] = row

    lines = [
        "Three-run medians; time in us. Negative percentages mean v7 is faster.",
        "This table intentionally compares sampled-adaptive v7 directly with MAIN.",
        "",
    ]
    distributions = ("narrow", "normal", "wide_uniform")
    widths = (4096, 12288, 32767, 32768, 32769, 65536, 131072, 250000)
    rows = (1, 8, 16, 32, 33, 64, 128)
    for distribution in distributions:
        for width in widths:
            lines.append(
                f"{distribution} | production dispatch | width={width:,} | time in us"
            )
            lines.append(
                " rows | dispatch     |"
                + "|".join(f"{'topk=' + str(k):^31}" for k in TOPKS)
            )
            lines.append(
                "      |              |"
                + "|".join(
                    "    MAIN       v7    vs MAIN "
                    for _ in TOPKS
                )
            )
            lines.append("-" * 118)
            for row_count in rows:
                records = grouped[(distribution, width, row_count)]
                dispatch = records[TOPKS[0]]["dispatcher"]
                lines.append(
                    f"{row_count:5d} | {dispatch:<12} |"
                    + "|".join(cell(records[k]) for k in TOPKS)
                )
            lines.append("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
