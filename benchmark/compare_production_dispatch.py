#!/usr/bin/env python3
import argparse
import json
import os
import random
import statistics
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

try:
    import vllm._C  # noqa: F401
except ModuleNotFoundError:
    import vllm._C_stable_libtorch  # noqa: F401


WORKSPACE_BYTES = 1 << 20


def load_op(name: str, source: str, include_dir: str):
    module = load(
        name=name,
        sources=[source],
        extra_include_paths=[include_dir],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        with_cuda=True,
        verbose=True,
    )
    return module.patched_persistent_topk


def make_logits(rows: int, width: int, seed: int,
                distribution: str) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    if distribution == "narrow":
        logits = 1.0 + torch.rand(rows, width, generator=generator) * 0.01
    elif distribution == "normal":
        logits = torch.randn(rows, width, generator=generator)
    elif distribution == "wide_uniform":
        logits = torch.rand(rows, width, generator=generator) * 200.0 - 100.0
    elif distribution == "quantized4":
        logits = torch.randint(0, 4, (rows, width), generator=generator).float()
    elif distribution == "equal":
        logits = torch.ones(rows, width)
    else:
        raise ValueError(f"unknown distribution: {distribution}")
    return logits.cuda()


def run_op(op, logits: torch.Tensor, k: int) -> torch.Tensor:
    rows, width = logits.shape
    lengths = torch.full((rows,), width, dtype=torch.int32, device="cuda")
    output = torch.full((rows, k), -1, dtype=torch.int32, device="cuda")
    workspace = torch.zeros(WORKSPACE_BYTES, dtype=torch.uint8, device="cuda")
    op(logits, lengths, output, workspace, k, width)
    torch.cuda.synchronize()
    return output


def correctness(op, logits: torch.Tensor, k: int) -> dict:
    rows, width = logits.shape
    output = run_op(op, logits, k)
    invalid = (output < 0) | (output >= width)
    safe_output = output.clamp(0, width - 1).long()
    selected = torch.gather(logits, 1, safe_output)
    selected = torch.sort(selected, dim=1, descending=True).values
    expected = torch.topk(logits, k, dim=1).values
    mismatch_per_row = (selected != expected).sum(dim=1)
    duplicate_rows = sum(
        torch.unique(output[row]).numel() != k for row in range(rows)
    )
    return {
        "bad_rows": int((mismatch_per_row > 0).sum().item()),
        "max_mismatches": int(mismatch_per_row.max().item()),
        "max_abs_delta": float((selected - expected).abs().max().item()),
        "invalid_indices": int(invalid.sum().item()),
        "rows_with_duplicates": int(duplicate_rows),
    }


def measure(op, logits: torch.Tensor, k: int, warmups: int, samples: int,
            launches: int) -> dict:
    rows, width = logits.shape
    lengths = torch.full((rows,), width, dtype=torch.int32, device="cuda")
    output = torch.empty((rows, k), dtype=torch.int32, device="cuda")
    workspace = torch.zeros(WORKSPACE_BYTES, dtype=torch.uint8, device="cuda")
    for _ in range(warmups):
        op(logits, lengths, output, workspace, k, width)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    timings = []
    for _ in range(samples):
        start.record()
        for _ in range(launches):
            op(logits, lengths, output, workspace, k, width)
        end.record()
        end.synchronize()
        timings.append(start.elapsed_time(end) * 1000.0 / launches)
    timings.sort()
    return {
        "mean_us": statistics.fmean(timings),
        "median_us": statistics.median(timings),
        "p05_us": timings[int(0.05 * (samples - 1))],
        "p95_us": timings[int(0.95 * (samples - 1))],
        "min_us": timings[0],
        "max_us": timings[-1],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=int, default=1)
    parser.add_argument("--widths", type=int, nargs="+",
                        default=(65536, 131072, 250000))
    parser.add_argument("--rows", type=int, nargs="+", default=(33, 64, 128))
    parser.add_argument("--topks", type=int, nargs="+",
                        default=(512, 1024, 2048))
    parser.add_argument("--warmups", type=int, default=30)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--launches", type=int, default=20)
    parser.add_argument(
        "--distribution",
        choices=("narrow", "normal", "wide_uniform", "quantized4", "equal"),
        default="narrow",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    suffix = os.environ.get("EXTENSION_SUFFIX", "r1")
    candidate_name = os.environ.get("CANDIDATE_NAME", "multi_cta")
    dispatch_source = os.environ["TOPK_DISPATCH_SOURCE"]
    variants = {
        "main": load_op(
            f"vllm_52149_main_production_dispatch_{suffix}",
            dispatch_source,
            os.environ["MAIN_TOPK_INCLUDE"],
        ),
        "exact": load_op(
            f"vllm_52149_exact_production_dispatch_{suffix}",
            dispatch_source,
            os.environ["EXACT_TOPK_INCLUDE"],
        ),
    }
    variants[candidate_name] = load_op(
        f"vllm_52149_candidate_production_dispatch_{suffix}",
        dispatch_source,
        os.environ["CANDIDATE_TOPK_INCLUDE"],
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    seed_base = 2026081800 + args.run * 10000
    rng = random.Random(seed_base)

    environment = {
        "kind": "environment",
        "run": args.run,
        "torch": torch.__version__,
        "vllm": __import__("vllm").__version__,
        "gpu": torch.cuda.get_device_name(0),
        "distribution": args.distribution,
        "widths": args.widths,
        "rows": args.rows,
        "topks": args.topks,
        "warmups": args.warmups,
        "samples": args.samples,
        "launches": args.launches,
        "variants": list(variants),
        "dispatcher": "production: rows>32 FilteredTopK, else persistent",
        "main_include": os.environ["MAIN_TOPK_INCLUDE"],
        "exact_include": os.environ["EXACT_TOPK_INCLUDE"],
        "candidate_include": os.environ["CANDIDATE_TOPK_INCLUDE"],
    }

    with out_path.open("w", encoding="utf-8") as stream:
        line = json.dumps(environment, sort_keys=True)
        stream.write(line + "\n")
        print(line, flush=True)
        case = 0
        for width in args.widths:
            for rows in args.rows:
                for k in args.topks:
                    seed = seed_base + case
                    logits = make_logits(rows, width, seed, args.distribution)
                    order = list(variants)
                    rng.shuffle(order)
                    for name in order:
                        result = {
                            "kind": "result",
                            "run": args.run,
                            "variant": name,
                            "width": width,
                            "rows": rows,
                            "topk": k,
                            "seed": seed,
                        }
                        result.update(correctness(variants[name], logits, k))
                        result.update(measure(
                            variants[name], logits, k, args.warmups,
                            args.samples, args.launches))
                        line = json.dumps(result, sort_keys=True)
                        stream.write(line + "\n")
                        stream.flush()
                        print(line, flush=True)
                    del logits
                    torch.cuda.empty_cache()
                    case += 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
