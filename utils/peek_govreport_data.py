#!/usr/bin/env python3
"""
Quick profiling script for GovReport summarization settings.

Usage examples:
  python peek_govreport_data.py
  python peek_govreport_data.py --split test --sample-size 2000
  python peek_govreport_data.py --tokenizer Qwen/Qwen2.5-7B-Instruct --save-json ../results/data_profile.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer


def pct(values: np.ndarray, p: float) -> int:
    return int(np.percentile(values, p))


def basic_stats(values: List[int], prefix: str) -> Dict[str, int]:
    arr = np.array(values)
    return {
        f"{prefix}_p50": pct(arr, 50),
        f"{prefix}_p75": pct(arr, 75),
        f"{prefix}_p90": pct(arr, 90),
        f"{prefix}_p95": pct(arr, 95),
        f"{prefix}_p99": pct(arr, 99),
        f"{prefix}_max": int(np.max(arr)),
        f"{prefix}_mean": int(np.mean(arr)),
    }


def truncation_rate(lengths: np.ndarray, cap: int) -> float:
    return float((lengths > cap).mean())


def recommend_caps(report_lens: np.ndarray, summary_lens: np.ndarray) -> Dict[str, int]:
    # Conservative single-pass defaults that still scale better on bigger GPUs.
    max_input = int(np.clip(np.percentile(report_lens, 90), 3500, 9000))
    max_new = int(np.clip(np.percentile(summary_lens, 90), 300, 900))
    return {
        "recommended_max_input_tokens": max_input,
        "recommended_max_new_tokens": max_new,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile GovReport token lengths")
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tokenizer", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--save-json", type=str, default="")
    args = parser.parse_args()

    print("Loading dataset: ccdv/govreport-summarization")
    dataset = load_dataset("ccdv/govreport-summarization")
    split_ds = dataset[args.split]

    n = min(args.sample_size, len(split_ds))
    if n <= 0:
        raise ValueError("sample-size must be > 0")

    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    rng = np.random.default_rng(args.seed)
    indices = rng.choice(len(split_ds), size=n, replace=False)

    report_char_lens: List[int] = []
    summary_char_lens: List[int] = []
    report_tok_lens: List[int] = []
    summary_tok_lens: List[int] = []

    for idx in indices:
        item = split_ds[int(idx)]
        report = item["report"]
        summary = item["summary"]

        report_char_lens.append(len(report))
        summary_char_lens.append(len(summary))
        report_tok_lens.append(len(tokenizer.encode(report, add_special_tokens=False)))
        summary_tok_lens.append(len(tokenizer.encode(summary, add_special_tokens=False)))

    report_tok_arr = np.array(report_tok_lens)
    summary_tok_arr = np.array(summary_tok_lens)

    stats: Dict[str, object] = {
        "dataset": "ccdv/govreport-summarization",
        "split": args.split,
        "num_profiled": int(n),
        "tokenizer": args.tokenizer,
    }
    stats.update(basic_stats(report_char_lens, "report_chars"))
    stats.update(basic_stats(summary_char_lens, "summary_chars"))
    stats.update(basic_stats(report_tok_lens, "report_tokens"))
    stats.update(basic_stats(summary_tok_lens, "summary_tokens"))
    stats.update(recommend_caps(report_tok_arr, summary_tok_arr))

    caps_to_check = [3500, 4500, 5500, 7000, 9000]
    trunc = {
        str(cap): round(truncation_rate(report_tok_arr, cap) * 100.0, 2)
        for cap in caps_to_check
    }
    stats["truncation_rate_percent_by_max_input"] = trunc

    print("\n=== GOVREPORT PROFILE RESULTS ===")
    print(json.dumps(stats, indent=2))

    print("\n=== INTERPRETATION ===")
    print(
        f"If max_input_tokens={stats['recommended_max_input_tokens']}, "
        f"about {trunc[str(stats['recommended_max_input_tokens'])] if str(stats['recommended_max_input_tokens']) in trunc else 'N/A'}% "
        "of sampled reports would still be truncated (see full table above)."
    )
    print(
        f"Suggested max_new_tokens baseline start: {stats['recommended_max_new_tokens']} "
        "(based on summary token P90)."
    )

    if args.save_json:
        out_path = Path(args.save_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        print(f"\nSaved JSON profile to: {out_path}")


if __name__ == "__main__":
    main()
