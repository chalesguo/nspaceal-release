#!/usr/bin/env python3
"""Summarize active-learning metrics as mean/std tables."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


METRIC_KEYS = [
    "test_acc",
    "feature_time_s",
    "svd_time_s",
    "density_time_s",
    "select_time_s",
    "select_wall_time_s",
    "acquisition_time_s",
]


def read_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def summarize(records):
    grouped = defaultdict(list)
    for rec in records:
        grouped[rec["cycle"]].append(rec)

    rows = []
    for cycle in sorted(grouped):
        recs = grouped[cycle]
        row = {
            "cycle": cycle,
            "n_trials": len({r["trial"] for r in recs}),
            "labeled_size": int(round(mean(r["labeled_size"] for r in recs))),
            "unlabeled_size": int(round(mean(r["unlabeled_size"] for r in recs))),
        }
        for key in METRIC_KEYS:
            values = [float(r[key]) for r in recs if key in r and r[key] is not None]
            if values:
                row[f"{key}_mean"] = mean(values)
                row[f"{key}_std"] = stdev(values) if len(values) > 1 else 0.0
        rows.append(row)
    return rows


def write_csv(rows, output):
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def print_table(rows):
    for row in rows:
        acc = row.get("test_acc_mean")
        acc_std = row.get("test_acc_std")
        acq = row.get("acquisition_time_s_mean")
        acq_std = row.get("acquisition_time_s_std")
        acc_text = "n/a" if acc is None else f"{acc:.2f} +/- {acc_std:.2f}"
        acq_text = "n/a" if acq is None else f"{acq:.2f} +/- {acq_std:.2f}s"
        print(
            f"cycle={row['cycle']} labeled={row['labeled_size']} "
            f"trials={row['n_trials']} acc={acc_text} acquisition={acq_text}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", help="Path to metrics.jsonl or an experiment directory.")
    parser.add_argument("--output", default=None, help="Optional CSV output path.")
    args = parser.parse_args()

    path = Path(args.metrics)
    if path.is_dir():
        path = path / "metrics.jsonl"
    records = read_jsonl(path)
    rows = summarize(records)
    print_table(rows)

    output = args.output
    if output is None:
        output = str(path.with_name("summary_mean_std.csv"))
    write_csv(rows, output)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
