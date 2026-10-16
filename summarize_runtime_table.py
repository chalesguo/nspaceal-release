#!/usr/bin/env python3
"""Build a runtime breakdown table from metrics.jsonl files."""

import argparse
import json
from pathlib import Path


DATASET_LABELS = {
    "cifar100": "CIFAR-100",
    "svhn": "SVHN",
    "caltech101": "Caltech-101",
}


def read_rows(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("num_select", 0) > 0 and "feature_time_s" in rec:
                rows.append(rec)
    return rows


def mean(rows, key):
    return sum(float(r.get(key, 0.0)) for r in rows) / len(rows)


def infer_dataset(metrics_path):
    config_path = metrics_path.parent / "config.json"
    if config_path.exists():
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        dataset = cfg.get("dataset")
        if dataset in DATASET_LABELS:
            return DATASET_LABELS[dataset]

    name = str(metrics_path).lower()
    for dataset, label in DATASET_LABELS.items():
        if dataset in name:
            return label
    return metrics_path.parent.name


def collect_runs(input_dir):
    runs = {}
    for metrics_path in sorted(Path(input_dir).rglob("metrics.jsonl")):
        rows = read_rows(metrics_path)
        if not rows:
            continue
        label = infer_dataset(metrics_path)
        runs.setdefault(label, []).extend(rows)
    return runs


def summarize(runs):
    summary = []
    for dataset in ["CIFAR-100", "SVHN", "Caltech-101"]:
        rows = runs.get(dataset, [])
        if not rows:
            summary.append((dataset, None))
            continue
        summary.append(
            (
                dataset,
                {
                    "pool_size": mean(rows, "unlabeled_size"),
                    "feature_time_s": mean(rows, "feature_time_s"),
                    "svd_time_s": mean(rows, "svd_time_s"),
                    "density_time_s": mean(rows, "density_time_s"),
                    "select_time_s": mean(rows, "select_time_s"),
                    "acquisition_time_s": mean(rows, "acquisition_time_s"),
                    "n": len(rows),
                },
            )
        )
    return summary


def markdown_table(summary):
    lines = [
        "| Dataset | Pool size | Feature extraction | SVD | Density | Greedy selection | Total acquisition time |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset, rec in summary:
        if rec is None:
            lines.append(f"| {dataset} | -- | -- | -- | -- | -- | -- |")
            continue
        lines.append(
            "| {dataset} | {pool:,.0f} | {feat:.2f} s | {svd:.2f} s | "
            "{den:.2f} s | {sel:.2f} s | {tot:.2f} s |".format(
                dataset=dataset,
                pool=rec["pool_size"],
                feat=rec["feature_time_s"],
                svd=rec["svd_time_s"],
                den=rec["density_time_s"],
                sel=rec["select_time_s"],
                tot=rec["acquisition_time_s"],
            )
        )
    return "\n".join(lines)


def latex_table(summary):
    lines = [
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "Dataset & Pool size & Feature extraction & SVD & Density & Greedy selection & Total acquisition \\\\",
        "\\midrule",
    ]
    for dataset, rec in summary:
        if rec is None:
            lines.append(f"{dataset} & -- & -- & -- & -- & -- & -- \\\\")
            continue
        lines.append(
            "{dataset} & {pool:,.0f} & {feat:.2f}s & {svd:.2f}s & "
            "{den:.2f}s & {sel:.2f}s & {tot:.2f}s \\\\".format(
                dataset=dataset,
                pool=rec["pool_size"],
                feat=rec["feature_time_s"],
                svd=rec["svd_time_s"],
                den=rec["density_time_s"],
                sel=rec["select_time_s"],
                tot=rec["acquisition_time_s"],
            )
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="results/ours_runtime")
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir or args.input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(collect_runs(args.input_dir))
    (output_dir / "runtime_table.md").write_text(markdown_table(summary) + "\n", encoding="utf-8")
    (output_dir / "runtime_table.tex").write_text(latex_table(summary) + "\n", encoding="utf-8")
    print(markdown_table(summary))


if __name__ == "__main__":
    main()
