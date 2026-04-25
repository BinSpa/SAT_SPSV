#!/usr/bin/env python3
"""
Aggregate all evaluation JSON results into a single summary table.
"""
import os
import json
import glob
from collections import defaultdict

RESULTS_DIR = "/root/autodl-tmp/spsv/results"

def main():
    pattern = os.path.join(RESULTS_DIR, "*_results.json")
    files = glob.glob(pattern)

    # Group by model
    model_results = defaultdict(dict)
    for f in sorted(files):
        basename = os.path.basename(f)
        # basename format: {model_name}_{dataset}_results.json
        # model name may contain underscores, dataset is one of cvbench, blink, sat
        if basename.endswith("_results.json"):
            parts = basename.replace("_results.json", "").split("_")
            # Try to identify dataset (last part)
            dataset = parts[-1]
            if dataset in ("cvbench", "blink", "sat"):
                model_name = "_".join(parts[:-1])
            else:
                # fallback
                model_name = "_".join(parts)
                dataset = "unknown"
            with open(f, "r") as fp:
                data = json.load(fp)
            metrics = data.get("metrics", {})
            model_results[model_name][dataset] = metrics

    # Print markdown table
    datasets = ["cvbench", "blink", "sat"]
    print("# Evaluation Results Summary\n")
    print("| Model | " + " | ".join(d.capitalize() for d in datasets) + " |")
    print("|" + "---|" * (len(datasets) + 1))

    for model_name in sorted(model_results.keys()):
        row = [model_name]
        for ds in datasets:
            metrics = model_results[model_name].get(ds, {})
            acc = metrics.get("overall_accuracy", None)
            if acc is not None:
                row.append(f"{acc:.4f}")
            else:
                row.append("-")
        print("| " + " | ".join(row) + " |")

    # Also save as JSON
    summary_path = os.path.join(RESULTS_DIR, "aggregated_summary.json")
    with open(summary_path, "w") as fp:
        json.dump(dict(model_results), fp, indent=2)
    print(f"\nAggregated summary saved to {summary_path}")

if __name__ == "__main__":
    main()
