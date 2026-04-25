#!/usr/bin/env python3
"""
Batch runner for SAT benchmark evaluations across all local models.
"""
import os
import sys
import json
import subprocess
import traceback

OUTPUT_DIR = "/root/autodl-tmp/spsv/results"
EVAL_SCRIPT = "/root/autodl-tmp/spsv/SAT_SPSV/evaluate.py"

MODELS = [
    {
        "path": "/root/autodl-fs/models/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17",
        "name": "Qwen3-VL-4B-Instruct",
        "adapter": "qwen",
        "batch_size": 4,
    },
    {
        "path": "/root/autodl-fs/models/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
        "name": "Qwen3-VL-8B-Instruct",
        "adapter": "qwen",
        "batch_size": 8,
    },
    {
        "path": "/root/autodl-fs/models/models--OpenGVLab--InternVL3-8B-Instruct/snapshots/ddb3a169d5582e5c76e0809a128e55ab63686ada",
        "name": "InternVL3-8B-Instruct",
        "adapter": "internvl3",
        "batch_size": 8,
    },
    {
        "path": "/root/autodl-fs/models/models--OpenGVLab--InternVL3_5-4B-HF/snapshots/6bd4487402110ef9889ba50eb7aefeb302526fed",
        "name": "InternVL3_5-4B-HF",
        "adapter": "internvl35",
        "batch_size": 16,
    },
    {
        "path": "/root/autodl-fs/models/models--allenai--Molmo2-4B/snapshots/042abfa7a38879a376cec03d949eff0aefaa0600",
        "name": "Molmo2-4B",
        "adapter": "molmo",
        "batch_size": 8,
    },
    {
        "path": "/root/autodl-fs/models/models--allenai--Molmo2-8B/snapshots/e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b",
        "name": "Molmo2-8B",
        "adapter": "molmo",
        "batch_size": 4,
    },
    {
        "path": "/root/autodl-fs/models/models--google--gemma-3-4b-it/snapshots/093f9f388b31de276ce2de164bdc2081324b9767",
        "name": "gemma-3-4b-it",
        "adapter": "gemma",
        "batch_size": 16,
    },
    {
        "path": "/root/autodl-fs/models/models--lmms-lab--LLaVA-OneVision-1.5-4B-Instruct/snapshots/3ad1d8780ca1e4c62c5dab3d5e17d28a66c5e7a0",
        "name": "LLaVA-OneVision-1.5-4B-Instruct",
        "adapter": "llava_ov",
        "batch_size": 8,
    },
    {
        "path": "/root/autodl-fs/models/models--lmms-lab--LLaVA-OneVision-1.5-8B-Instruct/snapshots/bdf95183e0f91549cc71820b12d920b1a63689a2",
        "name": "LLaVA-OneVision-1.5-8B-Instruct",
        "adapter": "llava_ov",
        "batch_size": 4,
    },
    {
        "path": "/root/autodl-fs/models/models--openbmb--MiniCPM-V-4_5/snapshots/fd3209b2e0580e346fc33d2c6f85b6e9332eecda",
        "name": "MiniCPM-V-4_5",
        "adapter": "minicpm",
        "batch_size": 8,
    },
    {
        "path": "/root/autodl-fs/models/models--BytedanceDouyinContent--SAIL-VL2-8B/snapshots/0677983371e7340c50666e3623bcb2a2e33afbff",
        "name": "SAIL-VL2-8B",
        "adapter": "internvl3",
        "batch_size": 8,
    },
]

DATASETS = ["cvbench", "blink", "sat"]

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for model_info in MODELS:
        name = model_info["name"]
        print(f"\n{'='*60}")
        print(f"Evaluating {name}")
        print(f"{'='*60}")
        for ds in DATASETS:
            log_path = os.path.join(OUTPUT_DIR, f"{name}_{ds}_log.txt")
            # Skip if results already exist
            result_path = os.path.join(OUTPUT_DIR, f"{name}_{ds}_results.json")
            if os.path.exists(result_path):
                print(f"  [{ds}] already done, skipping.")
                continue
            cmd = [
                "conda", "run", "-n", "torch24", "python", EVAL_SCRIPT,
                "--model_path", model_info["path"],
                "--model_name", name,
                "--adapter", model_info["adapter"],
                "--datasets", ds,
                "--batch_size", str(model_info["batch_size"]),
                "--output_dir", OUTPUT_DIR,
            ]
            print(f"  Running {ds} ...")
            with open(log_path, "w") as log_f:
                proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT)
                proc.wait()
            if proc.returncode != 0:
                print(f"  ERROR on {ds}! See {log_path}")
            else:
                print(f"  Done {ds}.")
    print("\nAll evaluations finished.")

if __name__ == "__main__":
    main()
