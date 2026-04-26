# SAT Benchmark Evaluation Report

This report summarizes the evaluation of 11 vision-language models (VLMs) on three spatial reasoning benchmarks: **CVBench**, **BLINK**, and **SAT-v2**. All evaluations were conducted using the `QAAccuracy` metric from the original SAT paper, with greedy decoding (`do_sample=False`) and `max_new_tokens=128`.

---

## Table of Contents

- [1. Evaluation Setup](#1-evaluation-setup)
- [2. Metrics](#2-metrics)
- [3. Overall Results](#3-overall-results)
- [4. Per-Benchmark Analysis](#4-per-benchmark-analysis)
  - [4.1 CVBench](#41-cvbench)
  - [4.2 BLINK](#42-blink)
  - [4.3 SAT-v2](#43-sat-v2)
- [5. Model Rankings](#5-model-rankings)
- [6. Key Findings](#6-key-findings)
- [7. Limitations & Notes](#7-limitations--notes)
- [8. Action Items](#8-action-items)

---

## 1. Evaluation Setup

| Component | Details |
|---|---|
| **Hardware** | 2 x NVIDIA A800 80GB |
| **Environment** | `torch24` conda env, `transformers==4.57.1`, `torch>=2.4` |
| **GPU Strategy** | `device_map="auto"` for model sharding across both GPUs |
| **Decoding** | Greedy, `max_new_tokens=128` |
| **Batch Sizes** | 4B models: 16 (CVBench/BLINK), 4 (SAT); 8B models: 8 (CVBench/BLINK), 4 (SAT) |
| **Datasets** | `nyu-visionx/CV-Bench`, `BLINK-Benchmark/BLINK`, `array/SAT-v2` |

---

## 2. Metrics

We use **QAAccuracy** (exact-match accuracy after normalization), following the original SAT paper. The scoring pipeline is:

1. **Normalize** both prediction and ground-truth:
   - Convert to lowercase
   - Strip punctuation (`.`, `!`, `?`) and quotation marks
   - Remove parentheses
2. **Exact Match**: If the normalized ground-truth is a substring of the prediction (or vice versa), score = 1.
3. **Dataset-Specific Heuristics**: For certain question types where models tend to produce longer responses, we extract the key answer word:
   - *"Which object is closer"* → extract object name before "is closer"
   - *"Is the camera moving"* → extract direction (clockwise/counter-clockwise mapped to left/right)
   - *"Considering the relative positions"* → extract "left" or "right"

This metric rewards concise, correct answers and penalizes verbose but ambiguous responses.

---

## 3. Overall Results

| Rank | Model | CVBench | BLINK | SAT-v2 | **Average** |
|---|---:|---:|---:|---:|---:|
| 1 | **Molmo2-4B** | **0.9181** | **0.9875** | **0.9067** | **0.9374** |
| 2 | Qwen3-VL-8B-Instruct | 0.8726 | 0.7300 | 0.6733 | 0.7586 |
| 3 | Qwen3-VL-4B-Instruct | 0.8563 | 0.7100 | 0.6800 | 0.7488 |
| 4 | Molmo2-8B | 0.8355 | 0.7625 | 0.5933 | 0.7304 |
| 5 | MiniCPM-V-4.5 | 0.8199 | 0.7000 | 0.6400 | 0.7200 |
| 6 | SAIL-VL2-8B | 0.8108 | 0.7375 | 0.5733 | 0.7072 |
| 7 | InternVL3-8B-Instruct | 0.8234 | 0.7400 | 0.5667 | 0.7100 |
| 8 | LLaVA-OneVision-1.5-8B-Instruct | 0.8143 | 0.7100 | 0.6000 | 0.7081 |
| 9 | InternVL3_5-4B-HF | 0.8074 | 0.7100 | 0.5800 | 0.6991 |
| 10 | LLaVA-OneVision-1.5-4B-Instruct | 0.7752 | 0.7000 | 0.5467 | 0.6740 |
| 11 | gemma-3-4b-it | 0.5785 | 0.6125 | 0.5333 | 0.5748 |

> **Note**: Molmo2-8B CVBench has 1 empty prediction out of 2638 due to a CUDA OOM on a single batch (negligible impact on overall score).

---

## 4. Per-Benchmark Analysis

### 4.1 CVBench

CVBench contains **2,638** test samples across 4 sub-tasks: 2D Relation, 3D Depth, 2D Count, and 3D Distance.

| Model | 2D Relation | 3D Depth | 2D Count | 3D Distance | **Overall** |
|---|---|---|---|---|---|
| Molmo2-4B | **0.9431** | **0.9483** | **0.8350** | **0.9700** | **0.9181** |
| Qwen3-VL-8B | 0.9462 | 0.9567 | 0.7246 | 0.9033 | 0.8726 |
| Qwen3-VL-4B | 0.9446 | 0.9467 | 0.6853 | 0.8950 | 0.8563 |
| Molmo2-8B | 0.9077 | 0.9433 | 0.7627 | 0.7450 | 0.8355 |
| MiniCPM-V-4.5 | 0.9292 | 0.8933 | 0.7246 | 0.7533 | 0.8199 |
| SAIL-VL2-8B | 0.9077 | 0.8500 | 0.7145 | 0.7933 | 0.8108 |
| InternVL3-8B | 0.9246 | 0.8450 | 0.7373 | 0.8050 | 0.8234 |
| LLaVA-OV-8B | 0.9046 | 0.8483 | 0.7284 | 0.7950 | 0.8143 |
| InternVL3.5-4B | 0.9200 | 0.8550 | 0.6980 | 0.7817 | 0.8074 |
| LLaVA-OV-4B | 0.8954 | 0.8300 | 0.6789 | 0.7167 | 0.7752 |
| gemma-3-4b | 0.6585 | 0.6200 | 0.4480 | 0.6217 | 0.5785 |

**Observations:**
- **Molmo2-4B** dominates CVBench, particularly on 3D Distance (97.0%) and 2D Relation (94.3%). Its strong performance on counting (83.5%) is notably higher than all other models.
- **Qwen3-VL** models show strong 3D Depth performance (>94%) but struggle relatively with 2D Count (~68-72%).
- **Gemma-3-4b** is significantly behind on all sub-tasks, especially counting (44.8%), suggesting poor numerical reasoning.
- **2D Count** is the hardest sub-task across the board — no 8B model exceeds 76.3%.

---

### 4.2 BLINK

BLINK contains **400** validation samples across 3 sub-tasks: Multi-view Reasoning, Relative Depth, and Spatial Relation.

| Model | Multi-view | Relative Depth | Spatial Relation | **Overall** |
|---|---|---|---|---|
| **Molmo2-4B** | **1.0000** | **1.0000** | **0.9650** | **0.9875** |
| Molmo2-8B | 0.7368 | 0.7258 | 0.8182 | 0.7625 |
| SAIL-VL2-8B | 0.5639 | 0.7177 | 0.9161 | 0.7375 |
| Qwen3-VL-8B | 0.5038 | 0.8387 | 0.8462 | 0.7300 |
| InternVL3-8B | 0.5338 | 0.8065 | 0.8741 | 0.7400 |
| Qwen3-VL-4B | 0.4436 | 0.8145 | 0.8671 | 0.7100 |
| InternVL3.5-4B | 0.4511 | 0.8468 | 0.8322 | 0.7100 |
| LLaVA-OV-8B | 0.5564 | 0.7097 | 0.8531 | 0.7100 |
| MiniCPM-V-4.5 | 0.4436 | 0.7903 | 0.8601 | 0.7000 |
| LLaVA-OV-4B | 0.5564 | 0.6855 | 0.8462 | 0.7000 |
| gemma-3-4b | 0.5564 | 0.6129 | 0.6643 | 0.6125 |

**Observations:**
- **Molmo2-4B achieves near-perfect BLINK scores** (98.75% overall), with perfect accuracy on Multi-view Reasoning and Relative Depth. This is a remarkable outlier.
- **Molmo2-8B is significantly worse than Molmo2-4B** on BLINK (76.25% vs 98.75%), suggesting the 8B variant may have different training data or alignment that hurts spatial reasoning.
- **Multi-view Reasoning** is the hardest sub-task for most models (all except Molmo2-4B score <57%).
- **Relative Depth** and **Spatial Relation** are more tractable, with most models scoring 70-85%.
- **Gemma-3-4b** again lags behind, particularly on Spatial Relation (66.4%).

---

### 4.3 SAT-v2

SAT-v2 (Real Test) contains **150** samples across 5 dynamic reasoning sub-tasks: Object Movement, Ego Movement, Action Consequence, Perspective Taking, and Goal Aim.

| Model | Obj Move | Ego Move | Action Conseq | Perspective | Goal Aim | **Overall** |
|---|---|---|---|---|---|---|
| **Molmo2-4B** | **0.9565** | **0.9130** | **0.9730** | **0.7879** | **0.9118** | **0.9067** |
| Qwen3-VL-4B | 0.6087 | **1.0000** | 0.5405 | 0.4545 | 0.8824 | 0.6800 |
| Qwen3-VL-8B | 0.3478 | **1.0000** | 0.7027 | 0.3636 | 0.9412 | 0.6733 |
| MiniCPM-V-4.5 | 0.7826 | 0.6087 | 0.6757 | 0.3636 | 0.7941 | 0.6400 |
| Molmo2-8B | 0.7826 | 0.0000 | 0.8108 | 0.4242 | 0.7941 | 0.5933 |
| LLaVA-OV-8B | 0.6522 | 0.5217 | 0.6757 | 0.4545 | 0.6765 | 0.6000 |
| InternVL3_5-4B | 0.9130 | 0.5217 | 0.4324 | 0.4848 | 0.6471 | 0.5800 |
| SAIL-VL2-8B | 0.8261 | 0.2174 | 0.6757 | 0.4242 | 0.6765 | 0.5733 |
| InternVL3-8B | 0.5652 | 0.0000 | 0.7297 | 0.6061 | 0.7353 | 0.5667 |
| LLaVA-OV-4B | 0.6087 | 0.2609 | 0.5946 | 0.4545 | 0.7353 | 0.5467 |
| gemma-3-4b | 0.3478 | 0.6522 | 0.7838 | 0.3939 | 0.4412 | 0.5333 |

**Observations:**
- **Molmo2-4B is the clear winner** on SAT-v2 (90.67%), with exceptionally strong performance across all sub-tasks, especially Action Consequence (97.3%) and Object Movement (95.7%).
- **Ego Movement is a major failure mode** for several models: Molmo2-8B (0%), InternVL3-8B (0%), SAIL-VL2-8B (21.7%). This suggests camera/self-motion reasoning is particularly challenging for current VLMs.
- **Qwen3-VL models achieve perfect Ego Movement scores** (100%) but struggle with Object Movement (34.8-60.9%) and Perspective Taking (36.4-45.5%).
- **Perspective Taking** is consistently difficult across all models (36-79%), indicating that mental rotation and viewpoint transformation remain hard.
- **InternVL3_5-4B** shows interesting strengths in Object Movement (91.3%) but weakness in Action Consequence (43.2%).

---

## 5. Model Rankings

### By Parameter Efficiency (Performance / Size)

Molmo2-4B is the standout winner, achieving the highest scores on all three benchmarks despite being one of the smallest models. This suggests the Molmo2 training recipe (data mixture, alignment, or architecture) is exceptionally well-suited for spatial reasoning.

| Model | Size | Avg Score | Score/Param |
|---|---|---|---|
| Molmo2-4B | 4B | 0.9374 | **0.2343** |
| Qwen3-VL-4B | 4B | 0.7488 | 0.1872 |
| InternVL3.5-4B | 4B | 0.6991 | 0.1748 |
| LLaVA-OV-4B | 4B | 0.6740 | 0.1685 |
| gemma-3-4b | 4B | 0.5748 | 0.1437 |
| Qwen3-VL-8B | 8B | 0.7586 | 0.0948 |
| MiniCPM-V-4.5 | ~4.5B | 0.7200 | 0.1600 |
| SAIL-VL2-8B | 8B | 0.7072 | 0.0884 |
| InternVL3-8B | 8B | 0.7100 | 0.0888 |
| LLaVA-OV-8B | 8B | 0.7081 | 0.0885 |
| Molmo2-8B | 8B | 0.7304 | 0.0913 |

**Key insight**: Scaling from 4B to 8B does not consistently improve spatial reasoning. In fact, Molmo2-8B is significantly worse than Molmo2-4B on BLINK and SAT-v2. This suggests spatial reasoning capability is more dependent on training data/alignment than pure model scale in this regime.

---

## 6. Key Findings

1. **Molmo2-4B is the best spatial reasoning VLM in this evaluation**, achieving >90% on all benchmarks and near-perfect BLINK scores. Its 4B size makes it highly efficient.

2. **Scaling does not guarantee better spatial reasoning**:
   - Molmo2-8B << Molmo2-4B (especially on BLINK and SAT)
   - Qwen3-VL-8B ≈ Qwen3-VL-4B (marginal gains)
   - LLaVA-OV-8B ≈ LLaVA-OV-4B (marginal gains)

3. **Task difficulty varies significantly by benchmark**:
   - **CVBench**: Most balanced, but 2D Counting is hardest
   - **BLINK**: Multi-view Reasoning is extremely hard for all except Molmo2-4B
   - **SAT-v2**: Ego Movement and Perspective Taking are the major bottlenecks

4. **Gemma-3-4b is not competitive** on spatial tasks, scoring ~15-30 points below the top models on all benchmarks.

5. **Dynamic reasoning (SAT-v2) is harder than static reasoning (CVBench/BLINK)** for most models, with the gap being 10-25 percentage points.

---

## 7. Limitations & Notes

- **Single-run evaluation**: No temperature tuning or prompt engineering was performed beyond the unified prompt template.
- **Image resizing**: Images larger than 2048px on any side were downscaled to prevent OOM and processor hangs. This could slightly affect models that benefit from high-resolution inputs.
- **SAT-v2 streaming**: Loaded via HuggingFace `streaming=True`, which may not guarantee deterministic ordering. However, the dataset is small (150 samples) and loaded in full.
- **Molmo2-8B CVBench**: 1 empty prediction due to CUDA OOM on a single batch. Impact on overall score is <0.04%.
- **QAAccuracy metric**: This is a strict exact-match metric. Models that produce correct but differently phrased answers may be penalized. The heuristics mitigate some of this, but not all.

---

## 8. Action Items

| Priority | Action | Rationale |
|---|---|---|
| **High** | Investigate Molmo2-4B training data and architecture | To understand why it outperforms all 8B models by a large margin |
| **High** | Analyze failure modes on Ego Movement | 0% scores on multiple 8B models suggest a systematic weakness |
| **High** | Evaluate with prompt engineering / few-shot | Current unified prompt may not be optimal for all architectures |
| **Medium** | Test at higher image resolutions | Current 2048px cap may limit high-res models (e.g., Qwen3-VL, InternVL) |
| **Medium** | Evaluate on SAT training split | Check if models have seen SAT data during pre-training |
| **Medium** | Run statistical significance tests | SAT-v2 has only 150 samples; some differences may not be significant |
| **Low** | Fine-tune top models on SAT data | Use the evaluation framework to measure fine-tuning gains |

---

*Report generated on 2026-04-26. Raw results available in `/root/autodl-tmp/spsv/results/`.*
