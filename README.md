<<<<<<< HEAD
# Proxy Migration in Language Models
### A Taxonomy of Reward Hacking Under Selection and Penalization

> *Submitted to NeurIPS 2026 (Confidential reviewer copy — do not distribute)*

---

## Overview

This repository contains the full experimental code for **Proxy Migration**, a predictive framework for understanding where optimization pressure goes when reward hacking is partially suppressed.

**Core finding:** Penalizing one exploit dimension does not stop reward hacking — it redirects it. This paper introduces a four-type taxonomy that predicts *which* migration regime will arise from a given constraint geometry, *before* any experiment is run.

```
Reward hacking is not a single failure mode.
It is a structured family of behavioral regimes,
each predictable from the reward constraint geometry.
```

---

## The Four Migration Types

| Type | Name | Condition | Observable Signature |
|------|------|-----------|----------------------|
| **I** | Direct | No constraints `(C = ∅)` | Proxy climbs monotonically; quality declines |
| **II** | Penalization Escape | One dimension constrained | Constrained dim ↓, free dim ↑ — bidirectional |
| **III** | Latent | Measurable dims constrained | `R(x̂_N) ↑` while `f*(x̂_N) ↓` — decoupling |
| **IV** | Structural Collapse | All dims constrained `(\|F\|=0)` | Reward oscillates/saturates near zero |

**Falsifiability condition:** No migration activates when `|F| = 0` OR `Δ(x) ≈ 0 ∀x`.

---

## Repository Structure

```
proxy-migration/
├── README.md
│
├── src/
│   ├── generate_pool_candidates.py          ← Generate Best-of-N candidate pool (any model/dataset)
│   ├── compute_multidimensional_metrics.py  ← 10-dim escape matrix sweep
│   └── score_armorm_baselines.py            ← All-4-types scoring + figures
│
└── notebooks/
    ├── experiment-01-llama.ipynb                          ← Minimal ERR reproduction (easy entry point)
    ├── experiment-1-final-test-on-t2.ipynb                ← Full 10-dim escape matrix (main result)
    ├── gsm8k-best-of-n-candidate-pool-analysis.ipynb      ← GSM8K falsifiability case
    ├── humaneval-best-of-n-proxy-migration-pass-1.ipynb   ← HumanEval pass@k vs proxy
    ├── llm-as-a-judge-qwen3-helpsteer.ipynb               ← LLM-as-judge quality signal
    ├── multi-selector-llm-as-a-judge-qwen3-helpsteer.ipynb ← Multi-selector judge comparison
    └── alpacaeval-fixed-pool-best-of-n-rm-analysis.ipynb  ← Cross-RM fixed pool analysis
```

---

## Candidate Pools

All experiments use **pre-generated fixed candidate pools** (JSONL format, one record per prompt).
Total pool: **~1.5 million candidates** across all models, datasets, and reward model scorings.

### Pool Summary

| Generator Model | Dataset | Prompts | Candidates/Prompt | Total |
|-----------------|---------|---------|-------------------|-------|
| LLaMA-3.1-8B | AlpacaEval | 500 | 256 | 128,000 |
| LLaMA-3.1-8B | HelpSteer | 500 | 256 | 128,000 |
| LLaMA-3.1-8B | HumanEval | 164 | 256 | 41,984 |
| LLaMA-3.1-8B | GSM8K | 500 | 512 | 256,000 |
| Mistral-7B-Instruct | AlpacaEval | 300 | 256 | 76,800 |
| Mistral-7B-Instruct | HelpSteer | 500 | 256 | 128,000 |
| Mistral-7B-Instruct | HumanEval | 164 | 256 | 41,984 |
| Mistral-7B-Instruct | GSM8K | 500 | 512 | 256,000 |
| Qwen3-14B | AlpacaEval 2.0 | 500 | 256 | 128,000 |
| Qwen3-14B | IFEval | 541 | 256 | 138,496 |
| Qwen3-14B | HumanEval | 164 | 256 | 41,984 |
| Qwen3-14B | GSM8K | 500 | 256 | 128,000 |

---

## Data Schemas

All pool files are **JSONL** (one JSON object per line). Score files are also JSONL.

### Pool JSONL Schemas

<details>
<summary><strong>AlpacaEval / HelpSteer pool (LLaMA-3.1-8B & Mistral-7B)</strong></summary>

```json
{
  "prompt_id": 0,
  "instruction": "What are the names of some famous actors...",
  "candidates": ["response_1", "response_2", "..."],
  "n_candidates": 256,
  "gen_time_sec": 12.4,
  "model": "/path/to/model",
  "max_new_tokens": 160,
  "temperature": 1.0,
  "top_p": 0.95,
  "repetition_penalty": 1.1,
  "slice_start": 0,
  "slice_end": 100,
  "slice_tag": "part0_100"
}
```
</details>

<details>
<summary><strong>AlpacaEval 2.0 / GSM8K / HumanEval pool (Qwen3-14B)</strong></summary>

```json
{
  "prompt_id": 0,
  "instruction": "...",
  "dataset": "alpaca_eval",
  "generator": "Qwen3-14B",
  "model_path": "/path/to/qwen3",
  "temperature": 1.0,
  "top_p": 0.95,
  "max_new_tokens": 512,
  "candidates": ["response_1", "..."]
}
```
</details>

<details>
<summary><strong>IFEval pool (Qwen3-14B)</strong></summary>

```json
{
  "prompt_id": 0,
  "key": "ifeval_key",
  "prompt": "...",
  "instruction_id_list": ["instruction_following:..."],
  "kwargs": [{}],
  "model_path": "/path/to/qwen3",
  "temperature": 1.0,
  "top_p": 0.95,
  "max_new_tokens": 512,
  "candidates": ["response_1", "..."]
}
```
</details>

<details>
<summary><strong>HumanEval pool (LLaMA-3.1-8B)</strong></summary>

```json
{
  "task_id": "HumanEval/0",
  "prompt": "def has_close_elements(numbers, threshold):\n    ...",
  "entry_point": "has_close_elements",
  "canonical_solution": "...",
  "test": "def check(candidate):...",
  "candidates": ["def has_close_elements...", "..."],
  "n_candidates": 256,
  "gen_time_sec": 8.2,
  "model": "/path/to/model",
  "temperature": 1.0,
  "top_p": 0.95,
  "repetition_penalty": 1.1,
  "max_new_tokens": 256
}
```
</details>

<details>
<summary><strong>GSM8K pool (LLaMA-3.1-8B)</strong></summary>

```json
{
  "prompt_id": 0,
  "instruction": "Janet's ducks lay 16 eggs per day...",
  "answer": "Janet sells 52×2=104 eggs...\n#### 104",
  "model_path": "/path/to/model",
  "temperature": 1.0,
  "top_p": 0.95,
  "max_new_tokens": 512,
  "candidates": ["response_1", "..."]
}
```
</details>

---

### Reward Model Score JSONL Schemas

<details>
<summary><strong>ArmoRM-Llama3-8B-v0.1</strong></summary>

```json
{ "promptid": 0, "candidx": 3, "armorm_raw": 0.123456 }
```
*(AlpacaEval/HelpSteer/GSM8K — note: `promptid` not `prompt_id`)*
</details>

<details>
<summary><strong>Skywork-Reward-Llama-3.1-8B</strong></summary>

**AlpacaEval / HelpSteer:**
```json
{
  "prompt_id": 0,
  "candidx": 3,
  "skywork_raw": 0.234567,
  "text": "response text here"
}
```

**HumanEval:**
```json
{
  "task_id": "HumanEval/0",
  "cand_idx": 3,
  "skywork_raw": 0.234567,
  "prompt": "...",
  "text": "...",
  "entry_point": "has_close_elements",
  "canonical_solution": "...",
  "test": "..."
}
```

**GSM8K:**
```json
{
  "prompt_id": 0,
  "candidx": 3,
  "skywork_raw": 0.234567,
  "answer": "#### 104",
  "text": "response text"
}
```
</details>

<details>
<summary><strong>Tulu3-RM</strong></summary>

**AlpacaEval / HelpSteer:**
```json
{
  "prompt_id": 0,
  "cand_idx": 3,
  "tulu3_raw": 0.345678,
  "instruction": "...",
  "text": "response text"
}
```

*(HelpSteer uses `promptid` instead of `prompt_id`)*
</details>

<details>
<summary><strong>FsfairX-LLaMA3-RM</strong></summary>

**HelpSteer:**
```json
{
  "promptid": 0,
  "candidx": 3,
  "score": 0.456789,
  "prompt": "...",
  "text": "response text"
}
```

**HumanEval:**
```json
{
  "task_id": "HumanEval/0",
  "cand_idx": 3,
  "fsfairx_raw": 0.456789,
  "prompt": "...",
  "text": "...",
  "entry_point": "has_close_elements",
  "canonical_solution": "...",
  "test": "..."
}
```
</details>

<details>
<summary><strong>MathShepherd (Process Reward Model — GSM8K only)</strong></summary>

```json
{
  "prompt_id": 0,
  "cand_idx": 3,
  "mathshepherd_raw": -560.82,
  "instruction": "Janet's ducks lay 16 eggs...",
  "answer": "#### 104",
  "text": "Step 1: ... Step 2: ..."
}
```
</details>

---

## Notebooks

| # | Notebook | What it demonstrates |
|---|----------|---------------------|
| 1 | [`experiment-01-llama.ipynb`](notebooks/experiment-01-llama.ipynb) | Minimal ERR reproduction — end-to-end Best-of-N sweep with ArmoRM on AlpacaEval. Easiest entry point |
| 2 | [`experiment-1-final-test-on-t2.ipynb`](notebooks/experiment-1-final-test-on-t2.ipynb) | **Main result** — 10-dimension penalization escape sweep; full escape matrix (9/10 dims escape) |
| 3 | [`gsm8k-best-of-n-candidate-pool-analysis.ipynb`](notebooks/gsm8k-best-of-n-candidate-pool-analysis.ipynb) | **Falsifiability case** — MathShepherd migrates (EM −37.6%), Skywork-RM does not (+20.6%) |
| 4 | [`humaneval-best-of-n-proxy-migration-pass-1.ipynb`](notebooks/humaneval-best-of-n-proxy-migration-pass-1.ipynb) | HumanEval pass@k decoupling from proxy reward under Best-of-N |
| 5 | [`llm-as-a-judge-qwen3-helpsteer.ipynb`](notebooks/llm-as-a-judge-qwen3-helpsteer.ipynb) | Qwen3 as independent judge on 500 HelpSteer prompts — strong true-objective signal |
| 6 | [`multi-selector-llm-as-a-judge-qwen3-helpsteer.ipynb`](notebooks/multi-selector-llm-as-a-judge-qwen3-helpsteer.ipynb) | Multi-selector comparison under LLM-as-judge — which reward models resist migration? |
| 7 | [`alpacaeval-fixed-pool-best-of-n-rm-analysis.ipynb`](notebooks/alpacaeval-fixed-pool-best-of-n-rm-analysis.ipynb) | Fixed shared pool across all reward models — cross-RM generalizability of migration |

---

## Key Results

### Type I — Direct Migration
Under unconstrained keyword density proxy, Best-of-N selection (N=1→256, LLaMA-3.1-8B, 500 prompts):
- Keyword density: **+533%**
- Flesch readability: **−8.5 points**
- ERR = **1.120** at N=256

### Type II — Penalization Escape (Migration Chain)

| Phase | Reward function | KD | Length | Formality |
|-------|----------------|-----|--------|-----------|
| Baseline | — | 0.00730 | 68.4w | 0.00490 |
| Phase 1 (Weak KD) | `R = KD(r)` | **0.01834 (+151%)** | 68.0 | 0.00506 |
| Phase 2 (KD Penalized) | `R = KD − α·max(0,KD−θ₁)` | 0.00269 | **73.7 (+5.3w)** | 0.00553 |
| Phase 3 (Fortress) | `R = Phase2 − β·len + γ·form` | 0.00187 | 68.2 | **0.01141 (+133%)** |

### Type II — Escape Matrix (10 dimensions, 500 prompts)
- **9 out of 10** penalized dimensions trigger escape into at least one other dimension
- ArmoRM raw score continues to scale monotonically under all 9 penalty conditions (Spearman ρ = 1.0)
- Strongest escape route: `certainty_density` → `formatting_complexity`

### Type III — Latent Decoupling
PPO (N=10 seeds): proxy Cohen's *d* = **3.303** (p<0.0001) alongside true-objective *d* = **−1.922** (p<0.0001).  
ArmoRM replication (500 prompts): STS *d* = **−2.200** (p<0.0001) — Type III confirmed under a learned RM.

### Type IV — Structural Collapse
PPO: **10/10** seeds collapse under symmetric reward; **0/10** under directional.  
Phase 3 "Fortress" (AlpacaEval): mean best reward = **−0.0058**, collapse rate **>20%**.

### GSM8K Falsifiability (Notebook 3)

| Selector | Migration? | EM% at N=1 | EM% at N=256 | EM% Δ |
|----------|-----------|-----------|-------------|-------|
| MathShepherd (PRM) | **YES** | 65.8% | 28.2% | **−37.6%** |
| Skywork-RM | **NO** | 40.1% | 60.7% | **+20.6%** |

MathShepherd has a proxy-true gap (step scores ≠ final answer correctness) — migration activates.  
Skywork-RM selects correct answers directly — `Δ(x) ≈ 0`, no migration.

### Mitigation: Directional Gap Loss (DGL)

$$R_{\text{DGL}}(r) = \text{KD}(r) - \alpha \cdot \max(0,\, \text{KD}(r) - \theta_1) - \beta \cdot \max(0,\, |r| - L_{\max}) + \gamma \cdot \min(\text{formality}(r),\, 0.05)$$

With `θ₁=0.01, α=8.0, β=0.04, γ=0.5, L_max=80`.

| Condition | Proxy@N=256 | Flesch | STS | Migration |
|-----------|-------------|--------|-----|-----------|
| Vanilla | 0.0328 | 45.36 | 0.7201 | Type I |
| KL Penalty | 0.0000 | 46.34 | 0.7629 | None (signal killed) |
| Length Penalty | 0.0306 | 45.43 | 0.7275 | Type I (partial) |
| **DGL** | **0.0000** | **44.98** | **0.7527** | **None** |

---

## Exploitation Rate Ratio (ERR)

$$\text{ERR}(N) = \frac{R(\hat{x}_N) - R(\hat{x}_1)}{R(\hat{x}_1) + \varepsilon}$$

ERR is detectable at **10% of training budget** — providing an early warning indicator before true-objective collapse.

---

## Quickstart

### 1. Install dependencies

```bash
pip install torch transformers accelerate bitsandbytes \
            sentence-transformers textstat scipy tqdm pandas matplotlib
```

### 2. Generate a candidate pool

```bash
python src/generate_pool_candidates.py \
    --data_path data/helpsteer_train.csv \
    --model_path mistralai/Mistral-7B-Instruct-v0.1 \
    --output_path outputs/pool_candidates.jsonl \
    --candidates 256 \
    --slice_start 0 \
    --slice_end 500
```

### 3. Score pool → run all-4-types migration analysis

```bash
python src/score_armorm_baselines.py \
    --pool_path outputs/pool_candidates.jsonl \
    --output_dir outputs/migration_results \
    --n_list 1 4 16 64 256
```

### 4. Run per-dimension escape matrix

```bash
python src/compute_multidimensional_metrics.py \
    --pool_path outputs/pool_candidates.jsonl \
    --score_path outputs/armorm_cache.jsonl \
    --output_dir outputs/escape_analysis
```

### 5. Explore interactively

Open [`notebooks/experiment-01-llama.ipynb`](notebooks/experiment-01-llama.ipynb) for a self-contained entry point, or [`notebooks/experiment-1-final-test-on-t2.ipynb`](notebooks/experiment-1-final-test-on-t2.ipynb) for the full 10-dimension escape analysis.

---

## Models Used

All models are publicly available checkpoints:

| Model | Role | Reference |
|-------|------|-----------|
| [LLaMA-3.1-8B](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B) | Generator (AlpacaEval, HelpSteer, HumanEval, GSM8K) | Meta 2024 |
| [Mistral-7B-Instruct-v0.1](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1) | Generator (AlpacaEval, HelpSteer, HumanEval, GSM8K) | Jiang et al. 2023 |
| [Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B) | Generator (AlpacaEval 2.0, IFEval, HumanEval, GSM8K) | Qwen Team 2025 |
| [ArmoRM-Llama3-8B-v0.1](https://huggingface.co/RLHFlow/ArmoRM-Llama3-8B-v0.1) | Reward model | Wang et al. 2024 |
| [Skywork-Reward-Llama-3.1-8B](https://huggingface.co/Skywork/Skywork-Reward-Llama-3.1-8B) | Reward model | — |
| [FsfairX-LLaMA3-RM-v0.1](https://huggingface.co/sfairXC/FsfairX-LLaMA3-RM-v0.1) | Reward model | — |
| [Tulu3-RM](https://huggingface.co/allenai/Llama-3.1-Tulu-3-8B-RM) | Reward model | — |
| [MathShepherd](https://huggingface.co/peiyi9979/math-shepherd-mistral-7b-prm) | Process reward model (GSM8K) | — |
| [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | STS quality signal | — |
| [Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B) | LLM-as-judge | — |

---

## Hardware

All experiments run on **consumer-grade GPUs (2× NVIDIA T4, 30 GiB VRAM total)** using NF4 4-bit quantization via `bitsandbytes`.

---

## Citation

```bibtex
@inproceedings{proxymigration2026,
  title     = {Proxy Migration in Language Models: A Taxonomy of Reward Hacking
               Under Selection and Penalization},
  author    = {Anonymous},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026},
  note      = {Under review}
}
```

---

## License

Code released under the [MIT License](LICENSE). Models and datasets retain their original licenses.
=======
# Proxy Migration

### Investigating Optimization-Induced Migration Across Imperfect Reward Dimensions in Language Models

[![Research Status](https://img.shields.io/badge/status-active%20research-orange)](#research-status)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

> **Proxy Migration** is an ongoing research program investigating how optimization pressure shifts between alternative reward-relevant dimensions when previously exploited dimensions are constrained, penalized, or otherwise made less accessible.

This repository contains the code, experimental configurations, evaluation tools, analysis, and research artifacts developed throughout the project.

---

## Research Status

**Active research — not all claims in this repository are established scientific conclusions.**

The project has evolved through multiple experimental stages. Early experiments used controlled synthetic proxy dimensions to study the basic phenomenon. Subsequent work moved toward learned neural reward models and larger candidate-pool experiments in response to limitations identified during peer review.

The repository intentionally preserves this evolution, including unsuccessful experiments, limitations, and directions that remain under investigation.

---

## Core Research Question

When an optimization process exploits one imperfect proxy, what happens when that proxy dimension is constrained?

The central hypothesis investigated in this project is:

> **Optimization pressure may not disappear when an exploited proxy is constrained; instead, it may shift toward other available reward-relevant dimensions.**

We refer to this hypothesized/observed behavior as **Proxy Migration**.

Conceptually:

```text
Optimization Pressure
        │
        ▼
   Proxy Dimension A
        │
        │  constraint / penalty
        ▼
   Proxy Dimension A
   becomes less exploitable
        │
        ▼
 Optimization searches for
 another available direction
        │
        ▼
   Proxy Dimension B
>>>>>>> 4303461e3c1029f92a5e66e8c70c1f001f23e697
