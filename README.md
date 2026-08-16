# Proxy Migration in Language Models
### Investigating Optimization-Induced Migration Across Imperfect Reward Dimensions

> **Proxy Migration** investigates how optimization pressure shifts between alternative reward-relevant dimensions when previously exploited dimensions are constrained, penalized, or otherwise made less accessible.

This repository contains the experimental code, evaluation tools, notebooks, and analysis developed for the Proxy Migration research project.

---

## Overview

Reward hacking is often treated as a single failure mode: an optimizer finds a way to increase a proxy reward while moving away from the intended objective.

Proxy Migration studies a more specific question:

> **What happens to optimization pressure when an exploited proxy dimension is constrained?**

The central hypothesis is that constraining one exploitable dimension does not necessarily eliminate reward hacking. Instead, optimization pressure may shift toward another available reward-relevant dimension.

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
````

The project investigates whether this shift is systematic, under what conditions it occurs, and how it can be observed across different models, datasets, reward signals, and evaluation settings.

---

## Repository Structure

```text
proxy-migration/
│
├── README.md
│
├── src/
│   ├── generate_pool_candidates.py
│   ├── compute_multidimensional_metrics.py
│   └── score_armorm_baselines.py
│
└── notebooks/
    ├── experiment-01-llama.ipynb
    ├── experiment-1-final-test-on-t2.ipynb
    ├── gsm8k-best-of-n-candidate-pool-analysis.ipynb
    ├── humaneval-best-of-n-proxy-migration-pass-1.ipynb
    ├── llm-as-a-judge-qwen3-helpsteer.ipynb
    ├── multi-selector-llm-as-a-judge-qwen3-helpsteer.ipynb
    └── alpacaeval-fixed-pool-best-of-n-rm-analysis.ipynb
```

---

## Candidate Pools

The experiments use **pre-generated fixed candidate pools** in JSONL format. 
Candidate pools allow the same set of generated responses to be evaluated under different reward models and selection pressures.

Across the experiments, approximately **1.5 million candidate responses** were generated across multiple models and datasets.

| Generator Model | Dataset | Prompts | Candidates / Prompt | Total Candidates |
|-----------------|---------|--------:|---------------------:|-----------------:|
| LLaMA-3.1-8B-Instruct | AlpacaEval | 500 | 256 | 128,000 |
| LLaMA-3.1-8B-Instruct | HelpSteer | 500 | 256 | 128,000 |
| LLaMA-3.1-8B-Instruct | HumanEval | 164 | 256 | 41,984 |
| LLaMA-3.1-8B-Instruct | GSM8K | 500 | 512 | 256,000 |
| Mistral-7B-Instruct | AlpacaEval | 300 | 256 | 76,800 |
| Mistral-7B-Instruct | HelpSteer | 500 | 256 | 128,000 |
| Mistral-7B-Instruct | HumanEval | 164 | 256 | 41,984 |
| Mistral-7B-Instruct | GSM8K | 500 | 512 | 256,000 |
| Qwen3-14B | AlpacaEval 2.0 | 500 | 256 | 128,000 |
| Qwen3-14B | IFEval | 541 | 256 | 138,496 |
| Qwen3-14B | HumanEval | 164 | 256 | 41,984 |
| Qwen3-14B | GSM8K | 500 | 256 | 128,000 |

The fixed-pool setup makes it possible to compare different reward signals and selection pressures on the **same underlying candidate responses**, reducing variation caused by regenerating responses for each condition.

---

## Notebooks

| # | Notebook                                                                                                               | Description                                                        |
| - | ---------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| 1 | [`experiment-01-llama.ipynb`](notebooks/experiment-01-llama.ipynb)                                                     | Minimal end-to-end Best-of-N experiment using ArmoRM on AlpacaEval |
| 2 | [`experiment-1-final-test-on-t2.ipynb`](notebooks/experiment-1-final-test-on-t2.ipynb)                                 | Multidimensional penalization and escape analysis                  |
| 3 | [`gsm8k-best-of-n-candidate-pool-analysis.ipynb`](notebooks/gsm8k-best-of-n-candidate-pool-analysis.ipynb)             | GSM8K analysis examining migration under different reward signals  |
| 4 | [`humaneval-best-of-n-proxy-migration-pass-1.ipynb`](notebooks/humaneval-best-of-n-proxy-migration-pass-1.ipynb)       | HumanEval analysis of proxy reward and functional correctness      |
| 5 | [`llm-as-a-judge-qwen3-helpsteer.ipynb`](notebooks/llm-as-a-judge-qwen3-helpsteer.ipynb)                               | Independent LLM-as-a-judge evaluation on HelpSteer                 |
| 6 | [`multi-selector-llm-as-a-judge-qwen3-helpsteer.ipynb`](notebooks/multi-selector-llm-as-a-judge-qwen3-helpsteer.ipynb) | Comparison of multiple reward selectors using an independent judge |
| 7 | [`alpacaeval-fixed-pool-best-of-n-rm-analysis.ipynb`](notebooks/alpacaeval-fixed-pool-best-of-n-rm-analysis.ipynb)     | Fixed-pool analysis across multiple reward models                  |

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

### 3. Run reward-model analysis

```bash
python src/score_armorm_baselines.py \
    --pool_path outputs/pool_candidates.jsonl \
    --output_dir outputs/migration_results \
    --n_list 1 4 16 64 256
```

### 4. Run multidimensional analysis

```bash
python src/compute_multidimensional_metrics.py \
    --pool_path outputs/pool_candidates.jsonl \
    --score_path outputs/armorm_cache.jsonl \
    --output_dir outputs/escape_analysis
```

### 5. Explore the notebooks

For a minimal entry point, open:

[`notebooks/experiment-01-llama.ipynb`](notebooks/experiment-01-llama.ipynb)

For the multidimensional analysis, open:

[`notebooks/experiment-1-final-test-on-t2.ipynb`](notebooks/experiment-1-final-test-on-t2.ipynb)

---

## Models Used

The experiments use publicly available model checkpoints.

| Model                       | Role                                 |
| --------------------------- | ------------------------------------ |
| LLaMA-3.1-8B                | Generator                            |
| Mistral-7B-Instruct-v0.1    | Generator                            |
| Qwen3-14B                   | Generator / LLM-as-judge             |
| ArmoRM-Llama3-8B-v0.1       | Reward model                         |
| Skywork-Reward-Llama-3.1-8B | Reward model                         |
| FsfairX-LLaMA3-RM-v0.1      | Reward model                         |
| Tulu3-RM                    | Reward model                         |
| MathShepherd                | Process reward model                 |
| all-MiniLM-L6-v2            | Semantic similarity / quality signal |

---

## Hardware

Experiments were conducted using consumer-grade GPU hardware, including:

* **2 × NVIDIA T4**
* Approximately **30 GiB total VRAM**
* NF4 4-bit quantization using `bitsandbytes`

The exact hardware requirements may vary depending on the model, candidate-pool size, and experiment configuration.

---

## Research Status

**Active research.**

The project has evolved through multiple experimental stages, including controlled proxy experiments, learned reward models, candidate-pool experiments, and independent quality evaluation.

The repository contains experimental code and research artifacts intended to support reproducibility and further investigation of Proxy Migration.

---

## License

Code in this repository is released under the **MIT License**.
