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
