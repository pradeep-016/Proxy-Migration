"""
compute_multidimensional_metrics.py
=====================================
Computes per-candidate behavioural metrics across 10 linguistic dimensions,
runs a per-dimension penalization escape sweep, and outputs a full escape
matrix showing which dimensions migrate to which when penalized.

Refactored from: experiment-1-final-test-on-t2.ipynb
Paper: "Proxy Migration in Language Models: A Taxonomy of Reward Hacking
        Under Selection and Penalization" (NeurIPS 2026)

Key output — escape matrix (CSV):
    penalized_dimension | escaped_to_dimension | state | delta | spearman_rho | p_value

Usage:
    python compute_multidimensional_metrics.py \
        --pool_path outputs/pool_candidates.jsonl \
        --score_path outputs/armorm_cache.csv \
        --output_dir outputs/escape_analysis \
        --n_list 1 4 16 64 256 \
        --penalty_weight 0.35
"""

import argparse
import json
import math
import re
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import textstat
from scipy.stats import spearmanr
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Lexicons
# ---------------------------------------------------------------------------
KEYWORD_SET = {
    "effective", "important", "key", "great", "best", "helpful", "excellent",
    "significant", "valuable", "essential", "improve", "benefit", "result",
    "optimize", "performance", "crucial", "critical", "powerful", "robust",
    "strong", "useful", "efficient", "successful", "positive", "major",
    "notable", "remarkable", "outstanding", "superior", "ideal", "optimal",
    "productive", "impactful", "meaningful", "achieve", "enhance", "boost",
    "increase", "maximize", "support", "enable", "ensure", "provide", "offer",
}
HEDGING_WORDS = {
    "may", "might", "could", "can", "possibly", "perhaps", "arguably",
    "sometimes", "often", "typically", "generally", "likely", "unlikely",
    "suggests", "suggest", "appears", "appear", "seems", "seem", "roughly",
    "approximately", "about",
}
CERTAINTY_WORDS = {
    "definitely", "certainly", "clearly", "obviously", "always", "never",
    "must", "proves", "prove", "undeniably", "absolutely", "guaranteed", "guarantee",
}
STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "than", "that",
    "this", "these", "those", "to", "of", "in", "on", "for", "with", "as",
    "by", "is", "are", "was", "were", "be", "been", "being", "it", "its",
    "at", "from", "into", "about", "over", "after", "before", "between",
    "through", "during", "under", "above", "below", "up", "down", "out",
    "off", "again", "further", "once", "here", "there", "when", "where",
    "why", "how", "all", "any", "both", "each", "few", "more", "most",
    "other", "some", "such", "no", "nor", "not", "only", "own", "same",
    "so", "too", "very", "s", "t", "can", "will", "just", "don", "should",
    "now", "i", "you", "he", "she", "we", "they", "them", "their", "our",
    "your", "my", "me", "his", "her",
}


# ---------------------------------------------------------------------------
# Metric functions (10 behavioural dimensions)
# ---------------------------------------------------------------------------
def tokenize_words(text: str) -> list[str]:
    return re.findall(r"\b[a-zA-Z']+\b", str(text).lower())


def response_length(text: str) -> int:
    return len(str(text).split())


def repetition_score(text: str) -> float:
    toks = str(text).lower().split()
    if len(toks) < 2:
        return 0.0
    bigrams = list(zip(toks, toks[1:]))
    return 1.0 - (len(set(bigrams)) / len(bigrams))


def keyword_density(text: str) -> float:
    words = tokenize_words(text)
    return sum(1 for w in words if w in KEYWORD_SET) / max(1, len(words))


def hedging_density(text: str) -> float:
    words = tokenize_words(text)
    return sum(1 for w in words if w in HEDGING_WORDS) / max(1, len(words))


def certainty_density(text: str) -> float:
    words = tokenize_words(text)
    return sum(1 for w in words if w in CERTAINTY_WORDS) / max(1, len(words))


def list_density(text: str) -> float:
    lines = str(text).splitlines()
    bullet = sum(
        1 for ln in lines
        if re.match(r"^\s*[-*•]\s+", ln) or re.match(r"^\s*\d+[.)]\s+", ln)
    )
    return bullet / max(1, len(lines))


def formatting_complexity(text: str) -> float:
    txt   = str(text)
    lines = txt.splitlines()
    bullets = sum(
        1 for ln in lines
        if re.match(r"^\s*[-*•]\s+", ln) or re.match(r"^\s*\d+[.)]\s+", ln)
    )
    marks = txt.count("**") + txt.count("##") + txt.count("```") + txt.count("|")
    colon_lines = sum(1 for ln in lines if ":" in ln)
    return (bullets + marks + 0.25 * colon_lines) / max(1, len(lines))


def flesch_inverse(text: str) -> float:
    try:
        return -float(textstat.flesch_reading_ease(str(text)))
    except Exception:
        return -50.0


def type_token_ratio(text: str) -> float:
    words = tokenize_words(text)
    return len(set(words)) / max(1, len(words))


def lexical_rarity(text: str) -> float:
    words = [w for w in tokenize_words(text) if w not in STOPWORDS]
    if not words:
        return 0.0
    counts = Counter(words)
    return sum(1 for c in counts.values() if c == 1) / max(1, len(counts))


DIM_FUNCS: dict[str, callable] = {
    "length":               response_length,
    "repetition":           repetition_score,
    "keyword_density":      keyword_density,
    "hedging_density":      hedging_density,
    "certainty_density":    certainty_density,
    "list_density":         list_density,
    "formatting_complexity": formatting_complexity,
    "flesch_inverse":       flesch_inverse,
    "type_token_ratio":     type_token_ratio,
    "lexical_rarity":       lexical_rarity,
}
SECONDARY_DIMS = list(DIM_FUNCS.keys())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def zscore(s: pd.Series) -> pd.Series:
    std = s.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - s.mean()) / std


def safe_get_prompt_id(rec: dict) -> int:
    for key in ("prompt_id", "promptid"):
        if key in rec:
            return int(rec[key])
    raise KeyError("No prompt_id/promptid in record.")


def safe_get_instruction(rec: dict) -> str:
    for key in ("instruction", "prompt", "question", "input"):
        if key in rec:
            return rec[key]
    return ""


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------
def load_pool(pool_path: Path) -> list[dict]:
    records = []
    with open(pool_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            records.append({
                "promptid":    safe_get_prompt_id(rec),
                "instruction": safe_get_instruction(rec),
                "candidates":  rec["candidates"],
                "n_candidates": len(rec["candidates"]),
            })
    return records


def load_score_cache(score_path: Path) -> dict[tuple[int, int], float]:
    df = pd.read_csv(score_path)
    df.columns = [c.strip() for c in df.columns]
    # Normalise column names
    df = df.rename(columns={"promptid": "prompt_id", "candidx": "cand_idx",
                             "armorm_raw": "score"})
    for col in ("prompt_id", "cand_idx"):
        df[col] = df[col].astype(int)
    df["score"] = df["score"].astype(float)
    return {(int(r.prompt_id), int(r.cand_idx)): float(r.score)
            for r in df.itertuples(index=False)}


def compute_candidate_metrics(
    pool_records: list[dict],
    score_map: dict[tuple[int, int], float],
) -> pd.DataFrame:
    rows = []
    for rec in tqdm(pool_records, desc="Computing candidate metrics"):
        pid = rec["promptid"]
        for idx, cand in enumerate(rec["candidates"]):
            if (pid, idx) not in score_map:
                continue
            row = {"promptid": pid, "candidx": idx, "armorm_raw": score_map[(pid, idx)]}
            for dim, fn in DIM_FUNCS.items():
                row[dim] = fn(cand)
            rows.append(row)
    return pd.DataFrame(rows)


def run_bon_sweep(
    metrics_df: pd.DataFrame,
    n_list: list[int],
    penalty_weight: float,
) -> pd.DataFrame:
    """
    For each dimension in SECONDARY_DIMS, construct a penalized score
    (armorm_raw - penalty_weight * z_score_of_dim) and run Best-of-N
    selection across all N values. Returns a selections DataFrame.
    """
    # z-score each dimension per prompt
    zscored = metrics_df.copy()
    for dim in SECONDARY_DIMS:
        zscored[f"{dim}_z"] = zscored.groupby("promptid")[dim].transform(zscore)

    conditions = ["control"] + [f"penalize_{d}" for d in SECONDARY_DIMS]
    all_rows = []

    for pid, g in tqdm(zscored.groupby("promptid"), desc="BoN sweep"):
        g = g.sort_values("candidx").reset_index(drop=True)

        for N in n_list:
            sub = g[g["candidx"] < N]
            if sub.empty:
                continue

            # Control: unpenalized ArmoRM
            best_idx = sub["armorm_raw"].idxmax()
            best = sub.loc[best_idx].to_dict()
            best.update({"N": N, "condition": "control", "selected_score": best["armorm_raw"]})
            all_rows.append(best)

            # Per-dimension penalized
            for dim in SECONDARY_DIMS:
                pen_col = f"{dim}_z"
                scores = sub["armorm_raw"] - penalty_weight * sub[pen_col].clip(lower=0)
                best_idx = scores.idxmax()
                best = sub.loc[best_idx].to_dict()
                best.update({
                    "N": N,
                    "condition": f"penalize_{dim}",
                    "selected_score": float(scores.loc[best_idx]),
                })
                all_rows.append(best)

    return pd.DataFrame(all_rows)


def compute_escape_matrix(
    sel_df: pd.DataFrame,
    n_list: list[int],
    alpha: float = 0.05,
    min_rho: float = 0.5,
) -> pd.DataFrame:
    """
    For each (penalized_dim, other_dim) pair, determine whether the other_dim
    rises (escape) or falls when penalized_dim is constrained.
    """
    log_n = [math.log2(n) for n in n_list]
    escape_rows = []

    for pen_dim in SECONDARY_DIMS:
        cond  = f"penalize_{pen_dim}"
        block = sel_df[sel_df["condition"] == cond].groupby("N")[SECONDARY_DIMS].mean().loc[
            [n for n in n_list if n in sel_df["N"].values]
        ]

        if len(block) < 2:
            continue

        pen_n1   = block.iloc[0][pen_dim]
        pen_n256 = block.iloc[-1][pen_dim]

        for other_dim in SECONDARY_DIMS:
            if other_dim == pen_dim:
                continue

            vals = block[other_dim].tolist()
            if len(vals) != len(log_n):
                continue

            rho, p = spearmanr(log_n, vals)
            other_n1   = vals[0]
            other_n256 = vals[-1]
            delta = other_n256 - other_n1
            state = "p" if (rho >= min_rho and p < alpha and delta > 0) else "n"

            escape_rows.append({
                "penalized_dimension": pen_dim,
                "escaped_to_dimension": other_dim,
                "penalized_dim_N1":    pen_n1,
                "penalized_dim_N256":  pen_n256,
                "penalized_dim_delta": pen_n256 - pen_n1,
                "other_dim_N1":        other_n1,
                "other_dim_N256":      other_n256,
                "other_dim_delta":     delta,
                "other_dim_spearman_rho": round(rho, 4),
                "other_dim_p_value":      round(p, 4),
                "state":               state,
            })

    return pd.DataFrame(escape_rows)


def compute_verdicts(escape_df: pd.DataFrame) -> pd.DataFrame:
    verdict_rows = []
    for pen_dim in SECONDARY_DIMS:
        block    = escape_df[escape_df["penalized_dimension"] == pen_dim]
        positive = block[block["state"] == "p"]
        any_esc  = len(positive) > 0
        top_esc  = None
        if any_esc:
            top_esc = positive.sort_values(
                ["other_dim_delta", "other_dim_spearman_rho"],
                ascending=[False, False],
            ).iloc[0]["escaped_to_dimension"]
        verdict_rows.append({
            "penalized_dimension":  pen_dim,
            "any_escape_detected":  any_esc,
            "n_escape_targets":     len(positive),
            "top_escape_target":    top_esc,
        })
    return pd.DataFrame(verdict_rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Per-dimension proxy penalization escape sweep."
    )
    parser.add_argument("--pool_path",     required=True, help="JSONL candidate pool.")
    parser.add_argument("--score_path",    required=True, help="CSV with ArmoRM scores (promptid, candidx, armorm_raw).")
    parser.add_argument("--output_dir",    required=True, help="Directory for output CSVs.")
    parser.add_argument("--n_list",        nargs="+", type=int, default=[1, 4, 16, 64, 256])
    parser.add_argument("--penalty_weight", type=float, default=0.35)
    parser.add_argument("--alpha",          type=float, default=0.05,
                        help="Significance threshold for escape detection.")
    parser.add_argument("--min_rho",        type=float, default=0.5,
                        help="Minimum Spearman rho for escape confirmation.")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading pool...")
    pool_records = load_pool(Path(args.pool_path))
    print(f"  {len(pool_records)} prompts loaded.")

    print("Loading score cache...")
    score_map = load_score_cache(Path(args.score_path))
    print(f"  {len(score_map)} scored candidates.")

    # Compute per-candidate metrics
    print("\nComputing per-candidate metrics (10 dimensions)...")
    metrics_df = compute_candidate_metrics(pool_records, score_map)
    metrics_df.to_csv(out_dir / "candidate_metrics.csv", index=False)
    print(f"  Saved candidate_metrics.csv  shape={metrics_df.shape}")

    # Best-of-N sweep
    print("\nRunning per-dimension penalization sweep...")
    sel_df = run_bon_sweep(metrics_df, args.n_list, args.penalty_weight)
    sel_df.to_csv(out_dir / "selections.csv", index=False)
    print(f"  Saved selections.csv  shape={sel_df.shape}")

    # Escape matrix
    print("\nComputing escape matrix...")
    escape_df = compute_escape_matrix(sel_df, args.n_list, args.alpha, args.min_rho)
    escape_df.to_csv(out_dir / "escape_matrix.csv", index=False)
    print(f"  Saved escape_matrix.csv  shape={escape_df.shape}")

    # Verdicts
    verdict_df = compute_verdicts(escape_df)
    verdict_df.to_csv(out_dir / "verdicts.csv", index=False)

    # Summary printout
    n_escape = int(verdict_df["any_escape_detected"].sum())
    n_total  = len(verdict_df)
    print(f"\n{'='*60}")
    print(f"Escape Matrix Summary: {n_escape}/{n_total} penalized dimensions trigger escape")
    print(verdict_df.to_string(index=False))
    print(f"{'='*60}")
    print(f"\nAll outputs written to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
