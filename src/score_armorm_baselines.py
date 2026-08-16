"""
score_armorm_baselines.py
==========================
Scores a candidate pool against all four Proxy Migration reward conditions
(Phase 1 / Phase 2 / Phase 3 / DGL) and an optional ArmoRM neural reward
model, then runs Best-of-N selection and outputs per-type migration evidence.

Refactored from: Mistral-armorm-score-proxy-rewards.ipynb
Paper: "Proxy Migration in Language Models: A Taxonomy of Reward Hacking
        Under Selection and Penalization" (NeurIPS 2026)

Outputs (in --output_dir):
    type1_direct_migration.csv
    type2_penalization_escape.csv
    type3_latent_migration.csv
    type4_structural_collapse.csv
    migration_taxonomy_summary.csv
    fig1_type1_scaling.png  ... fig5_master_summary.png

Usage:
    # Heuristic proxies only (no neural RM needed)
    python score_armorm_baselines.py \
        --pool_path outputs/pool_candidates.jsonl \
        --output_dir outputs/migration_results \
        --n_list 1 4 16 64 256 \
        --n_bon 256

    # With ArmoRM scoring (requires GPU + model path)
    python score_armorm_baselines.py \
        --pool_path outputs/pool_candidates.jsonl \
        --output_dir outputs/migration_results \
        --armorm_path /path/to/ArmoRM-Llama3-8B-v0.1 \
        --n_list 1 4 16 64 256
"""

import argparse
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import textstat
from scipy.stats import ttest_1samp
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

plt.rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#dddddd",
    "grid.linestyle": "--", "grid.linewidth": 0.5,
    "figure.facecolor": "white", "axes.facecolor": "#fafafa",
})


# ---------------------------------------------------------------------------
# Lexicons (Appendix F of the paper)
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
FORMALITY_WORDS = {
    "therefore", "furthermore", "consequently", "however", "nevertheless",
    "moreover", "thus", "hence", "accordingly", "subsequently", "additionally",
    "alternatively", "similarly", "conversely", "nonetheless", "meanwhile",
    "whereby", "whereas", "thereby", "notwithstanding",
}
FORMALITY_SUFFIXES = ("tion", "ment", "ance", "ence", "ity", "ism", "ization", "isation")


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------
def keyword_density(text: str) -> float:
    words = re.findall(r"\b\w+\b", text.lower())
    return sum(1 for w in words if w in KEYWORD_SET) / max(1, len(words))


def response_length(text: str) -> int:
    return len(text.split())


def formality_score(text: str) -> float:
    words = re.findall(r"\b\w+\b", text.lower())
    formal = sum(
        1 for w in words
        if w in FORMALITY_WORDS or w.endswith(FORMALITY_SUFFIXES)
    )
    return formal / max(1, len(words))


def flesch_score(text: str) -> float:
    try:
        return float(textstat.flesch_reading_ease(text))
    except Exception:
        return 50.0


def exploitation_rate_ratio(proxy_by_n: dict[int, float], eps: float = 1e-9) -> dict[int, float]:
    """ERR(N) = (R(x̂_N) - R(x̂_1)) / (R(x̂_1) + ε)"""
    base = list(proxy_by_n.values())[0]
    return {n: (v - base) / (base + eps) for n, v in proxy_by_n.items()}


# ---------------------------------------------------------------------------
# Reward functions (Table 7 / Equation 7 of the paper)
# ---------------------------------------------------------------------------
STUFFING_THRESHOLD = 0.01   # θ₁
ALPHA_STUFF        = 8.0    # α
VERBOSITY_CAP      = 80     # L_max
BETA_VERB          = 0.04   # β
GAMMA_FORM         = 0.5    # γ


def phase1_reward(text: str) -> float:
    """Type I — unconstrained keyword density proxy."""
    return keyword_density(text)


def phase2_reward(text: str) -> float:
    """Type II — KD with stuffing penalty; verbosity free."""
    kd  = keyword_density(text)
    pen = max(0.0, kd - STUFFING_THRESHOLD) * ALPHA_STUFF
    return kd - pen


def phase3_reward(text: str) -> float:
    """Type IV — Fortress: all dimensions penalized."""
    kd   = keyword_density(text)
    n    = response_length(text)
    form = formality_score(text)
    return (
        kd
        - max(0.0, kd - STUFFING_THRESHOLD) * ALPHA_STUFF
        - max(0.0, n  - VERBOSITY_CAP) * BETA_VERB
        - GAMMA_FORM * form
    )


def dgl_reward(text: str) -> float:
    """Directional Gap Loss — closes all free dimensions simultaneously."""
    kd   = keyword_density(text)
    n    = response_length(text)
    form = formality_score(text)
    return (
        kd
        - max(0.0, kd - STUFFING_THRESHOLD) * ALPHA_STUFF
        - max(0.0, n  - VERBOSITY_CAP) * BETA_VERB
        + min(form, 0.05) * GAMMA_FORM       # formality bonus capped at 0.025
    )


REWARD_FNS = {
    "Phase1_Weak":     phase1_reward,
    "Phase2_Hardened": phase2_reward,
    "Phase3_Fortress": phase3_reward,
    "DGL":             dgl_reward,
}


# ---------------------------------------------------------------------------
# Optional ArmoRM scoring (requires GPU + model)
# ---------------------------------------------------------------------------
def load_armorm(model_path: str):
    """Load ArmoRM neural reward model via HuggingFace transformers."""
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig
    except ImportError:
        raise ImportError("Install torch + transformers to use --armorm_path.")

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def score_armorm_batch(
    prompts: list[str],
    responses: list[str],
    model,
    tokenizer,
    batch_size: int = 16,
) -> list[float]:
    """Score (prompt, response) pairs using ArmoRM. Returns scalar scores."""
    import torch

    scores = []
    for i in range(0, len(prompts), batch_size):
        batch_p = prompts[i:i + batch_size]
        batch_r = responses[i:i + batch_size]
        chats = [
            [{"role": "user", "content": p}, {"role": "assistant", "content": r}]
            for p, r in zip(batch_p, batch_r)
        ]
        texts = [tokenizer.apply_chat_template(c, tokenize=False) for c in chats]
        enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=2048)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
        logits = out.logits.squeeze(-1)
        scores.extend(logits.float().cpu().tolist())
    return scores


# ---------------------------------------------------------------------------
# STS quality signal
# ---------------------------------------------------------------------------
def load_sts_model():
    from sentence_transformers import SentenceTransformer
    from scipy.spatial.distance import cosine as cosine_dist

    sts = SentenceTransformer("all-MiniLM-L6-v2")

    def score(text: str, reference: str) -> float:
        e1 = sts.encode([text])[0]
        e2 = sts.encode([reference])[0]
        return float(1.0 - cosine_dist(e1, e2))

    return score


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------
def load_pool(pool_path: Path) -> list[dict]:
    records = []
    with open(pool_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            pid = int(rec.get("prompt_id", rec.get("promptid", len(records))))
            records.append({
                "prompt_id":   pid,
                "instruction": rec.get("instruction", rec.get("prompt", "")),
                "candidates":  rec["candidates"],
            })
    for rec in records:
        rec["reference"] = rec["candidates"][0]  # N=1 baseline as reference
    return records


# ---------------------------------------------------------------------------
# Migration type analyses
# ---------------------------------------------------------------------------
def analyze_type_i(pool_records, n_list, sts_fn=None) -> pd.DataFrame:
    rows = []
    for N in n_list:
        for rec in tqdm(pool_records, desc=f"Type I N={N}", leave=False):
            cands = rec["candidates"][:min(N, len(rec["candidates"]))]
            best  = max(cands, key=phase1_reward)
            row = {
                "N":         N,
                "prompt_id": rec["prompt_id"],
                "kd":        keyword_density(best),
                "resp_len":  response_length(best),
                "formality": formality_score(best),
                "flesch":    flesch_score(best),
                "score":     phase1_reward(best),
            }
            if sts_fn:
                row["sts"] = sts_fn(best, rec["reference"])
            rows.append(row)
    return pd.DataFrame(rows)


def analyze_type_ii(pool_records, n_bon) -> pd.DataFrame:
    rows = []
    for rec in tqdm(pool_records, desc="Type II"):
        cands = rec["candidates"][:min(n_bon, len(rec["candidates"]))]
        for phase_name, rfn in REWARD_FNS.items():
            best = max(cands, key=rfn)
            rows.append({
                "phase":     phase_name,
                "prompt_id": rec["prompt_id"],
                "kd":        keyword_density(best),
                "resp_len":  response_length(best),
                "formality": formality_score(best),
                "flesch":    flesch_score(best),
                "score":     rfn(best),
            })
    return pd.DataFrame(rows)


def analyze_type_iii(pool_records, seeds=(42, 123, 456), n_bon=256, sts_fn=None) -> pd.DataFrame:
    rows = []
    rng_global = np.random.default_rng(42)
    for seed in seeds:
        rng = np.random.default_rng(seed)
        for rec in tqdm(pool_records, desc=f"Type III seed={seed}", leave=False):
            cands = list(rec["candidates"])
            rng.shuffle(cands)
            pool = cands[:min(n_bon, len(cands))]
            ref  = rec["reference"]
            for condition, fn, sel_cands in [
                ("Baseline_N1",  phase1_reward, [cands[0]]),
                ("Vanilla_N256", phase1_reward, pool),
                ("DGL_N256",     dgl_reward,    pool),
            ]:
                best = max(sel_cands, key=fn)
                row = {
                    "seed":      seed,
                    "prompt_id": rec["prompt_id"],
                    "condition": condition,
                    "kd":        keyword_density(best),
                    "flesch":    flesch_score(best),
                    "resp_len":  response_length(best),
                }
                if sts_fn:
                    row["sts"] = sts_fn(best, ref)
                rows.append(row)
    return pd.DataFrame(rows)


def analyze_type_iv(pool_records, n_bon) -> tuple[pd.DataFrame, float, float]:
    rows = []
    collapse_count = 0
    for rec in tqdm(pool_records, desc="Type IV"):
        cands  = rec["candidates"][:min(n_bon, len(rec["candidates"]))]
        scored = [(c, phase3_reward(c)) for c in cands]
        best_text, best_reward = max(scored, key=lambda x: x[1])
        if best_reward < 0.0:
            collapse_count += 1
        for cand, reward in scored:
            rows.append({
                "prompt_id":       rec["prompt_id"],
                "fortress_reward": reward,
                "kd":              keyword_density(cand),
                "resp_len":        response_length(cand),
                "formality":       formality_score(cand),
                "flesch":          flesch_score(cand),
                "is_best":         (cand == best_text),
            })
    df  = pd.DataFrame(rows)
    mean_reward   = df[df["is_best"]]["fortress_reward"].mean()
    collapse_rate = collapse_count / max(1, len(pool_records))
    return df, mean_reward, collapse_rate


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_all_types(df_t1, df_t2, df_t3, df_t4, n_list, out_dir: Path,
                   mean_fortress: float, collapse_rate: float):
    CONDITIONS = ["Phase1_Weak", "Phase2_Hardened", "Phase3_Fortress", "DGL"]
    LABELS     = ["Phase 1\n(Weak KD)", "Phase 2\n(KD Penalized)", "Phase 3\n(Fortress)", "DGL"]
    COLORS     = ["#CC3311", "#0077BB", "#EE7733", "#009988"]
    CONDS3     = ["Baseline_N1", "Vanilla_N256", "DGL_N256"]
    CLRS3      = ["#888888", "#CC3311", "#0077BB"]
    LABS3      = ["Baseline N=1", "Vanilla N=256", "DGL N=256"]

    agg1 = df_t1.groupby("N")[["kd", "flesch"]].mean().reset_index()

    # Figure 1 — Type I
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.plot(agg1["N"], agg1["kd"],     "o-", color="#1565C0", lw=2.5, ms=8, label="Proxy Reward (KD)")
    ax1.plot(agg1["N"], agg1["flesch"], "s--", color="#C62828", lw=2.5, ms=8, label="Flesch Readability")
    ax1.set_title("Type I — Direct Migration", fontweight="bold")
    ax1.set_xticks(n_list); ax1.legend(fontsize=9)
    ax1.set_xlabel("N (Best-of-N)"); ax1.set_ylabel("Score")
    ax2.plot(agg1["N"], agg1["kd"], "o-", color="#1565C0", lw=2, ms=7)
    ax2.set_title("Type I — KD Scaling", fontweight="bold")
    ax2.set_xticks(n_list)
    fig.suptitle("Figure 1 — Type I Direct Migration", fontsize=12, fontweight="bold")
    plt.tight_layout(); plt.savefig(out_dir / "fig1_type1_scaling.png", dpi=200, bbox_inches="tight")
    plt.close()

    # Figure 2 — Type II
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, metric, ylabel, title in [
        (axes[0], "kd",       "Keyword Density",     "Panel A — KD"),
        (axes[1], "resp_len", "Response Length",      "Panel B — Length"),
        (axes[2], "formality","Formality Score",      "Panel C — Formality"),
    ]:
        means = [df_t2[df_t2["phase"] == c][metric].mean() for c in CONDITIONS]
        stds  = [df_t2[df_t2["phase"] == c][metric].std()  for c in CONDITIONS]
        axes_i = ax
        axes_i.bar(range(4), means, color=COLORS, alpha=0.85, yerr=stds,
                   capsize=5, edgecolor="#333", linewidth=0.6, width=0.6)
        axes_i.set_xticks(range(4)); axes_i.set_xticklabels(LABELS, fontsize=8)
        axes_i.set_ylabel(ylabel); axes_i.set_title(title, fontweight="bold")
    fig.suptitle("Figure 2 — Type II Penalization Escape", fontsize=12, fontweight="bold")
    plt.tight_layout(); plt.savefig(out_dir / "fig2_type2_escape.png", dpi=200, bbox_inches="tight")
    plt.close()

    # Figure 3 — Type III (if STS available)
    if "sts" in df_t3.columns:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for ax, metric, ylabel, title in [
            (axes[0], "kd",    "Keyword Density",    "Proxy Reward"),
            (axes[1], "sts",   "STS Similarity",     "Semantic Quality"),
            (axes[2], "flesch","Flesch Score",        "Readability"),
        ]:
            means = [df_t3[df_t3["condition"] == c][metric].mean() for c in CONDS3]
            stds  = [df_t3[df_t3["condition"] == c][metric].std()  for c in CONDS3]
            ax.bar(range(3), means, yerr=stds, color=CLRS3, alpha=0.85,
                   capsize=5, edgecolor="#333", width=0.55)
            ax.set_xticks(range(3)); ax.set_xticklabels(LABS3, fontsize=9)
            ax.set_ylabel(ylabel); ax.set_title(title, fontweight="bold")
        fig.suptitle("Figure 3 — Type III Latent Migration", fontsize=12, fontweight="bold")
        plt.tight_layout(); plt.savefig(out_dir / "fig3_type3_latent.png", dpi=200, bbox_inches="tight")
        plt.close()

    # Figure 4 — Type IV
    best_rewards = df_t4[df_t4["is_best"]]["fortress_reward"].values
    bar_colors   = ["#C62828" if r < 0 else "#2E7D32" for r in best_rewards]
    show_n = min(60, len(best_rewards))
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(range(show_n), best_rewards[:show_n], color=bar_colors[:show_n],
           edgecolor="#333", linewidth=0.4, alpha=0.85)
    ax.axhline(0, color="red", lw=1.8, ls="--", label="Collapse boundary")
    ax.set_title(
        f"Type IV — Structural Collapse  |  Mean={mean_fortress:.4f}  Collapse={collapse_rate:.1%}",
        fontweight="bold",
    )
    ax.set_xlabel("Prompt"); ax.set_ylabel("Best Fortress Reward"); ax.legend()
    fig.suptitle("Figure 4 — Type IV Structural Collapse", fontsize=12, fontweight="bold")
    plt.tight_layout(); plt.savefig(out_dir / "fig4_type4_fortress.png", dpi=200, bbox_inches="tight")
    plt.close()

    print("Figures saved.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Score candidate pool against all four migration reward conditions."
    )
    parser.add_argument("--pool_path",   required=True)
    parser.add_argument("--output_dir",  required=True)
    parser.add_argument("--n_list",      nargs="+", type=int, default=[1, 4, 16, 64, 256])
    parser.add_argument("--n_bon",       type=int, default=256, help="Fixed N for phase comparison.")
    parser.add_argument("--armorm_path", default=None,
                        help="Optional: path to ArmoRM-Llama3-8B-v0.1 for neural RM scoring.")
    parser.add_argument("--no_sts", action="store_true",
                        help="Skip STS semantic similarity computation (faster).")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load pool
    print("Loading candidate pool...")
    pool_records = load_pool(Path(args.pool_path))
    print(f"  {len(pool_records)} prompts, {len(pool_records[0]['candidates'])} candidates each.")

    # STS model
    sts_fn = None
    if not args.no_sts:
        print("Loading STS model (all-MiniLM-L6-v2)...")
        sts_fn = load_sts_model()

    # ── Type I ──
    print("\n" + "="*60)
    print("TYPE I — Direct Migration")
    df_t1 = analyze_type_i(pool_records, args.n_list, sts_fn)
    df_t1.to_csv(out_dir / "type1_direct_migration.csv", index=False)
    agg1 = df_t1.groupby("N")[["kd", "flesch"]].mean()
    n1_kd, nlast_kd = agg1.iloc[0]["kd"], agg1.iloc[-1]["kd"]
    proxy_gain = (nlast_kd - n1_kd) / (n1_kd + 1e-9) * 100
    print(f"  Proxy gain N=1→{args.n_list[-1]}: {proxy_gain:.1f}%")
    print("  TYPE I CONFIRMED" if proxy_gain > 30 else "  TYPE I partial signal")

    # ERR
    err = exploitation_rate_ratio(dict(zip(agg1.index, agg1["kd"])))
    print(f"  ERR: { {k: round(v, 3) for k, v in err.items()} }")

    # ── Type II ──
    print("\n" + "="*60)
    print("TYPE II — Penalization Escape")
    df_t2 = analyze_type_ii(pool_records, args.n_bon)
    df_t2.to_csv(out_dir / "type2_penalization_escape.csv", index=False)
    agg2   = df_t2.groupby("phase")[["kd", "resp_len", "formality"]].mean()
    kd_drop  = (agg2.loc["Phase2_Hardened", "kd"] - agg2.loc["Phase1_Weak", "kd"]) / (agg2.loc["Phase1_Weak", "kd"] + 1e-9) * 100
    len_rise = agg2.loc["Phase2_Hardened", "resp_len"] - agg2.loc["Phase1_Weak", "resp_len"]
    print(f"  KD change Phase1→2: {kd_drop:.1f}%")
    print(f"  Length rise Phase1→2: {len_rise:.2f} words")
    print("  TYPE II CONFIRMED" if kd_drop < -15 and len_rise > 0 else "  TYPE II partial")

    # ── Type III ──
    print("\n" + "="*60)
    print("TYPE III — Latent Migration")
    df_t3 = analyze_type_iii(pool_records, sts_fn=sts_fn)
    df_t3.to_csv(out_dir / "type3_latent_migration.csv", index=False)
    if "sts" in df_t3.columns:
        base_sts    = df_t3[df_t3["condition"] == "Baseline_N1"]["sts"].mean()
        vanilla_sts = df_t3[df_t3["condition"] == "Vanilla_N256"]["sts"].mean()
        dgl_sts     = df_t3[df_t3["condition"] == "DGL_N256"]["sts"].mean()
        print(f"  Baseline STS={base_sts:.4f}  Vanilla={vanilla_sts:.4f}  DGL={dgl_sts:.4f}")
        print("  TYPE III CONFIRMED" if vanilla_sts < base_sts - 0.003 else "  TYPE III partial")
        print("  DGL SUPPRESSES MIGRATION" if dgl_sts > vanilla_sts else "  DGL partial suppression")

    # ── Type IV ──
    print("\n" + "="*60)
    print("TYPE IV — Structural Collapse")
    df_t4, mean_fortress, collapse_rate = analyze_type_iv(pool_records, args.n_bon)
    df_t4.to_csv(out_dir / "type4_structural_collapse.csv", index=False)
    print(f"  Mean best fortress reward: {mean_fortress:.4f}")
    print(f"  Collapse rate: {collapse_rate:.1%}")
    print("  TYPE IV CONFIRMED" if mean_fortress < 0.005 or collapse_rate > 0.2 else "  TYPE IV partial")

    # ── Summary CSV ──
    summary = pd.DataFrame([
        {"Type": "I",   "Confirmed": proxy_gain > 30,              "KeyMetric": f"Proxy gain={proxy_gain:.1f}%"},
        {"Type": "II",  "Confirmed": kd_drop < -15 and len_rise > 0, "KeyMetric": f"KD Δ={kd_drop:.1f}% | Length Δ={len_rise:.1f}w"},
        {"Type": "III", "Confirmed": ("sts" in df_t3.columns and vanilla_sts < base_sts - 0.003) if "sts" in df_t3.columns else None,
         "KeyMetric": f"Vanilla STS={vanilla_sts:.4f}" if "sts" in df_t3.columns else "STS skipped"},
        {"Type": "IV",  "Confirmed": mean_fortress < 0.005 or collapse_rate > 0.2, "KeyMetric": f"Collapse={collapse_rate:.1%}"},
    ])
    summary.to_csv(out_dir / "migration_taxonomy_summary.csv", index=False)
    print("\n" + "="*60)
    print("MIGRATION TAXONOMY SUMMARY")
    print(summary.to_string(index=False))

    # ── Figures ──
    print("\nGenerating figures...")
    plot_all_types(df_t1, df_t2, df_t3, df_t4, args.n_list, out_dir, mean_fortress, collapse_rate)

    print(f"\nAll outputs saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
