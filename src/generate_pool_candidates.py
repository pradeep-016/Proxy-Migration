"""
generate_pool_candidates.py
============================
Generates a fixed candidate pool (Best-of-N candidates per prompt) for any
instruction dataset and saves it as a JSONL file for downstream migration
analysis.

Refactored from: helpsteer-fixed-pool-part-a.ipynb
Paper: "Proxy Migration in Language Models: A Taxonomy of Reward Hacking
        Under Selection and Penalization" (NeurIPS 2026)

Usage:
    python generate_pool_candidates.py \
        --data_path data/helpsteer_train.csv \
        --model_path /path/to/mistral-7b-instruct \
        --output_path outputs/pool_candidates.jsonl \
        --candidates 256 \
        --slice_start 0 \
        --slice_end 500

Output format (JSONL, one record per prompt):
    {
        "promptid": 0,
        "instruction": "...",
        "candidates": ["response_1", "response_2", ...],
        "ncandidates": 256,
        "gen_time_sec": 12.4,
        "model": "/path/to/model",
        ...
    }
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

# ---------------------------------------------------------------------------
# Instruction template (Mistral-style; adapt for other models)
# ---------------------------------------------------------------------------
INST_TEMPLATE = "[INST] {instruction} [/INST]"


# ---------------------------------------------------------------------------
# Config defaults (override via CLI args)
# ---------------------------------------------------------------------------
DEFAULT_CANDIDATES    = 256
DEFAULT_MAX_NEW_TOKENS = 160
DEFAULT_TEMPERATURE   = 1.0
DEFAULT_TOP_P         = 0.95
DEFAULT_REP_PENALTY   = 1.1
DEFAULT_SEED          = 42
DEFAULT_MIN_WORDS     = 10
DEFAULT_MAX_WORDS     = 60
DEFAULT_PILOT_N       = 3     # number of prompts for ETA estimation


# ---------------------------------------------------------------------------
# Pool generation
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate_pool(
    instruction: str,
    model,
    tokenizer,
    n: int = DEFAULT_CANDIDATES,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    rep_penalty: float = DEFAULT_REP_PENALTY,
) -> list[str]:
    """Generate `n` candidate responses for a single instruction."""
    prompt  = INST_TEMPLATE.format(instruction=instruction)
    prompts = [prompt] * n
    inputs  = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(model.device)

    out = model.generate(
        **inputs,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=rep_penalty,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    new_tokens = out[:, inputs["input_ids"].shape[1]:]
    texts = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
    return [t.strip() for t in texts]


def eta_hours(elapsed: float, n_done: int, n_total: int) -> float:
    if n_done == 0:
        return float("inf")
    return (elapsed / n_done) * (n_total - n_done) / 3600


# ---------------------------------------------------------------------------
# Data loading & filtering
# ---------------------------------------------------------------------------
def load_and_filter_prompts(
    data_path: str,
    min_words: int = DEFAULT_MIN_WORDS,
    max_words: int = DEFAULT_MAX_WORDS,
    slice_start: int = 0,
    slice_end: int | None = None,
) -> pd.DataFrame:
    """
    Load instruction dataset CSV, deduplicate on 'prompt'/'instruction',
    filter by word count, slice, and assign stable promptids.
    """
    df_raw = pd.read_csv(data_path)
    print("Columns:", df_raw.columns.tolist())
    print("Shape:", df_raw.shape)

    # Normalise column name
    if "prompt" in df_raw.columns and "instruction" not in df_raw.columns:
        df_raw = df_raw.rename(columns={"prompt": "instruction"})

    df = (
        df_raw[["instruction"]]
        .drop_duplicates("instruction")
        .reset_index(drop=True)
    )
    df["nwords"] = df["instruction"].str.split().str.len()
    df = df[(df["nwords"] >= min_words) & (df["nwords"] <= max_words)].reset_index(drop=True)
    df["promptid"] = df.index

    print(f"Total unique prompts after filtering: {len(df)}")

    # Slice AFTER filtering — keeps promptids stable across dataset shards
    end = slice_end if slice_end is not None else len(df)
    df  = df.iloc[slice_start:end].reset_index(drop=True)
    print(f"Prompts in this slice [{slice_start}:{end}]: {len(df)}")

    if len(df) == 0:
        raise ValueError("Slice is empty — check --slice_start / --slice_end.")
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Generate Best-of-N candidate pool for proxy migration analysis."
    )
    parser.add_argument("--data_path",    required=True, help="Path to instruction CSV (must have 'prompt' or 'instruction' column).")
    parser.add_argument("--model_path",   required=True, help="Path to HuggingFace model (local or hub ID).")
    parser.add_argument("--output_path",  required=True, help="Output JSONL path for the candidate pool.")
    parser.add_argument("--candidates",   type=int,   default=DEFAULT_CANDIDATES,    help="Number of candidates per prompt.")
    parser.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--temperature",  type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top_p",        type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--rep_penalty",  type=float, default=DEFAULT_REP_PENALTY)
    parser.add_argument("--seed",         type=int,   default=DEFAULT_SEED)
    parser.add_argument("--min_words",    type=int,   default=DEFAULT_MIN_WORDS)
    parser.add_argument("--max_words",    type=int,   default=DEFAULT_MAX_WORDS)
    parser.add_argument("--slice_start",  type=int,   default=0)
    parser.add_argument("--slice_end",    type=int,   default=None)
    parser.add_argument("--pilot_n",      type=int,   default=DEFAULT_PILOT_N,
                        help="Number of prompts to use for ETA estimation.")
    parser.add_argument("--no_4bit", action="store_true",
                        help="Disable 4-bit NF4 quantization (load in fp16 instead).")
    args = parser.parse_args()

    # Reproducibility
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

    # Output setup
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    timing_path = output_path.with_suffix(".timing.json")

    # Load prompts
    df = load_and_filter_prompts(
        args.data_path,
        min_words=args.min_words,
        max_words=args.max_words,
        slice_start=args.slice_start,
        slice_end=args.slice_end,
    )

    # Resume support — skip already-generated prompts
    done_ids: set[int] = set()
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["promptid"])
                except Exception:
                    pass
        print(f"Resuming — {len(done_ids)} prompts already done.")

    remaining = df[~df["promptid"].isin(done_ids)].reset_index(drop=True)
    print(f"Remaining: {len(remaining)} prompts to generate.")

    if len(remaining) == 0:
        print("All prompts already generated. Nothing to do.")
        return

    # Load model
    print(f"Loading model from: {args.model_path}")
    if args.no_4bit:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.float16,
            device_map="auto",
        )
    else:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            quantization_config=bnb,
            device_map="auto",
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model.eval()
    print("Model loaded ✅")

    # Generation loop
    pilot_times: list[float] = []
    session_start = time.time()

    with open(output_path, "a", encoding="utf-8") as fout:
        for loop_idx, (_, row) in enumerate(
            tqdm(remaining.iterrows(), total=len(remaining), desc="Generating pool")
        ):
            pid   = int(row["promptid"])
            instr = row["instruction"]
            t0    = time.time()

            try:
                cands = generate_pool(
                    instr, model, tokenizer,
                    n=args.candidates,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    rep_penalty=args.rep_penalty,
                )
            except torch.cuda.OutOfMemoryError:
                print(f"OOM at prompt {pid}. Try reducing --candidates.")
                torch.cuda.empty_cache()
                raise

            elapsed = time.time() - t0
            pilot_times.append(elapsed)

            record = dict(
                promptid=pid,
                instruction=instr,
                candidates=cands,
                ncandidates=len(cands),
                gen_time_sec=round(elapsed, 2),
                model=args.model_path,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.rep_penalty,
                slice_start=args.slice_start,
                slice_end=args.slice_end,
            )
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()

            # ETA estimate after pilot prompts
            if loop_idx == args.pilot_n - 1:
                total_elapsed = time.time() - session_start
                avg_sec = sum(pilot_times) / len(pilot_times)
                eta = eta_hours(total_elapsed, args.pilot_n, len(remaining))
                print("=" * 55)
                print(f"Pilot done | avg {avg_sec:.1f} sec/prompt")
                print(f"ETA for this slice: {eta:.1f} hrs")
                print("=" * 55)
                with open(timing_path, "w") as tf:
                    json.dump({
                        "avg_sec_per_prompt": round(avg_sec, 2),
                        "eta_hours": round(eta, 2),
                        "n_remaining": len(remaining),
                        "candidates_per_prompt": args.candidates,
                        "max_new_tokens": args.max_new_tokens,
                    }, tf, indent=2)

    total_hours = (time.time() - session_start) / 3600
    print(f"\nDone. Total time: {total_hours:.2f} hrs")
    print(f"Pool saved to: {output_path}")


if __name__ == "__main__":
    main()
