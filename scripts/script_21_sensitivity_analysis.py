"""
script_21_sensitivity_analysis.py  —  Per-type weight sensitivity analysis

PURPOSE (R1 #10)
  R1 flags that per-type routing weights (alpha_text=0.4, alpha_table=0.5,
  alpha_kg=0.7) appear heuristic. This script sweeps alpha_kg across
  [0.4, 0.5, 0.6, 0.7, 0.8, 0.9] and kg_upsample across [1.5, 2.0, 3.0, 4.0, 5.0],
  holding all other hyperparameters fixed at paper defaults.

RUNTIME
  Phase 3 only (no LLM). ~15 min per seed on RTX 5070 Ti.
  Full sweep: 2 × ~6 configs × 3 seeds × 15 min ≈ 4.5 hours.

USAGE
  python script_21_sensitivity_analysis.py              # both sweeps, 3 seeds
  python script_21_sensitivity_analysis.py --fast       # 1 seed each (~1.5h)
  python script_21_sensitivity_analysis.py --sweep alpha
  python script_21_sensitivity_analysis.py --sweep sample
"""

import argparse, json, gc, math, random, time
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

from sourceformer import (
    SourceFormerK3, SOURCE_TYPES, SOURCE_TYPE_IDX, K, EMBED_DIM,
)

# ── config (matches script_5b_k3_FINAL.py exactly) ───────────────────────────
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME   = "BAAI/bge-base-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
MAX_LEN      = 512
ENCODE_BATCH = 64
EPOCHS       = 50
LR           = 5e-4
WEIGHT_DECAY = 1e-3
DROPOUT      = 0.2
BATCH_SIZE   = 128
PATIENCE     = 15
MIN_DELTA    = 0.005
WARMUP_FRAC  = 0.1
GRAD_CLIP    = 1.0
DEV_FILE     = "mmrag_dev.json"
TRAIN_FILE   = "mmrag_train.json"
EMB_CACHE    = Path("query_emb_cache"); EMB_CACHE.mkdir(exist_ok=True)
RESULTS_DIR  = Path("phase5_results");  RESULTS_DIR.mkdir(exist_ok=True)

# ── paper defaults ────────────────────────────────────────────────────────────
DEFAULT_ALPHA      = {"text": 0.4, "table": 0.5, "kg": 0.7}
DEFAULT_KG_UPSAMPLE = 3.0
ALPHA_KG_GRID      = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
KG_UPSAMPLE_GRID   = [1.5, 2.0, 3.0, 4.0, 5.0]
SEEDS              = [42, 123, 2026]

# ── data loading (mirrors script_5b_k3_FINAL.py) ─────────────────────────────
import json as _json

def soft_target(item):
    ds = item.get("dataset_score", {})
    by_type = {}
    for src, v in ds.items():
        t = ("text" if src in ("nq","triviaqa")
             else "table" if src in ("ott","tat")
             else "kg")
        by_type[t] = max(by_type.get(t, 0.0), float(v))
    vec = np.array([by_type.get(t, 0.0) for t in SOURCE_TYPES], dtype=np.float32)
    total = vec.sum()
    return vec / total if total > 0 else None

def load_data(path):
    with open(path) as f:
        data = _json.load(f)
    records = []
    for item in data:
        soft = soft_target(item)
        if soft is None:
            continue
        records.append({
            "query":      item["query"],
            "soft_label": soft,
            "hard_label": int(np.argmax(soft)),
        })
    return records

def encode_to_disk(records, cache_path, tok, model):
    if cache_path.exists():
        return np.load(cache_path)
    queries = [QUERY_PREFIX + r["query"] for r in records]
    n = len(queries)
    out = np.empty((n, EMBED_DIM), dtype=np.float32)
    with torch.inference_mode():
        for s in tqdm(range(0, n, ENCODE_BATCH), desc="encoding"):
            e = min(s + ENCODE_BATCH, n)
            enc = tok(queries[s:e], padding=True, truncation=True,
                      max_length=MAX_LEN, return_tensors="pt").to(DEVICE)
            emb = model(**enc).last_hidden_state[:, 0]
            out[s:e] = F.normalize(emb.float(), p=2, dim=1).cpu().numpy()
    np.save(cache_path, out)
    return out

class RoutingDataset(Dataset):
    def __init__(self, records, embs):
        self.embs = torch.from_numpy(embs).float()
        self.soft = torch.tensor(
            np.stack([r["soft_label"] for r in records]), dtype=torch.float32)
        self.hard = torch.tensor([r["hard_label"] for r in records], dtype=torch.long)
    def __len__(self): return len(self.hard)
    def __getitem__(self, i):
        return self.embs[i], self.soft[i], self.hard[i]

# ── LR schedule ──────────────────────────────────────────────────────────────
def cosine_warmup(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * p)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# ── evaluation ───────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_logits, all_hard, all_soft = [], [], []
    for embs, soft, hard in loader:
        all_logits.append(model(embs.to(DEVICE)))
        all_hard.append(hard.to(DEVICE))
        all_soft.append(soft.to(DEVICE))
    logits = torch.cat(all_logits)
    hard   = torch.cat(all_hard)
    soft   = torch.cat(all_soft)
    preds  = logits.argmax(-1)
    per_type = {}
    for idx, t in enumerate(SOURCE_TYPES):
        mask = (hard == idx)
        per_type[t] = (preds[mask] == idx).float().mean().item() if mask.sum() > 0 else float("nan")
    valid = [v for v in per_type.values() if not np.isnan(v)]
    macro = float(np.mean(valid)) if valid else float("nan")
    return macro, per_type

# ── weighted loss (alpha per source type) ────────────────────────────────────
def make_weighted_loss(alpha_map):
    alpha_t = torch.tensor(
        [alpha_map[s] for s in SOURCE_TYPES], dtype=torch.float32).to(DEVICE)
    def loss_fn(logits, soft, hard):
        log_pred = F.log_softmax(logits, dim=-1)
        kl_per_sample = -(soft * log_pred).sum(-1)          # (B,)
        weights = alpha_t[hard]                              # (B,)
        return (weights * kl_per_sample).mean()
    return loss_fn

# ── training ─────────────────────────────────────────────────────────────────
def train_one_config(seed, alpha_map, kg_upsample,
                     train_records, dev_records,
                     train_embs, val_embs):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    upsample_w = {
        SOURCE_TYPE_IDX["text"]:  1.0,
        SOURCE_TYPE_IDX["table"]: 1.5,
        SOURCE_TYPE_IDX["kg"]:    kg_upsample,
    }
    sample_weights = torch.tensor(
        [upsample_w[r["hard_label"]] for r in train_records], dtype=torch.float32)
    sampler = WeightedRandomSampler(
        sample_weights, len(sample_weights), replacement=True)

    train_ds = RoutingDataset(train_records, train_embs)
    val_ds   = RoutingDataset(dev_records,   val_embs)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                               num_workers=0, pin_memory=(DEVICE=="cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                               num_workers=0, pin_memory=(DEVICE=="cuda"))

    loss_fn = make_weighted_loss(alpha_map)
    model   = SourceFormerK3(dropout=DROPOUT).to(DEVICE)
    opt     = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps  = EPOCHS * len(train_loader)
    warmup_steps = int(WARMUP_FRAC * total_steps)
    sched = cosine_warmup(opt, warmup_steps, total_steps)

    best_macro, patience_ctr = 0.0, 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for embs, soft, hard in train_loader:
            embs, soft, hard = embs.to(DEVICE), soft.to(DEVICE), hard.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(embs), soft, hard)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step(); sched.step()

        macro, per_type = evaluate(model, val_loader)
        if macro > best_macro + MIN_DELTA:
            best_macro = macro
            patience_ctr = 0
        else:
            patience_ctr += 1
        if patience_ctr >= PATIENCE:
            break

    return best_macro, per_type

# ── sweep runner ─────────────────────────────────────────────────────────────
def run_sweep(sweep_name, configs, train_records, dev_records,
              train_embs, val_embs, seeds):
    results = {}
    for label, alpha_map, kg_upsample in configs:
        seed_macros, seed_per_type = [], []
        print(f"\n  [{sweep_name}] {label}")
        for seed in seeds:
            t0 = time.time()
            macro, per_type = train_one_config(
                seed, alpha_map, kg_upsample,
                train_records, dev_records, train_embs, val_embs)
            elapsed = time.time() - t0
            seed_macros.append(macro)
            seed_per_type.append(per_type)
            print(f"    seed {seed}: macro={macro:.4f}  "
                  f"text={per_type.get('text',float('nan')):.3f}  "
                  f"table={per_type.get('table',float('nan')):.3f}  "
                  f"kg={per_type.get('kg',float('nan')):.3f}  "
                  f"({elapsed/60:.1f}min)")
        results[label] = {
            "macro_mean": float(np.mean(seed_macros)),
            "macro_std":  float(np.std(seed_macros, ddof=1) if len(seed_macros)>1 else 0.0),
            "per_seed":   seed_macros,
        }
        print(f"  -> mean={np.mean(seed_macros):.4f} ± "
              f"{np.std(seed_macros, ddof=1) if len(seed_macros)>1 else 0.0:.4f}")
    return results

# ── main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", choices=["alpha","sample","both"], default="both")
    ap.add_argument("--fast", action="store_true", help="1 seed only (~1.5h)")
    args = ap.parse_args()
    seeds = [42] if args.fast else SEEDS

    print("=" * 60)
    print("Phase 3 sensitivity analysis (R1 #10)")
    print(f"  Seeds: {seeds}  |  Device: {DEVICE}")
    print("=" * 60)

    print("\nLoading data...")
    dev_records   = load_data(DEV_FILE)
    train_records = load_data(TRAIN_FILE)

    print("Encoding embeddings (reuses cache)...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    bge = AutoModel.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16).to(DEVICE).eval()
    train_embs = encode_to_disk(
        train_records, EMB_CACHE/"train_embs_k3.npy", tok, bge)
    val_embs   = encode_to_disk(
        dev_records,   EMB_CACHE/"dev_embs_k3.npy",   tok, bge)
    del bge, tok; gc.collect(); torch.cuda.empty_cache()

    summary = {"paper_defaults": {
        "alpha": DEFAULT_ALPHA,
        "kg_upsample": DEFAULT_KG_UPSAMPLE,
        "paper_dev_macro": 0.735,
    }}

    if args.sweep in ("alpha", "both"):
        print("\n" + "="*60)
        print("SWEEP 1: alpha_kg  (alpha_text=0.4, alpha_table=0.5 fixed)")
        configs = [
            (f"alpha_kg={v:.1f}" + ("  [paper]" if v == 0.7 else ""),
             {"text": 0.4, "table": 0.5, "kg": v},
             DEFAULT_KG_UPSAMPLE)
            for v in ALPHA_KG_GRID
        ]
        res = run_sweep("alpha_kg", configs, train_records, dev_records,
                        train_embs, val_embs, seeds)
        summary["alpha_kg_sweep"] = res
        with open(RESULTS_DIR/"sensitivity_alpha_kg.json", "w") as f:
            json.dump(res, f, indent=2)

    if args.sweep in ("sample", "both"):
        print("\n" + "="*60)
        print("SWEEP 2: kg_upsample  (all alpha at paper defaults)")
        configs = [
            (f"kg_upsample={v:.1f}" + ("  [paper]" if v == 3.0 else ""),
             DEFAULT_ALPHA,
             v)
            for v in KG_UPSAMPLE_GRID
        ]
        res = run_sweep("kg_upsample", configs, train_records, dev_records,
                        train_embs, val_embs, seeds)
        summary["kg_upsample_sweep"] = res
        with open(RESULTS_DIR/"sensitivity_kg_upsample.json", "w") as f:
            json.dump(res, f, indent=2)

    out = RESULTS_DIR / "sensitivity_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*60)
    print("SENSITIVITY SUMMARY")
    print("="*60)
    for sweep_key in ("alpha_kg_sweep", "kg_upsample_sweep"):
        if sweep_key not in summary:
            continue
        print(f"\n{sweep_key}:")
        print(f"  {'Config':<32} {'Macro mean':>12} {'Std':>8}")
        print("  " + "-"*54)
        for label, v in summary[sweep_key].items():
            flag = "  ← paper" if "[paper]" in label else ""
            clean = label.replace("  [paper]", "")
            print(f"  {clean:<32} {v['macro_mean']:>12.4f} {v['macro_std']:>8.4f}{flag}")
    print(f"\nSaved → {out}")

if __name__ == "__main__":
    main()
