"""
script_11_learned_baselines.py — Learned Routing Baselines for SourceFormer

PURPOSE:
  Implements two learned baselines on the same BGE embeddings to isolate
  SourceFormer's contributions (soft KL labels, weighted sampling, Gumbel bridge).

  Baseline 1: Logistic Regression (sklearn) — linear probe on BGE embeddings
  Baseline 2: MLP with CrossEntropy on hard labels — same architecture as
              SourceFormer but standard classification (no soft labels, no
              weighted sampling)

  Both train on mmrag_train.json and evaluate on mmrag_test.json.
  Routing metrics only (no LLM generation needed).

USAGE:
  python script_11_learned_baselines.py

  # With a subset for quick testing
  python script_11_learned_baselines.py --n 200

OUTPUT:
  phase5_results/learned_baselines_metrics.json
  phase5_results/learned_baselines_comparison.txt
"""

import json
import time
import gc
import math
import random
import argparse
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from sourceformer import (
    SourceFormerK3, SOURCE_TYPES, SOURCE_TYPE_IDX, K,
    DATASET_TO_TYPE, TYPE_TO_DATASETS, EMBED_DIM,
)

# =============================================================================
# Config
# =============================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "BAAI/bge-base-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
MAX_LEN = 512
ENCODE_BATCH = 64

TRAIN_FILE = "mmrag_train.json"
DEV_FILE = "mmrag_dev.json"
TEST_FILE = "mmrag_test.json"
EMB_CACHE = Path("query_emb_cache"); EMB_CACHE.mkdir(exist_ok=True)
CKPT_DIR = Path("checkpoints")
PHASE4_CKPT_DIR = Path("checkpoints_phase4")
RESULTS_DIR = Path("phase5_results"); RESULTS_DIR.mkdir(exist_ok=True)

# MLP baseline hyperparameters — IDENTICAL to SourceFormer except loss
EPOCHS = 50
LR = 5e-4
WEIGHT_DECAY = 1e-3
DROPOUT = 0.2
BATCH_SIZE = 128
PATIENCE = 15
MIN_DELTA = 0.005
WARMUP_FRAC = 0.1
GRAD_CLIP = 1.0
MLP_SEEDS = [42, 123, 2026]  # same seeds as SourceFormer


# =============================================================================
# Data loading (same label construction as script_5b)
# =============================================================================
def aggregated_type_scores(item):
    ts = {t: 0.0 for t in SOURCE_TYPES}
    for src, s in item["dataset_score"].items():
        if src in DATASET_TO_TYPE:
            ts[DATASET_TO_TYPE[src]] += float(s)
    return ts


def soft_target(item):
    ts = aggregated_type_scores(item)
    total = sum(ts.values())
    if total == 0:
        return None
    return np.array([ts[t] / total for t in SOURCE_TYPES], dtype=np.float32)


def load_records(path):
    with open(path) as f:
        data = json.load(f)
    records, skipped = [], 0
    for item in data:
        soft = soft_target(item)
        if soft is None:
            skipped += 1
            continue
        records.append({
            "query": item["query"],
            "soft_label": soft,
            "hard_label": int(np.argmax(soft)),
        })
    counts = Counter(r["hard_label"] for r in records)
    print(f"  {path}: {len(records):,} labelled ({skipped} skipped)")
    for idx, t in enumerate(SOURCE_TYPES):
        n = counts.get(idx, 0)
        print(f"    {t:6s}: {n:>5,} ({100*n/len(records):.1f}%)")
    return records


# =============================================================================
# Embedding encoding (reuses caches from script_5b)
# =============================================================================
def encode_to_disk(records, cache_path):
    """Encode queries with BGE, caching to disk."""
    if cache_path.exists():
        print(f"  Cache hit: {cache_path}")
        return np.load(cache_path)
    from transformers import AutoTokenizer, AutoModel
    print(f"  Encoding {len(records)} queries to {cache_path}...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    bge = AutoModel.from_pretrained(MODEL_NAME, torch_dtype=torch.float16).to(DEVICE).eval()
    queries = [QUERY_PREFIX + r["query"] for r in records]
    n = len(queries)
    out = np.empty((n, EMBED_DIM), dtype=np.float32)
    with torch.inference_mode():
        for s in tqdm(range(0, n, ENCODE_BATCH), desc="encoding"):
            e = min(s + ENCODE_BATCH, n)
            enc = tok(queries[s:e], padding=True, truncation=True,
                      max_length=MAX_LEN, return_tensors="pt").to(DEVICE)
            emb = bge(**enc).last_hidden_state[:, 0]
            emb = F.normalize(emb.float(), p=2, dim=1)
            out[s:e] = emb.cpu().numpy()
    np.save(cache_path, out)
    del bge, tok; gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return out


# =============================================================================
# Evaluation (routing metrics only — no LLM needed)
# =============================================================================
def eval_routing(preds, labels, method_name):
    """Compute routing accuracy, macro, per-type accuracy."""
    n = len(labels)
    acc = float((preds == labels).mean())
    per_type = {}
    for idx, t in enumerate(SOURCE_TYPES):
        mask = (labels == idx)
        if mask.sum() > 0:
            per_type[t] = float((preds[mask] == labels[mask]).mean())
        else:
            per_type[t] = float("nan")
    valid = [v for v in per_type.values() if not np.isnan(v)]
    macro = float(np.mean(valid)) if valid else float("nan")

    metrics = {
        "method": method_name,
        "n": n,
        "routing_acc": acc,
        "routing_macro": macro,
        "routing_per_type_acc": per_type,
    }
    type_str = "  ".join(f"{t}={per_type[t]:.3f}" for t in SOURCE_TYPES)
    print(f"  {method_name:<30s} acc={acc:.4f}  macro={macro:.4f}  [{type_str}]")
    return metrics


# =============================================================================
# Baseline 1: Logistic Regression
# =============================================================================
def run_logistic_regression(train_embs, train_labels, test_embs, test_labels,
                            dev_embs, dev_labels):
    """L2-regularized logistic regression on frozen BGE embeddings."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    print("\n" + "=" * 70)
    print("BASELINE 1: Logistic Regression on BGE embeddings")
    print("=" * 70)

    # Standardize features (important for logistic regression)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_embs)
    X_dev = scaler.transform(dev_embs)
    X_test = scaler.transform(test_embs)

    # Grid search over regularization strength
    best_C, best_macro = None, 0.0
    for C in [0.01, 0.1, 1.0, 10.0, 100.0]:
        clf = LogisticRegression(
            C=C, max_iter=1000, solver="lbfgs",
            multi_class="multinomial", random_state=42,
        )
        clf.fit(X_train, train_labels)
        dev_preds = clf.predict(X_dev)
        m = eval_routing(dev_preds, dev_labels, f"LR (C={C}, dev)")
        if m["routing_macro"] > best_macro:
            best_macro = m["routing_macro"]
            best_C = C

    print(f"\n  Best C={best_C} (dev macro={best_macro:.4f})")

    # Retrain with best C and evaluate on test
    clf = LogisticRegression(
        C=best_C, max_iter=1000, solver="lbfgs",
        multi_class="multinomial", random_state=42,
    )
    clf.fit(X_train, train_labels)
    test_preds = clf.predict(X_test)
    test_metrics = eval_routing(test_preds, test_labels, "Logistic Regression")
    test_metrics["best_C"] = best_C
    test_metrics["dev_macro"] = best_macro

    # Save decisions for downstream F1/EM/NLL scoring via script_10's
    # --decisions_file mechanism. Sanity check after running: routing_acc/
    # routing_macro printed above should match Table 5 (0.704 / 0.642,
    # KG accuracy 0.419) -- if they don't, something upstream (embeddings,
    # data split, sklearn version) has changed since the original run.
    decisions_path = RESULTS_DIR / "lr_decisions_k3.npy"
    np.save(decisions_path, test_preds.astype(np.int32))
    print(f"  Saved decisions -> {decisions_path}")

    return test_metrics


# =============================================================================
# Baseline 2: MLP with hard cross-entropy (same arch as SourceFormer)
# =============================================================================
class HardCEDataset(Dataset):
    def __init__(self, embs, labels):
        self.embs = torch.from_numpy(embs).float()
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embs[idx], self.labels[idx]


def cosine_warmup(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * p)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def evaluate_mlp(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    total_loss, n = 0.0, 0
    for embs, labels in loader:
        embs, labels = embs.to(DEVICE), labels.to(DEVICE)
        logits = model(embs)
        loss = F.cross_entropy(logits, labels)
        total_loss += loss.item() * labels.size(0)
        all_preds.append(logits.argmax(-1).cpu())
        all_labels.append(labels.cpu())
        n += labels.size(0)
    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    acc = float((preds == labels).mean())
    per_type = {}
    for idx, t in enumerate(SOURCE_TYPES):
        mask = (labels == idx)
        if mask.sum() > 0:
            per_type[t] = float((preds[mask] == labels[mask]).mean())
    valid = [v for v in per_type.values() if not np.isnan(v)]
    macro = float(np.mean(valid)) if valid else float("nan")
    return total_loss / n, acc, macro, per_type


def train_mlp_hard_ce(seed, train_embs, train_labels, dev_embs, dev_labels):
    """
    Train SourceFormerK3 architecture with standard cross-entropy on hard labels.
    NO soft labels. NO weighted sampling. Everything else identical.
    """
    print(f"\n  --- MLP-HardCE seed={seed} ---")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    train_ds = HardCEDataset(train_embs, train_labels)
    dev_ds = HardCEDataset(dev_embs, dev_labels)

    # Standard shuffle — NO WeightedRandomSampler
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=(DEVICE == "cuda"))
    dev_loader = DataLoader(dev_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=(DEVICE == "cuda"))

    model = SourceFormerK3(dropout=DROPOUT).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = EPOCHS * len(train_loader)
    warmup_steps = int(WARMUP_FRAC * total_steps)
    scheduler = cosine_warmup(optimizer, warmup_steps, total_steps)

    best_val_macro = 0.0
    patience_ctr = 0
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        tr_loss, tr_correct, tr_n = 0.0, 0, 0
        for embs, labels in train_loader:
            embs, labels = embs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            logits = model(embs)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            scheduler.step()
            tr_loss += loss.item() * labels.size(0)
            tr_correct += (logits.argmax(-1) == labels).sum().item()
            tr_n += labels.size(0)

        vl_loss, vl_acc, vl_macro, vl_per_type = evaluate_mlp(model, dev_loader)
        flag = ""
        if vl_macro > best_val_macro + MIN_DELTA:
            best_val_macro = vl_macro
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
            flag = " ← best"
        else:
            patience_ctr += 1

        if epoch <= 5 or epoch % 10 == 0 or flag or patience_ctr >= PATIENCE:
            type_str = "  ".join(f"{t}={vl_per_type.get(t, 0):.3f}" for t in SOURCE_TYPES)
            print(f"    Ep {epoch:3d} tr_loss={tr_loss/tr_n:.4f} "
                  f"vl_acc={vl_acc:.3f} macro={vl_macro:.3f} [{type_str}]{flag}")

        if patience_ctr >= PATIENCE:
            print(f"    Early stopping at epoch {epoch}")
            break

    print(f"    Best dev macro: {best_val_macro:.4f}")
    return best_state, best_val_macro


def run_mlp_hard_ce(train_embs, train_labels, test_embs, test_labels,
                    dev_embs, dev_labels):
    """Train MLP-HardCE across seeds and evaluate on test."""
    print("\n" + "=" * 70)
    print("BASELINE 2: MLP with Hard Cross-Entropy (same arch as SourceFormer)")
    print("=" * 70)
    print("  Architecture: Linear(768→512) LN GELU Drop → Linear(512→128) LN GELU Drop → Linear(128→3)")
    print("  Loss: CrossEntropy on hard argmax labels")
    print("  Sampling: uniform (NO weighted sampling)")
    print("  This isolates SourceFormer's soft-label + weighted-sampling contributions.\n")

    per_seed_metrics = []

    for seed in MLP_SEEDS:
        best_state, dev_macro = train_mlp_hard_ce(
            seed, train_embs, train_labels, dev_embs, dev_labels
        )

        # Evaluate on test
        model = SourceFormerK3(dropout=DROPOUT).to(DEVICE)
        model.load_state_dict(best_state)
        model.eval()

        with torch.no_grad():
            test_embs_t = torch.from_numpy(test_embs).float().to(DEVICE)
            test_preds = model(test_embs_t).argmax(-1).cpu().numpy()

        m = eval_routing(test_preds, test_labels, f"MLP-HardCE seed={seed}")
        m["seed"] = seed
        m["dev_macro"] = dev_macro
        per_seed_metrics.append(m)

        # Save only the first seed's decisions, matching the convention
        # already used for SG-Phase3/4 elsewhere in this project (routing
        # accuracy/macro in Table 4/5 are still the 3-seed mean±std; this
        # single-seed array is only for downstream F1/EM/NLL scoring via
        # script_10's --decisions_file, which needs one array, not three).
        if seed == MLP_SEEDS[0]:
            decisions_path = RESULTS_DIR / "mlp_hardce_decisions_k3.npy"
            np.save(decisions_path, test_preds.astype(np.int32))
            print(f"  Saved seed={seed} decisions -> {decisions_path}")

        del model

    # Aggregate across seeds
    macros = [m["routing_macro"] for m in per_seed_metrics]
    accs = [m["routing_acc"] for m in per_seed_metrics]
    agg = {
        "method": "MLP-HardCE",
        "n_seeds": len(per_seed_metrics),
        "routing_macro": float(np.mean(macros)),
        "routing_macro_std": float(np.std(macros)),
        "routing_acc": float(np.mean(accs)),
        "routing_acc_std": float(np.std(accs)),
        "per_seed": per_seed_metrics,
    }

    # Aggregate per-type
    agg["routing_per_type_acc"] = {}
    for t in SOURCE_TYPES:
        vals = [m["routing_per_type_acc"][t] for m in per_seed_metrics]
        agg["routing_per_type_acc"][t] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
        }

    print(f"\n  MLP-HardCE aggregate: macro={agg['routing_macro']:.4f}±{agg['routing_macro_std']:.4f}")
    return agg


# =============================================================================
# Load SourceFormer checkpoints for comparison
# =============================================================================
def eval_sourceformer_checkpoints(test_embs, test_labels):
    """Load Phase 3 + Phase 4 checkpoints and evaluate routing on test set."""
    results = {}

    for phase, ckpt_dir, prefix in [
        ("Phase3", CKPT_DIR, "sourceformer_k3_seed"),
        ("Phase4", PHASE4_CKPT_DIR, "phase4_seed"),
    ]:
        per_seed = []
        for seed in [42, 123, 2026]:
            ckpt_path = ckpt_dir / f"{prefix}{seed}_best.pt"
            if not ckpt_path.exists():
                print(f"  [skip] {phase} seed {seed} — {ckpt_path} not found")
                continue
            ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
            sf = SourceFormerK3().to(DEVICE)
            sf.load_state_dict(ckpt["state_dict"])
            sf.eval()
            with torch.no_grad():
                preds = sf(torch.from_numpy(test_embs).float().to(DEVICE)).argmax(-1).cpu().numpy()
            m = eval_routing(preds, test_labels, f"SF {phase} seed={seed}")
            m["seed"] = seed
            per_seed.append(m)
            del sf

        if per_seed:
            macros = [m["routing_macro"] for m in per_seed]
            accs = [m["routing_acc"] for m in per_seed]
            results[phase] = {
                "method": f"SourceFormer {phase}",
                "routing_macro": float(np.mean(macros)),
                "routing_macro_std": float(np.std(macros)),
                "routing_acc": float(np.mean(accs)),
                "routing_acc_std": float(np.std(accs)),
                "n_seeds": len(per_seed),
            }
            # Per-type
            results[phase]["routing_per_type_acc"] = {}
            for t in SOURCE_TYPES:
                vals = [m["routing_per_type_acc"][t] for m in per_seed]
                results[phase]["routing_per_type_acc"][t] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                }

    return results


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=None,
                        help="Limit test eval to first N queries")
    args = parser.parse_args()

    print("=" * 70)
    print("Learned Routing Baselines for SourceFormer")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    train_records = load_records(TRAIN_FILE)
    dev_records = load_records(DEV_FILE)
    test_records = load_records(TEST_FILE)

    train_labels = np.array([r["hard_label"] for r in train_records])
    dev_labels = np.array([r["hard_label"] for r in dev_records])
    test_labels = np.array([r["hard_label"] for r in test_records])

    if args.n:
        test_records = test_records[:args.n]
        test_labels = test_labels[:args.n]

    # Load / compute embeddings
    print("\nLoading embeddings...")
    train_embs = encode_to_disk(train_records, EMB_CACHE / "train_embs.npy")
    dev_embs = encode_to_disk(dev_records, EMB_CACHE / "val_embs.npy")
    test_embs = encode_to_disk(test_records, EMB_CACHE / "test_embs.npy")
    if args.n:
        test_embs = test_embs[:args.n]

    # --- Baselines ---
    print("\n\nBGE-confidence baseline (from prior results):")
    print("  macro=0.661 (K=3, deterministic — no seeds needed)\n")

    lr_metrics = run_logistic_regression(
        train_embs, train_labels, test_embs, test_labels, dev_embs, dev_labels
    )

    mlp_metrics = run_mlp_hard_ce(
        train_embs, train_labels, test_embs, test_labels, dev_embs, dev_labels
    )

    # --- SourceFormer checkpoints ---
    print("\n" + "=" * 70)
    print("SourceFormer (for comparison)")
    print("=" * 70)
    sf_metrics = eval_sourceformer_checkpoints(test_embs, test_labels)

    # --- Final comparison table ---
    print("\n\n" + "=" * 80)
    print("ROUTING BASELINE COMPARISON (test set)")
    print("=" * 80)
    print(f"{'Method':<35} {'Acc':>8} {'Macro':>10} {'Text':>8} {'Table':>8} {'KG':>8}")
    print("-" * 80)

    rows = []

    # Fixed baselines
    rows.append(("Random", 0.333, 0.337, {"text": 0.333, "table": 0.333, "kg": 0.333}))
    rows.append(("Majority (text)", 0.558, 0.333, {"text": 1.000, "table": 0.000, "kg": 0.000}))
    rows.append(("BGE-confidence", 0.740, 0.661, {"text": 0.85, "table": 0.73, "kg": 0.40}))

    # Logistic Regression
    lr_pt = lr_metrics["routing_per_type_acc"]
    rows.append(("Logistic Regression", lr_metrics["routing_acc"],
                 lr_metrics["routing_macro"], lr_pt))

    # MLP-HardCE
    mlp_pt = {t: mlp_metrics["routing_per_type_acc"][t]["mean"]
              for t in SOURCE_TYPES}
    rows.append((f"MLP-HardCE (n={mlp_metrics['n_seeds']})",
                 mlp_metrics["routing_acc"], mlp_metrics["routing_macro"], mlp_pt))

    # SourceFormer
    for phase in ["Phase3", "Phase4"]:
        if phase in sf_metrics:
            m = sf_metrics[phase]
            pt = {t: m["routing_per_type_acc"][t]["mean"] for t in SOURCE_TYPES}
            rows.append((f"SourceFormer {phase} (n={m['n_seeds']})",
                         m["routing_acc"], m["routing_macro"], pt))

    for name, acc, macro, pt in rows:
        if isinstance(pt, dict) and "text" in pt:
            pt_vals = pt
        else:
            pt_vals = pt
        text_v = pt_vals.get("text", float("nan"))
        table_v = pt_vals.get("table", float("nan"))
        kg_v = pt_vals.get("kg", float("nan"))
        print(f"{name:<35} {acc:>8.3f} {macro:>10.3f} {text_v:>8.3f} {table_v:>8.3f} {kg_v:>8.3f}")

    print("=" * 80)

    # Key insight
    if "Phase3" in sf_metrics:
        sf_macro = sf_metrics["Phase3"]["routing_macro"]
        lr_macro = lr_metrics["routing_macro"]
        mlp_macro = mlp_metrics["routing_macro"]
        print(f"\nSourceFormer Phase 3 advantage over baselines:")
        print(f"  vs BGE-confidence:      +{sf_macro - 0.661:.3f} macro")
        print(f"  vs Logistic Regression:  +{sf_macro - lr_macro:.3f} macro")
        print(f"  vs MLP-HardCE:           +{sf_macro - mlp_macro:.3f} macro")
        print(f"\nKey: SourceFormer's soft KL labels + weighted sampling explain")
        print(f"the gap over MLP-HardCE (same architecture, different training).")

    # Save results
    all_results = {
        "logistic_regression": lr_metrics,
        "mlp_hard_ce": mlp_metrics,
        "sourceformer": sf_metrics,
        "bge_confidence": {"routing_acc": 0.740, "routing_macro": 0.661},
    }
    out_path = RESULTS_DIR / "learned_baselines_metrics.json"
    with open(out_path, "w") as f:
        def safe(o):
            if isinstance(o, dict): return {k: safe(v) for k, v in o.items()}
            if isinstance(o, list): return [safe(v) for v in o]
            if isinstance(o, (np.integer,)): return int(o)
            if isinstance(o, (np.floating, np.float64)): return float(o)
            if isinstance(o, np.ndarray): return o.tolist()
            return o
        json.dump(safe(all_results), f, indent=2)
    print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    main()
