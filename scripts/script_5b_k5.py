"""
script_5b_k5.py: Phase 3 pretraining for K=5 source routing.

K=5 routes to individual datasets: nq, triviaqa, ott, tat, kg.
All four patches applied from the start.
Seeds: 7 and 99 (independent from K=3 seeds 42/123/2026).
"""

import json, time, gc, math, random
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
    SourceFormerK5, gumbel_softmax,
    SOURCE_TYPES_K5, SOURCE_TYPE_IDX_K5, K5,
    TYPE_TO_DATASET_K5, EMBED_DIM,
)

# ---------- utils ----------
def to_json_safe(obj):
    if isinstance(obj, dict): return {k: to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list): return [to_json_safe(v) for v in obj]
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.bool_): return bool(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, torch.Tensor): return obj.item() if obj.numel()==1 else obj.tolist()
    return obj

print(f"Source taxonomy: K={K5}")
for t in SOURCE_TYPES_K5:
    print(f"  {t} → {TYPE_TO_DATASET_K5[t]}")

# ---------- config ----------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "BAAI/bge-base-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
MAX_LEN = 512
ENCODE_BATCH = 64
EPOCHS = 40
LR = 5e-4
WEIGHT_DECAY = 1e-3
DROPOUT = 0.2
BATCH_SIZE = 128
PATIENCE = 15
MIN_DELTA = 0.005
WARMUP_FRAC = 0.1
GRAD_CLIP = 1.0
SEEDS = [7, 99]         # new seeds, independent from K=3
# Allow override via --seeds CLI arg (parsed after data is loaded)
import argparse as _ap
_parser = _ap.ArgumentParser(add_help=False)
_parser.add_argument("--seeds", type=int, nargs="+", default=None)
_cli_args, _ = _parser.parse_known_args()
if _cli_args.seeds is not None:
    SEEDS = _cli_args.seeds
DEV_FILE = "mmrag_dev.json"
TRAIN_FILE = "mmrag_train.json"
CKPT_DIR = Path("checkpoints"); CKPT_DIR.mkdir(exist_ok=True)
EMB_CACHE = Path("query_emb_cache_k5"); EMB_CACHE.mkdir(exist_ok=True)

# K=5 dataset→type mapping (for label derivation)
DATASET_TO_TYPE_K5 = {s: s for s in SOURCE_TYPES_K5}  # identity at K=5


# ---------- soft labels at K=5 ----------
def soft_target_k5(item):
    """Soft target over 5 datasets directly from dataset_score."""
    scores = item["dataset_score"]
    vec = np.array(
        [float(scores.get(t, 0.0)) for t in SOURCE_TYPES_K5],
        dtype=np.float32
    )
    total = vec.sum()
    if total == 0:
        return None
    return vec / total


def load_data(path):
    with open(path) as f:
        data = json.load(f)
    records, skipped = [], 0
    for item in data:
        soft = soft_target_k5(item)
        if soft is None:
            skipped += 1
            continue
        records.append({
            "query": item["query"],
            "soft_label": soft,
            "hard_label": int(np.argmax(soft)),
        })
    counts = Counter(r["hard_label"] for r in records)
    print(f"  {len(records):,} labelled ({skipped} skipped)")
    for idx, t in enumerate(SOURCE_TYPES_K5):
        n = counts.get(idx, 0)
        print(f"    {t:10s}: {n:>5,} ({100*n/len(records):.1f}%)")
    return records


print("\n=== Step 1: Loading K=5 routing labels ===")
dev_records = load_data(DEV_FILE)
train_records = load_data(TRAIN_FILE)


# ---------- baselines ----------
print("\n=== Step 1b: Baselines ===")
val_hard = np.array([r["hard_label"] for r in dev_records])
counts_val = Counter(val_hard.tolist())
majority_class = int(max(counts_val, key=counts_val.get))
majority_acc = (val_hard == majority_class).mean()
majority_macro = 1.0 / K5   # majority macro is always 1/K
rng = np.random.default_rng(42)
random_acc = (rng.integers(0, K5, size=len(val_hard)) == val_hard).mean()
unambiguous = sum(1 for r in dev_records if (r["soft_label"] > 0.99).any())
ambiguity_rate = 1.0 - unambiguous / len(dev_records)
print(f"  Majority ({SOURCE_TYPES_K5[majority_class]}): overall={majority_acc:.4f}  macro={majority_macro:.4f}")
print(f"  Random overall: {random_acc:.4f}")
print(f"  Label ambiguity: {ambiguity_rate:.3f}")


# ---------- encoding ----------
def encode_to_disk(records, cache_path, tok, model):
    if cache_path.exists():
        print(f"  Cache hit: {cache_path}")
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


print(f"\n=== Step 2: Encoding queries ===")
tok = AutoTokenizer.from_pretrained(MODEL_NAME)
bge = AutoModel.from_pretrained(MODEL_NAME, torch_dtype=torch.float16).to(DEVICE).eval()
train_embs = encode_to_disk(train_records, EMB_CACHE/"train_embs_k5.npy", tok, bge)
val_embs = encode_to_disk(dev_records, EMB_CACHE/"val_embs_k5.npy", tok, bge)
del bge, tok; gc.collect(); torch.cuda.empty_cache()


# ---------- dataset ----------
class RoutingDataset(Dataset):
    def __init__(self, records, embs):
        self.embs = torch.from_numpy(embs).float()
        self.soft = torch.tensor(np.stack([r["soft_label"] for r in records]),
                                 dtype=torch.float32)
        self.hard = torch.tensor([r["hard_label"] for r in records], dtype=torch.long)

    def __len__(self): return len(self.hard)

    def __getitem__(self, idx):
        return self.embs[idx], self.soft[idx], self.hard[idx]


# Upsampling: TAT (label=3) and KG (label=4) are underrepresented
# nq=787, triviaqa=705, ott=564, tat=349, kg=667
UPSAMPLE_WEIGHTS_K5 = {
    SOURCE_TYPE_IDX_K5["nq"]:       1.0,
    SOURCE_TYPE_IDX_K5["triviaqa"]: 1.0,
    SOURCE_TYPE_IDX_K5["ott"]:      1.5,
    SOURCE_TYPE_IDX_K5["tat"]:      2.5,  # smallest, most underrepresented
    SOURCE_TYPE_IDX_K5["kg"]:       2.0,
}

sample_weights = torch.tensor([
    UPSAMPLE_WEIGHTS_K5[r["hard_label"]] for r in train_records
], dtype=torch.float32)
sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

train_ds = RoutingDataset(train_records, train_embs)
val_ds = RoutingDataset(dev_records, val_embs)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0,
                          pin_memory=(DEVICE=="cuda"))
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
                        pin_memory=(DEVICE=="cuda"))


# ---------- KL loss ----------
def kl_soft_loss(logits, soft_targets):
    log_pred = F.log_softmax(logits, dim=-1)
    return -(soft_targets * log_pred).sum(-1).mean()


# ---------- LR schedule ----------
def cosine_warmup(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * p)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------- evaluation ----------
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_logits, all_hard, all_soft = [], [], []
    for embs, soft, hard in loader:
        all_logits.append(model(embs.to(DEVICE)))
        all_hard.append(hard.to(DEVICE))
        all_soft.append(soft.to(DEVICE))
    logits = torch.cat(all_logits)
    hard = torch.cat(all_hard)
    soft = torch.cat(all_soft)
    loss = kl_soft_loss(logits, soft).item()
    preds = logits.argmax(-1)
    acc = (preds == hard).float().mean().item()
    per_type = {}
    for idx, t in enumerate(SOURCE_TYPES_K5):
        mask = (hard == idx)
        per_type[t] = (preds[mask] == idx).float().mean().item() if mask.sum()>0 else float("nan")
    valid = [v for v in per_type.values() if not np.isnan(v)]
    macro = float(np.mean(valid)) if valid else float("nan")
    return loss, acc, per_type, macro


def train_epoch(model, loader, optimizer, scheduler):
    model.train()
    total_loss, total_correct, n = 0.0, 0, 0
    for embs, soft, hard in loader:
        embs, soft, hard = embs.to(DEVICE), soft.to(DEVICE), hard.to(DEVICE)
        optimizer.zero_grad()
        logits = model(embs)
        loss = kl_soft_loss(logits, soft)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step(); scheduler.step()
        total_loss += loss.item() * hard.size(0)
        total_correct += (logits.argmax(-1) == hard).sum().item()
        n += hard.size(0)
    return total_loss / n, total_correct / n


def train_k5(seed):
    print(f"\n{'='*55}\nK=5 SourceFormer | seed={seed}\n{'='*55}")
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    model = SourceFormerK5(dropout=DROPOUT).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = EPOCHS * len(train_loader)
    warmup_steps = int(WARMUP_FRAC * total_steps)
    scheduler = cosine_warmup(optimizer, warmup_steps, total_steps)
    best_val_macro = 0.0
    best_ckpt = CKPT_DIR / f"sourceformer_k5_seed{seed}_best.pt"
    patience_ctr = 0
    log = []
    for epoch in range(1, EPOCHS+1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, scheduler)
        vl_loss, vl_acc, vl_per_type, vl_macro = evaluate(model, val_loader)
        elapsed = time.time() - t0
        log.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
                    "val_loss": vl_loss, "val_acc": vl_acc, "val_macro": vl_macro,
                    "val_per_type": vl_per_type, "lr": scheduler.get_last_lr()[0]})
        flag = ""
        if vl_macro > best_val_macro + MIN_DELTA:
            best_val_macro = vl_macro
            torch.save({
                "epoch": epoch, "seed": seed, "k": K5,
                "source_types": SOURCE_TYPES_K5,
                "type_to_dataset": TYPE_TO_DATASET_K5,
                "state_dict": model.state_dict(),
                "val_acc": vl_acc, "val_macro": vl_macro,
                "val_per_type": vl_per_type,
            }, best_ckpt)
            patience_ctr = 0; flag = "  ← best (macro)"
        else:
            patience_ctr += 1
        type_str = "  ".join(f"{t}={vl_per_type.get(t, float('nan')):.3f}"
                             for t in SOURCE_TYPES_K5)
        print(f"  Ep {epoch:3d} tr_loss={tr_loss:.4f} tr_acc={tr_acc:.3f} "
              f"vl_acc={vl_acc:.3f} macro={vl_macro:.3f} [{type_str}] {elapsed:.1f}s{flag}")
        if patience_ctr >= PATIENCE:
            print(f"  Early stopping at epoch {epoch}."); break
    print(f"\n  Best macro: {best_val_macro:.4f} → {best_ckpt}")
    return best_val_macro, log


all_logs = {}; all_results = []
for seed in SEEDS:
    best_macro, log = train_k5(seed)
    all_results.append({"seed": seed, "macro": best_macro})
    all_logs[f"seed_{seed}"] = {"best_val_macro": best_macro, "epochs": log}

mean_macro = np.mean([r["macro"] for r in all_results])
std_macro = np.std([r["macro"] for r in all_results])

with open("routing_pretrain_k5_log.json", "w") as f:
    json.dump(to_json_safe(all_logs), f, indent=2)

# Gradient bridge check
print(f"\n{'='*55}\nGradient bridge check (K=5)\n{'='*55}")
best_seed = all_results[int(np.argmax([r["macro"] for r in all_results]))]["seed"]
ckpt_data = torch.load(CKPT_DIR / f"sourceformer_k5_seed{best_seed}_best.pt", map_location=DEVICE)
sf = SourceFormerK5(dropout=DROPOUT).to(DEVICE)
sf.load_state_dict(ckpt_data["state_dict"]); sf.train()
embs_b, _, _ = next(iter(val_loader))
embs_b = embs_b[:32].to(DEVICE)
logits = sf(embs_b)
routing = gumbel_softmax(logits, tau=1.0, hard=True)
L_ans_proxy = -(routing * torch.randn(K5).to(DEVICE)).sum(-1).mean()
L_ans_proxy.backward()
total_gnorm = sum(p.grad.norm().item() for p in sf.parameters() if p.grad is not None)
bridge_ok = total_gnorm > 1e-8
print(f"  Total grad norm: {total_gnorm:.6f}")
print("  ✓ BRIDGE VERIFIED" if bridge_ok else "  ✗ BRIDGE FAILED")

print(f"\n{'='*55}\nK=5 SUMMARY\n{'='*55}")
print(f"Seeds: {[r['seed'] for r in all_results]}")
macro_strs = [f"{r['macro']:.4f}" for r in all_results]
print(f"Macros: {macro_strs}")
print(f"Mean: {mean_macro:.4f} ± {std_macro:.4f}")
print(f"Majority baseline macro: {majority_macro:.4f}")
print(f"Lift: {mean_macro - majority_macro:+.4f}")

with open("k5_summary.json", "w") as f:
    json.dump(to_json_safe({
        "k": K5, "source_types": SOURCE_TYPES_K5,
        "mean_val_macro": float(mean_macro), "std_val_macro": float(std_macro),
        "per_seed": all_results,
        "best_checkpoint": str(CKPT_DIR / f"sourceformer_k5_seed{best_seed}_best.pt"),
        "gradient_bridge_ok": bridge_ok,
        "baselines": {"majority_macro": float(majority_macro), "random_acc": float(random_acc),
                      "ambiguity_rate": float(ambiguity_rate)},
        "lift_over_majority_macro": float(mean_macro - majority_macro),
        "upsample_weights": {SOURCE_TYPES_K5[k]: v for k, v in UPSAMPLE_WEIGHTS_K5.items()},
    }), f, indent=2)
print("Saved k5_summary.json")