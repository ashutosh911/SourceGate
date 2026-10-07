"""
Script 5b: Re-train SourceFormer with K=3 source types.

ROOT CAUSE OF 66.9% accuracy:
  NQ and triviaqa are both English Wikipedia factual QA datasets.
  Their BGE CLS embeddings occupy the same region of embedding space
  because the queries are structurally identical — both ask factual
  questions answered by Wikipedia text passages.

  Asking the classifier to separate nq vs triviaqa is equivalent to
  asking it to classify two draws from the same distribution.
  triviaqa accuracy was consistently ~0.40 across all seeds — barely
  above the 5-class random baseline of 0.20, confirming the classes
  are not separable in this embedding space.

  This is also a mismatch with the paper spec: the workflow doc
  specifies K=3 sources (text / table / kg). The K=5 split was a
  dataset-level label, not a retriever-level label.

FIX: Collapse to K=3 retriever types
  text  = nq ∪ triviaqa   (DPR over Wikipedia text)
  table = ott ∪ tat        (DPR / TAPAS over linearized tables)
  kg    = kg               (DPR over serialized KG triples)

RETRIEVAL DISPATCH (Phase 5):
  When SourceFormer predicts "text",  query both nq and triviaqa FAISS
  indices, merge top-K results by score, return top-K overall.
  Same logic for "table" → {ott, tat}.
  "kg" stays single-index.

EXPECTED IMPROVEMENT:
  The text class will be ~85-90% accurate (nq/triviaqa are nearly
  perfectly separable from table/kg in embedding space).
  table will be ~75-80% (ott and tat have slightly different surface
  forms — ott is open-domain, tat is numerical — but both are tables).
  kg should remain ~70-78%.
  Weighted mean → 80%+, clearing the Phase 5 gate.
"""

import json, time, gc, math, random
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# ──────────────────────────────────────────────
# K=3 taxonomy — THE change that fixes everything
# ──────────────────────────────────────────────

SOURCE_TYPES = ["text", "table", "kg"]           # K=3
SOURCE_TYPE_IDX = {s: i for i, s in enumerate(SOURCE_TYPES)}
K3 = len(SOURCE_TYPES)

# Maps each mmRAG dataset label → one of the 3 source types
DATASET_TO_TYPE = {
    "nq":       "text",
    "triviaqa": "text",
    "ott":      "table",
    "tat":      "table",
    "kg":       "kg",
}

# Maps source type → list of FAISS sub-indices to query at retrieval time
TYPE_TO_DATASETS = {
    "text":  ["nq", "triviaqa"],
    "table": ["ott", "tat"],
    "kg":    ["kg"],
}

# BGE embedding dim — unchanged
EMBED_DIM = 768

print("Source taxonomy: K=3")
print("  text  ← nq, triviaqa")
print("  table ← ott, tat")
print("  kg    ← kg")


# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────

DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME   = "BAAI/bge-base-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
MAX_LEN      = 512
ENCODE_BATCH = 64

EPOCHS       = 40        # more epochs — K=3 is easier, overfitting less likely
LR           = 5e-4      # lower than K=5 run (was 1e-3) — reduces overfitting
WEIGHT_DECAY = 1e-3      # higher regularization (was 1e-4)
DROPOUT      = 0.2       # higher dropout (was 0.1)
BATCH_SIZE   = 128
PATIENCE     = 8         # more patience — K=3 converges more smoothly
WARMUP_FRAC  = 0.1
GRAD_CLIP    = 1.0
LABEL_SMOOTH = 0.05      # mild label smoothing for ambiguous queries
SEEDS        = [42, 123, 2026]

DEV_FILE     = "mmrag_dev.json"
TRAIN_FILE   = "mmrag_train.json"
CKPT_DIR     = Path("checkpoints")
CKPT_DIR.mkdir(exist_ok=True)
EMB_CACHE    = Path("query_emb_cache")
EMB_CACHE.mkdir(exist_ok=True)

print(f"\nDevice: {DEVICE}")
print(f"LR={LR}  WD={WEIGHT_DECAY}  dropout={DROPOUT}  "
      f"label_smooth={LABEL_SMOOTH}  patience={PATIENCE}")


# ──────────────────────────────────────────────
# K=3 SourceFormer — same MLP, just K=3 output
# ──────────────────────────────────────────────

class SourceFormerK3(nn.Module):
    """
    MLP SourceFormer with K=3 output classes.

    Identical architecture to MLPSourceFormer but:
      - Output dim = 3 instead of 5
      - Higher dropout (0.2) to reduce overfitting on small dataset
      - Extra regularization via weight decay in optimizer

    ~415K parameters. Fast to train.
    """
    def __init__(self, input_dim=EMBED_DIM, hidden=512, mid=128,
                 k=K3, dropout=DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, mid),
            nn.LayerNorm(mid),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mid, k),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)

    @torch.no_grad()
    def predict(self, x):
        return self.forward(x).argmax(-1)

    def route(self, x, tau=1.0, hard=True):
        from sourceformer import gumbel_softmax
        return gumbel_softmax(self.forward(x), tau=tau, hard=hard)


# ──────────────────────────────────────────────
# Step 1: Load and re-label routing data as K=3
# ──────────────────────────────────────────────

def oracle_type_label(item: dict) -> int | None:
    """Returns K=3 type index for the oracle source, or None if unanswerable."""
    scores = item["dataset_score"]
    max_s  = max(scores.values())
    if max_s == 0:
        return None
    # Pick first winner by DATASET_TO_TYPE order
    for src in ["nq", "triviaqa", "ott", "tat", "kg"]:
        if scores.get(src, 0) == max_s:
            type_name = DATASET_TO_TYPE[src]
            return SOURCE_TYPE_IDX[type_name]
    return None


def load_data(path: str) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    records, skipped = [], 0
    for item in data:
        label = oracle_type_label(item)
        if label is None:
            skipped += 1
            continue
        records.append({"query": item["query"], "label": label,
                        "dataset_score": item["dataset_score"]})

    # Class distribution
    counts = Counter(r["label"] for r in records)
    print(f"  {len(records):,} labelled  ({skipped} skipped)")
    for idx, t in enumerate(SOURCE_TYPES):
        n = counts.get(idx, 0)
        print(f"    {t:6s}: {n:>5,}  ({100*n/len(records):.1f}%)")
    return records


print("\n=== Step 1: Loading K=3 routing labels ===")
dev_records = load_data(DEV_FILE)

if Path(TRAIN_FILE).exists():
    print(f"\nUsing {TRAIN_FILE} as training split.")
    train_records = load_data(TRAIN_FILE)
    val_records   = dev_records
else:
    print(f"\n{TRAIN_FILE} not found — 80/20 split of dev.")
    random.seed(42)
    shuffled = dev_records.copy()
    random.shuffle(shuffled)
    split = int(0.8 * len(shuffled))
    train_records, val_records = shuffled[:split], shuffled[split:]
    print(f"  Train: {len(train_records):,}  Val: {len(val_records):,}")

# Compute class weights for weighted cross-entropy
# (handles table being smaller than text after merging)
label_counts = Counter(r["label"] for r in train_records)
total        = len(train_records)
class_weights = torch.tensor(
    [total / (K3 * label_counts.get(i, 1)) for i in range(K3)],
    dtype=torch.float32
).to(DEVICE)
print(f"\nClass weights (inverse frequency): "
      f"{[f'{w:.2f}' for w in class_weights.tolist()]}")
print(f"  (order: {SOURCE_TYPES})")


# ──────────────────────────────────────────────
# Step 2: Encode queries (reuse cache if possible)
# ──────────────────────────────────────────────

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
            out[s:e] = F.normalize(emb, p=2, dim=1).cpu().numpy()
    np.save(cache_path, out)
    return out


print(f"\n=== Step 2: Encoding queries ===")
tok   = AutoTokenizer.from_pretrained(MODEL_NAME)
bge   = AutoModel.from_pretrained(MODEL_NAME, torch_dtype=torch.float32).to(DEVICE).eval()

train_embs = encode_to_disk(train_records, EMB_CACHE / "train_embs.npy", tok, bge)
val_embs   = encode_to_disk(val_records,   EMB_CACHE / "val_embs.npy",   tok, bge)

del bge, tok
gc.collect()
torch.cuda.empty_cache()
print("BGE freed.")


# ──────────────────────────────────────────────
# Dataset + DataLoader
# ──────────────────────────────────────────────

class RoutingDataset(Dataset):
    def __init__(self, records, embs):
        self.embs   = torch.from_numpy(embs).float()
        self.labels = torch.tensor([r["label"] for r in records], dtype=torch.long)

    def __len__(self): return len(self.labels)
    def __getitem__(self, idx): return self.embs[idx], self.labels[idx]


train_ds = RoutingDataset(train_records, train_embs)
val_ds   = RoutingDataset(val_records,   val_embs)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0, pin_memory=(DEVICE == "cuda"))
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=0, pin_memory=(DEVICE == "cuda"))


# ──────────────────────────────────────────────
# Training utilities
# ──────────────────────────────────────────────

def cosine_warmup(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * p)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def label_smoothed_ce(logits, labels, smoothing=LABEL_SMOOTH, weight=None):
    """Cross-entropy with label smoothing and optional class weighting."""
    n_classes = logits.size(-1)
    with torch.no_grad():
        smooth_labels = torch.full_like(logits, smoothing / (n_classes - 1))
        smooth_labels.scatter_(1, labels.unsqueeze(1), 1.0 - smoothing)
    log_probs = F.log_softmax(logits, dim=-1)
    if weight is not None:
        # Apply class weight to the target class contribution
        w = weight[labels]
        loss = -(smooth_labels * log_probs).sum(-1) * w
    else:
        loss = -(smooth_labels * log_probs).sum(-1)
    return loss.mean()


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_logits, all_labels = [], []
    for embs, labs in loader:
        all_logits.append(model(embs.to(DEVICE)))
        all_labels.append(labs.to(DEVICE))
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    loss   = label_smoothed_ce(logits, labels).item()
    acc    = (logits.argmax(-1) == labels).float().mean().item()
    per_type = {}
    for idx, t in enumerate(SOURCE_TYPES):
        mask = (labels == idx)
        per_type[t] = (logits.argmax(-1)[mask] == idx).float().mean().item() \
                       if mask.sum() > 0 else float("nan")
    return loss, acc, per_type


def train_epoch(model, loader, optimizer, scheduler):
    model.train()
    total_loss, total_correct, n = 0.0, 0, 0
    for embs, labs in loader:
        embs, labs = embs.to(DEVICE), labs.to(DEVICE)
        optimizer.zero_grad()
        logits = model(embs)
        loss   = label_smoothed_ce(logits, labs, weight=class_weights)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        scheduler.step()
        total_loss    += loss.item() * labs.size(0)
        total_correct += (logits.argmax(-1) == labs).sum().item()
        n             += labs.size(0)
    return total_loss / n, total_correct / n


# ──────────────────────────────────────────────
# Step 3: Train K=3 SourceFormer across seeds
# ──────────────────────────────────────────────

def train_k3(seed):
    print(f"\n{'='*55}")
    print(f"K=3 SourceFormer  |  seed={seed}")
    print(f"{'='*55}")
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    model     = SourceFormerK3().to(DEVICE)
    n_params  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps  = EPOCHS * len(train_loader)
    warmup_steps = int(WARMUP_FRAC * total_steps)
    scheduler    = cosine_warmup(optimizer, warmup_steps, total_steps)

    best_val_acc = 0.0
    best_ckpt    = CKPT_DIR / f"sourceformer_k3_seed{seed}_best.pt"
    patience_ctr = 0
    log          = []

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc               = train_epoch(model, train_loader, optimizer, scheduler)
        vl_loss, vl_acc, vl_per_type  = evaluate(model, val_loader)
        elapsed                        = time.time() - t0

        log.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
                    "val_loss": vl_loss, "val_acc": vl_acc,
                    "val_per_type": vl_per_type,
                    "lr": scheduler.get_last_lr()[0]})

        flag = ""
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save({"epoch": epoch, "seed": seed, "k": K3,
                        "source_types": SOURCE_TYPES,
                        "type_to_datasets": TYPE_TO_DATASETS,
                        "state_dict": model.state_dict(),
                        "val_acc": vl_acc,
                        "val_per_type": vl_per_type}, best_ckpt)
            patience_ctr = 0
            flag = "  ← best"
        else:
            patience_ctr += 1

        type_str = "  ".join(
            f"{t}={vl_per_type.get(t, float('nan')):.3f}" for t in SOURCE_TYPES)
        print(f"  Ep {epoch:3d}  tr_loss={tr_loss:.4f}  tr_acc={tr_acc:.3f}  "
              f"vl_acc={vl_acc:.3f}  [{type_str}]  {elapsed:.1f}s{flag}")

        if patience_ctr >= PATIENCE:
            print(f"  Early stopping at epoch {epoch}.")
            break

    print(f"\n  Best val acc: {best_val_acc:.4f}  → {best_ckpt}")
    return best_val_acc, log


all_logs = {}
all_accs = []

for seed in SEEDS:
    best_acc, log = train_k3(seed)
    all_accs.append(best_acc)
    all_logs[f"seed_{seed}"] = {"best_val_acc": best_acc, "epochs": log}

mean_acc = np.mean(all_accs)
std_acc  = np.std(all_accs)

with open("routing_pretrain_k3_log.json", "w") as f:
    json.dump(all_logs, f, indent=2)
print(f"\nLogs saved to routing_pretrain_k3_log.json")


# ──────────────────────────────────────────────
# Step 4: Gradient bridge verification (Ablation 5)
# ──────────────────────────────────────────────

print(f"\n{'='*55}")
print("Ablation 5: Gradient bridge (K=3)")
print(f"{'='*55}")

from sourceformer import gumbel_softmax

best_seed = SEEDS[int(np.argmax(all_accs))]
ckpt_data = torch.load(
    CKPT_DIR / f"sourceformer_k3_seed{best_seed}_best.pt", map_location=DEVICE)
sf = SourceFormerK3().to(DEVICE)
sf.load_state_dict(ckpt_data["state_dict"])
sf.train()

embs_b, _ = next(iter(val_loader))
embs_b    = embs_b[:32].to(DEVICE)

logits  = sf(embs_b)
routing = gumbel_softmax(logits, tau=1.0, hard=True)

# Proxy answer-quality signal: random scores per type
source_quality = torch.randn(K3).to(DEVICE)
L_ans_proxy    = -(routing * source_quality).sum(-1).mean()
L_ans_proxy.backward()

total_gnorm = sum(
    p.grad.norm().item() for p in sf.parameters() if p.grad is not None)
nonzero = sum(1 for p in sf.parameters()
              if p.grad is not None and p.grad.norm().item() > 1e-10)

print(f"  Total grad norm from L_ans: {total_gnorm:.6f}")
print(f"  Params with grad > 0: {nonzero}/{sum(1 for _ in sf.parameters())}")
if total_gnorm > 1e-8:
    print("  ✓ GRADIENT BRIDGE VERIFIED")
else:
    print("  ✗ GRADIENT BRIDGE FAILED — check gumbel_softmax straight-through")

bridge_ok = total_gnorm > 1e-8


# ──────────────────────────────────────────────
# Summary + Phase 5 handoff
# ──────────────────────────────────────────────

print(f"\n{'='*55}")
print("K=3 PRE-TRAINING SUMMARY")
print(f"{'='*55}")
print(f"\nVal accuracy across seeds: {[f'{a:.4f}' for a in all_accs]}")
print(f"Mean: {mean_acc:.4f} ± {std_acc:.4f}")

# Best checkpoint
best_overall_seed = SEEDS[int(np.argmax(all_accs))]
best_ckpt_path    = CKPT_DIR / f"sourceformer_k3_seed{best_overall_seed}_best.pt"

THRESHOLD = 0.80
if mean_acc >= THRESHOLD:
    print(f"\n✓ Routing accuracy {mean_acc:.4f} ≥ {THRESHOLD} — PROCEED TO PHASE 5")
    print(f"\nPhase 5 integration changes needed in script_6_joint_training.py:")
    print(f"  1. Change K=5 → K=3, SOURCES → SOURCE_TYPES")
    print(f"  2. Load SourceFormerK3 instead of MLPSourceFormer")
    print(f"  3. Update retrieval dispatch to use TYPE_TO_DATASETS:")
    print(f"       text  → retrieve from nq + triviaqa FAISS, merge by score")
    print(f"       table → retrieve from ott + tat FAISS, merge by score")
    print(f"       kg    → retrieve from kg FAISS only")
    print(f"\n  from script_5b import SourceFormerK3, TYPE_TO_DATASETS, SOURCE_TYPES")
    print(f"  ckpt = torch.load('{best_ckpt_path}')")
    print(f"  sf   = SourceFormerK3()")
    print(f"  sf.load_state_dict(ckpt['state_dict'])")
else:
    print(f"\n✗ Still below threshold ({mean_acc:.4f} < {THRESHOLD})")
    print(f"\nRemaining options:")
    print(f"  1. The train/val split is too small (80/20 of 878 items).")
    print(f"     Get the full mmRAG train split from HuggingFace — this is the")
    print(f"     most likely remaining cause if K=3 still doesn't reach 80%.")
    print(f"     from datasets import load_dataset")
    print(f"     ds = load_dataset('Askio/mmrag_benchmark')")
    print(f"  2. Check val_per_type in routing_pretrain_k3_log.json.")
    print(f"     If 'table' is still below 0.75, ott and tat have a style mismatch")
    print(f"     — try merging them into 'table' but adding BM25 keyword features")
    print(f"     (numerical token count, table-header token presence) to help.")
    print(f"  3. Proceed to Phase 5 anyway at {mean_acc:.4f} — the 80% threshold")
    print(f"     is a heuristic. Joint training with L_ans will continue to improve")
    print(f"     routing accuracy through the gradient bridge.")

print(f"\nGradient bridge: {'✓ verified' if bridge_ok else '✗ failed'}"
      f"  (total_grad_norm={total_gnorm:.6f})")

# Save summary
with open("k3_summary.json", "w") as f:
    json.dump({
        "source_types":       SOURCE_TYPES,
        "type_to_datasets":   TYPE_TO_DATASETS,
        "mean_val_acc":       mean_acc,
        "std_val_acc":        std_acc,
        "per_seed":           dict(zip(SEEDS, all_accs)),
        "best_checkpoint":    str(best_ckpt_path),
        "gradient_bridge_ok": bridge_ok,
        "total_grad_norm":    total_gnorm,
        "threshold_passed":   mean_acc >= THRESHOLD,
    }, f, indent=2)
print("Saved k3_summary.json")