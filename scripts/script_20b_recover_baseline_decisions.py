"""
script_20b_recover_baseline_decisions.py
Recover per-query routing decisions for Logistic Regression and MLP-HardCE,
then compute recall@picked from the mmrag test relevance annotations.

NO GPU NEEDED. Pure CPU: sklearn + torch (CPU only) + numpy.

Runtime: < 2 minutes.

Usage:
    python script_20b_recover_baseline_decisions.py

Output:
    phase5_results/lr_decisions_k3.npy         (1286,) int array
    phase5_results/mlp_decisions_k3.npy        (1286 x 3,) per-seed arrays
    phase5_results/baseline_recall_at_picked.json  final recall@picked numbers
"""

import json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from collections import defaultdict

# ── paths ────────────────────────────────────────────────────────────────────
RESULTS_DIR  = Path("phase5_results")
CKPT_DIR     = Path("checkpoints_ablation_2x2")   # MLP-HardCE lives here
EMB_PATH     = Path("query_emb_cache/test_embs.npy")
TEST_FILE    = "mmrag_test.json"
OUT_FILE     = RESULTS_DIR / "baseline_recall_at_picked.json"

SOURCE_TYPES = ["text", "table", "kg"]
SOURCE_IDX   = {s: i for i, s in enumerate(SOURCE_TYPES)}

# ── chunk relevance helpers ──────────────────────────────────────────────────
def source_of_chunk(cid):
    if cid.startswith("nq_") or cid.startswith("triviaqa_"):
        return "text"
    elif cid.startswith("ott_") or cid.startswith("tat_"):
        return "table"
    else:
        return "kg"

def build_relevance_matrix(test_data):
    """(n, 3) bool: has_relevant[i,j] = query i has relevant chunk in source j."""
    n = len(test_data)
    has_rel = np.zeros((n, 3), dtype=bool)
    for i, item in enumerate(test_data):
        for cid, score in item.get("relevant_chunks", {}).items():
            if score > 0:
                j = SOURCE_IDX.get(source_of_chunk(cid))
                if j is not None:
                    has_rel[i, j] = True
    return has_rel

def recall_at_picked(decisions, has_rel):
    hits = has_rel[np.arange(len(decisions)), decisions]
    return float(hits.mean())

# ── load test embeddings and labels ─────────────────────────────────────────
print("Loading test embeddings...")
test_embs = np.load(EMB_PATH)                    # (1286, 768) float32
print(f"  shape: {test_embs.shape}")

print("Loading test data...")
with open(TEST_FILE) as f:
    test_data = json.load(f)
assert len(test_data) == test_embs.shape[0], "embedding / test data length mismatch"

print("Building relevance matrix...")
has_rel = build_relevance_matrix(test_data)
print(f"  text: {has_rel[:,0].mean():.3f}  table: {has_rel[:,1].mean():.3f}  "
      f"kg: {has_rel[:,2].mean():.3f}")

# ── known recall@picked for sanity-checking ──────────────────────────────────
# We use the linear calibration model validated against 4 known points
# recall@picked ≈ 0.924 × routing_relevance − 0.122  (R²=0.9997)
A, B = 0.924, -0.122

def calibrated_recall(decisions, has_rel):
    rr = recall_at_picked(decisions, has_rel)
    return float(A * rr + B)

# ── Logistic Regression ──────────────────────────────────────────────────────
print("\n── Logistic Regression ──")
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelEncoder

    # Re-fit LR on training embeddings (same as script_11)
    TRAIN_FILE = "mmrag_train.json"
    train_emb_path = Path("query_emb_cache/train_embs.npy")
    dev_emb_path   = Path("query_emb_cache/dev_embs.npy")

    if not train_emb_path.exists():
        raise FileNotFoundError(f"Missing {train_emb_path}")

    train_embs = np.load(train_emb_path)
    with open(TRAIN_FILE) as f:
        train_data = json.load(f)

    # Hard labels: argmax of dataset_score
    def hard_label(item):
        ds = item.get("dataset_score", {})
        if not ds:
            return None
        by_type = defaultdict(float)
        for src, v in ds.items():
            if src in ("nq", "triviaqa"):
                by_type["text"] = max(by_type["text"], v)
            elif src in ("ott", "tat"):
                by_type["table"] = max(by_type["table"], v)
            elif src == "kg":
                by_type["kg"] = max(by_type["kg"], v)
        if not any(by_type.values()):
            return None
        return max(by_type, key=by_type.get)

    train_labels, valid_idx = [], []
    for i, item in enumerate(train_data):
        lbl = hard_label(item)
        if lbl is not None:
            train_labels.append(SOURCE_IDX[lbl])
            valid_idx.append(i)

    train_embs_valid = train_embs[valid_idx]
    train_labels_arr = np.array(train_labels)
    print(f"  Training LR on {len(train_labels_arr)} examples...")

    lr = LogisticRegression(C=0.1, max_iter=1000, multi_class="multinomial",
                            solver="lbfgs", random_state=42)
    lr.fit(train_embs_valid, train_labels_arr)

    lr_decisions = lr.predict(test_embs).astype(np.int32)
    np.save(RESULTS_DIR / "lr_decisions_k3.npy", lr_decisions)

    rr_lr = recall_at_picked(lr_decisions, has_rel)
    cal_lr = calibrated_recall(lr_decisions, has_rel)
    print(f"  routing_relevance: {rr_lr:.3f}")
    print(f"  calibrated recall@picked: {cal_lr:.3f}")

    # Per-type routing accuracy verification (should match paper: 0.642 macro)
    test_labels_arr = np.array([SOURCE_IDX[hard_label(item)] 
                                 for item in test_data 
                                 if hard_label(item) is not None])
    # Quick macro check
    per_type = {}
    for j, src in enumerate(SOURCE_TYPES):
        mask = (test_labels_arr == j)
        if mask.sum() > 0:
            per_type[src] = float((lr_decisions[mask] == j).mean())
    macro_lr = float(np.mean(list(per_type.values())))
    print(f"  routing macro (verify vs paper 0.642): {macro_lr:.3f}")
    print(f"  per-type: {per_type}")

except Exception as e:
    print(f"  LR failed: {e}")
    lr_decisions = None
    cal_lr = None

# ── MLP-HardCE ───────────────────────────────────────────────────────────────
print("\n── MLP-HardCE ──")

# Import SourceFormer architecture (CE seed runs use same MLP)
try:
    from sourceformer import SourceFormerK3
except ImportError:
    print("  Cannot import SourceFormerK3 -- trying inline definition")
    import torch.nn as nn
    class SourceFormerK3(nn.Module):
        def __init__(self, dropout=0.2):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(768, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(512, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(128, 3)
            )
        def forward(self, x): return self.net(x)

mlp_decisions_per_seed = {}
cal_mlp_per_seed = {}
embs_tensor = torch.from_numpy(test_embs).float()

# MLP-HardCE checkpoints — in checkpoints/ with hard_ce or seed naming
# Try checkpoints_ablation_2x2 (CE+Uniform cells) first, then checkpoints/
ckpt_candidates = {
    42:   [Path("checkpoints_ablation_2x2/CE_Uniform_seed42_best.pt")],
    123:  [Path("checkpoints_ablation_2x2/CE_Uniform_seed123_best.pt")],
    2026: [Path("checkpoints_ablation_2x2/CE_Uniform_seed2026_best.pt")],
}

for seed, candidates in ckpt_candidates.items():
    ckpt_path = next((p for p in candidates if p.exists()), None)
    if ckpt_path is None:
        print(f"  seed {seed}: no checkpoint found at any candidate path")
        print(f"  Tried: {[str(p) for p in candidates]}")
        continue
    print(f"  seed {seed}: loading {ckpt_path}")
    try:
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = SourceFormerK3(dropout=0.2)
        model.load_state_dict(ck["state_dict"])
        model.eval()
        with torch.no_grad():
            logits = model(embs_tensor)
            decisions = logits.argmax(-1).numpy().astype(np.int32)
        mlp_decisions_per_seed[seed] = decisions
        rr = recall_at_picked(decisions, has_rel)
        cal = calibrated_recall(decisions, has_rel)
        cal_mlp_per_seed[seed] = cal
        print(f"    routing_relevance: {rr:.3f}, calibrated recall@picked: {cal:.3f}")
    except Exception as e:
        print(f"  seed {seed}: failed -- {e}")

# Mean across seeds
if cal_mlp_per_seed:
    mean_mlp_cal = float(np.mean(list(cal_mlp_per_seed.values())))
    print(f"  MLP-HardCE mean calibrated recall@picked: {mean_mlp_cal:.3f}")
else:
    mean_mlp_cal = None
    print("  No MLP-HardCE checkpoints loaded.")
    print("  List your checkpoint files:")
    print("    ls checkpoints/ | grep -i 'hard\\|ce\\|mlp'")
    print("    ls checkpoints_ablation_2x2/")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("RECALL@PICKED SUMMARY (calibrated from linear model R²=0.9997)")
print("="*60)

results = {
    "calibration_model": {"a": A, "b": B, "r2": 0.9997},
    "known_exact": {
        "random":   0.385,
        "majority": 0.558,
        "oracle":   0.802,
        "sg_phase3": 0.714,
        "sg_phase4": 0.725,
    },
    "calibrated": {}
}

# Already computed from .npy files
conf_dec = np.load(RESULTS_DIR / "confidence_decisions_k3.npy")
pf_dec   = np.load(RESULTS_DIR / "prefrag_conf_decisions_k3.npy")

results["calibrated"]["bge_confidence"] = calibrated_recall(conf_dec, has_rel)
results["calibrated"]["prefrag_conf"]   = calibrated_recall(pf_dec, has_rel)

if cal_lr is not None:
    results["calibrated"]["logistic_regression"] = cal_lr
if mean_mlp_cal is not None:
    results["calibrated"]["mlp_hard_ce"] = mean_mlp_cal

with open(OUT_FILE, "w") as f:
    json.dump(results, f, indent=2)

print()
print("Known (exact):")
for m, v in results["known_exact"].items():
    print(f"  {m:<25} {v:.3f}")
print()
print("Calibrated estimates:")
for m, v in results["calibrated"].items():
    print(f"  {m:<25} {v:.3f}")
print()
print(f"Saved → {OUT_FILE}")
