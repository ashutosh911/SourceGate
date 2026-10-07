"""
script_22_exact_recall_at_picked.py
Compute exact recall@picked for ALL routing methods using saved routing
decisions + mmrag_test.json chunk relevance annotations.

NO GPU needed. Runtime: ~2 minutes.

recall@picked(method) = fraction of test queries where the method's chosen
source contains at least one chunk with relevance_score > 0.

This is the RETRIEVAL-LEVEL metric: it measures whether the routed source's
corpus contains relevant evidence, independent of whether the top-10 FAISS
search actually surfaced that chunk. It is therefore an upper bound on the
paper's recall@picked (which additionally requires the chunk to appear in
top-10 retrieval), and we compare it to the paper values to derive a
retrieval-discount factor that we apply to methods where we only have
routing decisions.

For SourceGate Phase 3/4, the paper's exact recall@picked is already known
(0.714, 0.725) from phase5_metrics.json. We use those as ground truth.
For all other methods, we compute exact routing-relevance and apply the
calibrated discount.

OUTPUT:
  phase5_results/exact_recall_at_picked.json
"""

import json, numpy as np
from pathlib import Path
from collections import defaultdict

RESULTS_DIR  = Path("phase5_results")
TEST_FILE    = "mmrag_test.json"
SOURCE_TYPES = ["text", "table", "kg"]
SOURCE_IDX   = {s: i for i, s in enumerate(SOURCE_TYPES)}

# ── chunk -> source type mapping ─────────────────────────────────────────────
def source_of_chunk(cid):
    if cid.startswith(("nq_", "triviaqa_")):   return "text"
    if cid.startswith(("ott_", "tat_")):        return "table"
    return "kg"

# ── load test data ────────────────────────────────────────────────────────────
print("Loading mmrag_test.json...")
with open(TEST_FILE) as f:
    test_data = json.load(f)
n = len(test_data)
print(f"  {n} test queries")

# ── build per-(query, source) relevance matrix ───────────────────────────────
print("Building relevance matrix...")
has_rel = np.zeros((n, 3), dtype=bool)
for i, item in enumerate(test_data):
    for cid, score in item.get("relevant_chunks", {}).items():
        if score > 0:
            j = SOURCE_IDX.get(source_of_chunk(cid))
            if j is not None:
                has_rel[i, j] = True

print(f"  Source availability:")
for j, src in enumerate(SOURCE_TYPES):
    print(f"    {src}: {has_rel[:,j].mean():.3f} of queries have relevant chunks")

# ── helper: routing-relevance and paper-calibrated recall@picked ──────────────
def routing_relevance(decisions):
    """Fraction of queries where routed source has relevant chunks."""
    hits = has_rel[np.arange(n), decisions]
    return float(hits.mean())

# Calibration: paper recall@picked = retrieval_recall * routing_relevance
# Estimated from Phase3 (0.714 exact) and oracle (0.802 exact)
oracle_decisions = np.array([
    SOURCE_IDX[
        max({"text": max(item["dataset_score"].get("nq",0),
                         item["dataset_score"].get("triviaqa",0)),
             "table": max(item["dataset_score"].get("ott",0),
                          item["dataset_score"].get("tat",0)),
             "kg": item["dataset_score"].get("kg",0)},
            key=lambda k: {"text": max(item["dataset_score"].get("nq",0),
                                       item["dataset_score"].get("triviaqa",0)),
                           "table": max(item["dataset_score"].get("ott",0),
                                        item["dataset_score"].get("tat",0)),
                           "kg": item["dataset_score"].get("kg",0)}[k])
    ] for item in test_data
], dtype=int)

rr_oracle  = routing_relevance(oracle_decisions)
rr_random  = float(has_rel[np.arange(n), np.random.RandomState(42).randint(0,3,n)].mean())
rr_majority = float(has_rel[:, 0].mean())  # majority = text

print(f"\n  Routing-relevance verification:")
print(f"    oracle:  {rr_oracle:.4f}  (paper recall@picked: 0.802)")
print(f"    random:  {rr_random:.4f}  (paper recall@picked: 0.385)")
print(f"    majority:{rr_majority:.4f}  (paper recall@picked: 0.558)")

# Fit linear calibration: paper_r@p = a * routing_relevance + b
# Using four exact points: oracle=0.802, phase3=0.714, majority=0.558, random=0.385
rr_pts     = np.array([rr_oracle, rr_random, rr_majority])
paper_pts  = np.array([0.802,     0.385,     0.558])
# We'll add phase3 once we load its decisions
# For now compute coefficients without phase3, verify then add it

# ── load routing decisions ────────────────────────────────────────────────────
print("\nLoading routing decisions...")

decisions = {}

# BGE-confidence
try:
    decisions["bge_confidence"] = np.load(RESULTS_DIR/"confidence_decisions_k3.npy")
    print(f"  bge_confidence: loaded {len(decisions['bge_confidence'])} decisions")
except FileNotFoundError:
    print("  bge_confidence: NOT FOUND")

# PrefRAG-Conf
try:
    decisions["prefrag_conf"] = np.load(RESULTS_DIR/"prefrag_conf_decisions_k3.npy")
    print(f"  prefrag_conf: loaded {len(decisions['prefrag_conf'])} decisions")
except FileNotFoundError:
    print("  prefrag_conf: NOT FOUND")

# Logistic Regression -- refit from training data
print("  Fitting Logistic Regression on train data...")
try:
    from sklearn.linear_model import LogisticRegression
    import torch, torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModel

    TRAIN_FILE = "mmrag_train.json"
    with open(TRAIN_FILE) as f:
        train_data = json.load(f)

    # Hard labels from dataset_score
    def hard_label(item):
        ds = item.get("dataset_score", {})
        by_type = {
            "text":  max(ds.get("nq",0), ds.get("triviaqa",0)),
            "table": max(ds.get("ott",0), ds.get("tat",0)),
            "kg":    ds.get("kg", 0),
        }
        best = max(by_type, key=by_type.get)
        return SOURCE_IDX[best] if by_type[best] > 0 else None

    # Use cached embeddings if available
    train_cache = Path("query_emb_cache/train_embs_k3.npy")
    test_cache  = Path("query_emb_cache/test_embs_k3.npy")

    # Also try without _k3 suffix
    if not train_cache.exists():
        train_cache = Path("query_emb_cache/train_embs.npy")
    if not test_cache.exists():
        test_cache = Path("query_emb_cache/test_embs.npy")

    if train_cache.exists() and test_cache.exists():
        train_embs = np.load(train_cache)
        test_embs  = np.load(test_cache)

        train_labels, valid_idx = [], []
        for i, item in enumerate(train_data):
            lbl = hard_label(item)
            if lbl is not None:
                train_labels.append(lbl)
                valid_idx.append(i)

        lr = LogisticRegression(C=0.1, max_iter=1000, solver="lbfgs",
                                 multi_class="multinomial", random_state=42)
        lr.fit(train_embs[valid_idx], np.array(train_labels))
        decisions["logistic_regression"] = lr.predict(test_embs).astype(int)
        print(f"  logistic_regression: fitted and predicted")
    else:
        print(f"  logistic_regression: embedding cache not found at {train_cache}")
        print(f"    Tried: query_emb_cache/train_embs_k3.npy and train_embs.npy")

except Exception as e:
    print(f"  logistic_regression: FAILED -- {e}")

# MLP-HardCE from checkpoints_ablation_2x2
print("  Loading MLP-HardCE (CE_Uniform checkpoints)...")
try:
    import torch
    from sourceformer import SourceFormerK3

    test_cache = Path("query_emb_cache/test_embs_k3.npy")
    if not test_cache.exists():
        test_cache = Path("query_emb_cache/test_embs.npy")
    test_embs_t = torch.from_numpy(np.load(test_cache)).float()

    mlp_all = []
    for seed in [42, 123, 2026]:
        ckpt = Path(f"checkpoints_ablation_2x2/CE_Uniform_seed{seed}_best.pt")
        if not ckpt.exists():
            print(f"    seed {seed}: {ckpt} not found")
            continue
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        model = SourceFormerK3(dropout=0.2)
        model.load_state_dict(ck["state_dict"])
        model.eval()
        with torch.no_grad():
            preds = model(test_embs_t).argmax(-1).numpy().astype(int)
        mlp_all.append(preds)
        print(f"    seed {seed}: loaded")

    if mlp_all:
        # Use mean routing-relevance across seeds (consistent with paper's n=3 treatment)
        decisions["mlp_hard_ce"] = mlp_all  # list of arrays, one per seed
        print(f"  mlp_hard_ce: {len(mlp_all)} seeds loaded")
    else:
        print("  mlp_hard_ce: no checkpoints found")

except Exception as e:
    print(f"  mlp_hard_ce: FAILED -- {e}")

# Phase 3 predictions (for verification)
print("  Loading Phase 3 predictions for calibration verification...")
try:
    phase3_preds = [json.loads(l) for l in open(RESULTS_DIR/"predictions_phase3_seed42.jsonl")]
    decisions["sg_phase3"] = np.array([
        SOURCE_IDX[p["picked_type"]] for p in phase3_preds], dtype=int)
    print(f"  sg_phase3: loaded {len(decisions['sg_phase3'])} decisions")
except Exception as e:
    print(f"  sg_phase3: {e}")

# ── compute exact routing-relevance for all methods ───────────────────────────
print("\n" + "="*60)
print("EXACT ROUTING-RELEVANCE (routing -> source has relevant chunks)")
print("="*60)

results = {}

# Methods with single decision array
single_methods = ["bge_confidence", "prefrag_conf", "logistic_regression", "sg_phase3"]
for m in single_methods:
    if m not in decisions:
        print(f"  {m:<25}: MISSING")
        continue
    rr = routing_relevance(decisions[m])
    results[m] = {"routing_relevance": rr}

# Baselines with known exact values
results["random"]   = {"routing_relevance": rr_random,   "paper_r_at_p_exact": 0.385}
results["majority"] = {"routing_relevance": rr_majority,  "paper_r_at_p_exact": 0.558}
results["oracle"]   = {"routing_relevance": rr_oracle,    "paper_r_at_p_exact": 0.802}

# MLP-HardCE: mean across seeds
if "mlp_hard_ce" in decisions and isinstance(decisions["mlp_hard_ce"], list):
    rrs = [routing_relevance(d) for d in decisions["mlp_hard_ce"]]
    results["mlp_hard_ce"] = {
        "routing_relevance": float(np.mean(rrs)),
        "routing_relevance_std": float(np.std(rrs, ddof=1)),
        "per_seed": rrs,
    }

# ── calibration: fit linear model using methods with exact paper values ───────
print("\n" + "="*60)
print("CALIBRATION MODEL: paper recall@picked = a * routing_relevance + b")
print("="*60)

# Use oracle, majority, random + sg_phase3 as calibration points
cal_rr  = [rr_oracle, rr_majority, rr_random]
cal_rp  = [0.802,     0.558,       0.385]

if "sg_phase3" in results:
    cal_rr.append(results["sg_phase3"]["routing_relevance"])
    cal_rp.append(0.714)  # known exact from phase5_metrics.json

cal_rr = np.array(cal_rr)
cal_rp = np.array(cal_rp)
a, b = np.polyfit(cal_rr, cal_rp, 1)
residuals = cal_rp - (a * cal_rr + b)
r2 = 1 - np.var(residuals) / np.var(cal_rp)

print(f"  Calibration: recall@picked ≈ {a:.4f} × routing_relevance + {b:.4f}")
print(f"  R² = {r2:.4f} across {len(cal_rr)} calibration points")
print()

# Apply calibration to methods without exact paper recall@picked
for m, v in results.items():
    rr = v["routing_relevance"]
    cal = float(a * rr + b)
    v["calibrated_recall_at_picked"] = cal
    if "paper_r_at_p_exact" in v:
        v["calibrated_recall_at_picked"] = v["paper_r_at_p_exact"]  # use exact

# Special: sg_phase3 exact = 0.714 (from phase5_metrics.json)
if "sg_phase3" in results:
    results["sg_phase3"]["paper_r_at_p_exact"] = 0.714
    results["sg_phase3"]["calibrated_recall_at_picked"] = 0.714

# ── print full table ──────────────────────────────────────────────────────────
print("="*70)
print("RECALL@PICKED — COMPLETE TABLE")
print("="*70)
print(f"  {'Method':<25} {'Routing-Rel':>12} {'R@P exact/cal':>14} {'Type':>10}")
print("  " + "-"*65)

method_order = [
    ("random",              "Random",             "exact"),
    ("majority",            "Majority",           "exact"),
    ("bge_confidence",      "BGE-confidence",     "calibrated"),
    ("prefrag_conf",        "PrefRAG-Conf",       "calibrated"),
    ("logistic_regression", "Logistic Regression","calibrated"),
    ("mlp_hard_ce",         "MLP-HardCE",         "calibrated"),
    ("sg_phase3",           "SG Phase 3",         "exact"),
    ("oracle",              "Oracle",             "exact"),
]

for key, label, typ in method_order:
    if key not in results:
        print(f"  {label:<25} {'N/A':>12} {'N/A':>14} {typ:>10}")
        continue
    v = results[key]
    rr = v["routing_relevance"]
    rp = v.get("paper_r_at_p_exact") or v.get("calibrated_recall_at_picked")
    flag = "" if typ == "exact" else " †"
    print(f"  {label:<25} {rr:>12.4f} {rp:>13.4f}{flag} {typ:>10}")

print()
print(f"  † calibrated from linear model (R²={r2:.4f})")
print()
print("KEY FINDINGS:")
print(f"  BGE-confidence routing-relevance: {results.get('bge_confidence',{}).get('routing_relevance','N/A')}")
print(f"  SG Phase 3 routing-relevance:     {results.get('sg_phase3',{}).get('routing_relevance','N/A')}")
print()
bge_rr = results.get("bge_confidence",{}).get("routing_relevance")
sg3_rr = results.get("sg_phase3",{}).get("routing_relevance")
if bge_rr and sg3_rr:
    print(f"  BGE-conf vs SG Phase3 routing-relevance gap: {sg3_rr-bge_rr:+.4f}")
    print(f"  BGE-conf calibrated R@P:  {a*bge_rr+b:.4f}")
    print(f"  SG Phase3 exact R@P:      0.714")
    print(f"  Difference:               {0.714-(a*bge_rr+b):+.4f}")

# ── save ─────────────────────────────────────────────────────────────────────
out = RESULTS_DIR / "exact_recall_at_picked.json"
# Make JSON-serializable
save = {}
for k, v in results.items():
    save[k] = {kk: (float(vv) if isinstance(vv, (np.floating, float)) else vv)
                for kk, vv in v.items() if kk != "per_seed"}
    if "per_seed" in v:
        save[k]["per_seed"] = [float(x) for x in v["per_seed"]]

with open(out, "w") as f:
    json.dump(save, f, indent=2)
print(f"\nSaved → {out}")
print("Upload this file to update Table 4 with exact/calibrated values.")
