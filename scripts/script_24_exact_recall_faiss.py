"""
script_24_exact_recall_faiss.py  —  Exact recall@picked for all routing methods

PURPOSE
  Compute recall@picked for BGE-confidence, PrefRAG-Conf, Logistic Regression,
  and MLP-HardCE by running actual FAISS retrieval for each method's chosen
  source per query and checking whether the top-10 chunks contain a relevant one.

  recall@picked(method, query i) = 1 if any of the top-10 chunks from
  method's chosen source for query i appears in item["relevant_chunks"]
  with score > 0.

  This is identical to how phase5_metrics.json computes recall@picked for
  SourceGate Phase3/4 (0.714, 0.725) and oracle (0.802).

NO GPU needed. Pure FAISS + numpy. Runtime: ~10 minutes.

USAGE
  python script_24_exact_recall_faiss.py

OUTPUT
  phase5_results/exact_recall_at_picked_faiss.json
"""

# ==========================================================================
# SUPERSEDED -- DO NOT USE FOR recall@picked
#
# retrieve_chunk_ids() below unions the top-10 of EACH constituent index,
# so a source type spanning two indices (text = nq+triviaqa, table =
# ott+tat) returns up to 20 chunks while kg returns 10. That contradicts
# the paper's definition of recall@picked ("top-10 chunks from the routed
# source") and the reader's actual 10-chunk context, and it inflates the
# metric asymmetrically in favour of KG-averse routers (+0.02 to +0.04 for
# text/table-heavy methods, ~0 for KG-collapsed ones).
#
# The canonical values come from script_32_recall_canonical.py, which calls
# script_7_evaluation.MultiSourceRetriever.search_type() -- merge the
# constituent indices by score, keep the global top-10. See
# script_31_recall_convention_audit.py for the side-by-side evidence.
# ==========================================================================


import json, numpy as np
from pathlib import Path
from collections import defaultdict
import faiss
import torch
import torch.nn.functional as F

RESULTS_DIR  = Path("phase5_results")
TEST_FILE    = "mmrag_test.json"
SOURCE_TYPES = ["text", "table", "kg"]
SOURCE_IDX   = {s: i for i, s in enumerate(SOURCE_TYPES)}
TOP_K        = 10

# Source -> dataset mapping (for FAISS index selection)
SOURCE_TO_DATASETS = {
    "text":  ["nq", "triviaqa"],
    "table": ["ott", "tat"],
    "kg":    ["kg"],
}

CHUNK_ID_PREFIX = {
    "nq": "nq_", "triviaqa": "triviaqa_",
    "ott": "ott_", "tat": "tat_",
    "kg": None,  # KG uses Freebase MIDs
}

def source_of_chunk(cid):
    if cid.startswith(("nq_", "triviaqa_")): return "text"
    if cid.startswith(("ott_", "tat_")):     return "table"
    return "kg"

# ── load test data ────────────────────────────────────────────────────────────
print("Loading test data...")
with open(TEST_FILE) as f:
    test_data = json.load(f)
n = len(test_data)
print(f"  {n} test queries")

# Build per-query relevant chunk sets
relevant_chunks = []
for item in test_data:
    rel = {cid for cid, score in item.get("relevant_chunks", {}).items() if score > 0}
    relevant_chunks.append(rel)

# ── load FAISS indices ────────────────────────────────────────────────────────
print("Loading FAISS indices...")
faiss_indices = {}
chunk_id_lists = {}  # dataset -> list of chunk IDs in FAISS order

FAISS_DIR = Path("faiss_indices")


for src, datasets in SOURCE_TO_DATASETS.items():
    faiss_indices[src] = {}
    chunk_id_lists[src] = {}
    for ds in datasets:
        idx_path = FAISS_DIR / ds / "index.faiss"
        if not idx_path.exists():
            print(f"  WARNING: {idx_path} not found -- skipping {ds}")
            continue
        index = faiss.read_index(str(idx_path))
        faiss_indices[src][ds] = index
        # chunk_ids.npy stores chunk IDs in FAISS positional order
        id_file = FAISS_DIR / ds / "chunk_ids.npy"
        if not id_file.exists():
            print(f"  WARNING: {id_file} not found -- skipping {ds}")
            continue
        ids = list(np.load(id_file, allow_pickle=True))
        print(f"  {ds}: {index.ntotal} vectors, {len(ids)} chunk IDs")
        chunk_id_lists[src][ds] = ids
        print(f"  {ds}: {index.ntotal} vectors, {len(ids)} chunk IDs")


# ── load query embeddings ─────────────────────────────────────────────────────
print("Loading query embeddings...")
test_cache = Path("query_emb_cache/test_embs_k3.npy")
if not test_cache.exists():
    test_cache = Path("query_emb_cache/test_embs.npy")
test_embs = np.load(test_cache).astype(np.float32)
print(f"  shape: {test_embs.shape}")

# ── retrieve top-k chunk IDs for a source ────────────────────────────────────
def retrieve_chunk_ids(query_emb, source):
    """Return set of top-K chunk IDs from the given source."""
    retrieved = set()
    q = query_emb.reshape(1, -1)
    for ds, index in faiss_indices[source].items():
        if index.ntotal == 0:
            continue
        k = min(TOP_K, index.ntotal)
        _, positions = index.search(q, k)
        ids = chunk_id_lists[source][ds]
        for pos in positions[0]:
            if 0 <= pos < len(ids):
                retrieved.add(ids[pos])
    return retrieved

# ── compute recall@picked for a routing decision array ───────────────────────
def compute_recall(decisions, name="method"):
    """decisions: (n,) int array of source indices."""
    hits = 0
    per_type = defaultdict(lambda: [0, 0])
    for i in range(n):
        src = SOURCE_TYPES[decisions[i]]
        retrieved = retrieve_chunk_ids(test_embs[i], src)
        hit = bool(retrieved & relevant_chunks[i])
        if hit:
            hits += 1
        true_src = max(
            {"text": max(test_data[i]["dataset_score"].get("nq",0),
                         test_data[i]["dataset_score"].get("triviaqa",0)),
             "table": max(test_data[i]["dataset_score"].get("ott",0),
                          test_data[i]["dataset_score"].get("tat",0)),
             "kg": test_data[i]["dataset_score"].get("kg",0)},
            key=lambda k: {"text": max(test_data[i]["dataset_score"].get("nq",0),
                                       test_data[i]["dataset_score"].get("triviaqa",0)),
                           "table": max(test_data[i]["dataset_score"].get("ott",0),
                                        test_data[i]["dataset_score"].get("tat",0)),
                           "kg": test_data[i]["dataset_score"].get("kg",0)}[k])
        per_type[true_src][1] += 1
        if src == true_src and hit:
            per_type[true_src][0] += 1

        if (i + 1) % 100 == 0:
            print(f"  [{name}] {i+1}/{n}  running recall={hits/(i+1):.4f}")

    recall = hits / n
    print(f"  [{name}] recall@picked = {recall:.4f}")
    return recall

# ── load routing decisions for all methods ────────────────────────────────────
print("\nLoading routing decisions...")

results = {}

# Phase 3 and Phase 4 -- already exact in phase5_metrics.json
p5 = json.load(open(RESULTS_DIR / "phase5_metrics.json"))
results["sg_phase3"] = {"recall_at_picked": p5["phase3"]["recall_at_picked"], "source": "exact (phase5_metrics)"}
results["sg_phase4"] = {"recall_at_picked": p5["phase4"]["recall_at_picked"], "source": "exact (phase5_metrics)"}
results["oracle"]    = {"recall_at_picked": p5["oracle"]["recall_at_picked"],  "source": "exact (phase5_metrics)"}
results["random"]    = {"recall_at_picked": p5["random"]["recall_at_picked"],  "source": "exact (phase5_metrics)"}
results["majority"]  = {"recall_at_picked": p5["majority"]["recall_at_picked"],"source": "exact (phase5_metrics)"}

print(f"  Loaded exact values for phase3/4, oracle, random, majority")

# BGE-confidence
try:
    conf_dec = np.load(RESULTS_DIR / "confidence_decisions_k3.npy")
    print(f"  Computing exact recall@picked for BGE-confidence...")
    r_conf = compute_recall(conf_dec, "BGE-conf")
    results["bge_confidence"] = {"recall_at_picked": r_conf, "source": "exact (FAISS)"}
except FileNotFoundError:
    print("  confidence_decisions_k3.npy not found")

# PrefRAG-Conf
try:
    pf_dec = np.load(RESULTS_DIR / "prefrag_conf_decisions_k3.npy")
    print(f"  Computing exact recall@picked for PrefRAG-Conf...")
    r_pf = compute_recall(pf_dec, "PrefRAG")
    results["prefrag_conf"] = {"recall_at_picked": r_pf, "source": "exact (FAISS)"}
except FileNotFoundError:
    print("  prefrag_conf_decisions_k3.npy not found")

# Logistic Regression -- refit from embeddings
try:
    from sklearn.linear_model import LogisticRegression

    with open("mmrag_train.json") as f:
        train_data = json.load(f)
    train_cache = Path("query_emb_cache/train_embs_k3.npy")
    if not train_cache.exists():
        train_cache = Path("query_emb_cache/train_embs.npy")
    train_embs = np.load(train_cache).astype(np.float32)

    def hard_label(item):
        ds = item.get("dataset_score", {})
        by = {"text": max(ds.get("nq",0), ds.get("triviaqa",0)),
              "table": max(ds.get("ott",0), ds.get("tat",0)),
              "kg": ds.get("kg",0)}
        best = max(by, key=by.get)
        return SOURCE_IDX[best] if by[best] > 0 else None

    train_labels, valid_idx = [], []
    for i, item in enumerate(train_data):
        lbl = hard_label(item)
        if lbl is not None:
            train_labels.append(lbl)
            valid_idx.append(i)

    lr = LogisticRegression(C=0.1, max_iter=1000, solver="lbfgs",
                             multi_class="multinomial", random_state=42)
    lr.fit(train_embs[valid_idx], np.array(train_labels))
    lr_dec = lr.predict(test_embs).astype(int)

    print("  Computing exact recall@picked for Logistic Regression...")
    r_lr = compute_recall(lr_dec, "LR")
    results["logistic_regression"] = {"recall_at_picked": r_lr, "source": "exact (FAISS)"}
except Exception as e:
    print(f"  LR failed: {e}")

# MLP-HardCE
try:
    from sourceformer import SourceFormerK3
    test_t = torch.from_numpy(test_embs).float()
    mlp_recalls = []
    for seed in [42, 123, 2026]:
        ckpt = Path(f"checkpoints_ablation_2x2/CE_Uniform_seed{seed}_best.pt")
        if not ckpt.exists():
            continue
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        model = SourceFormerK3(dropout=0.2)
        model.load_state_dict(ck["state_dict"])
        model.eval()
        with torch.no_grad():
            dec = model(test_t).argmax(-1).numpy().astype(int)
        print(f"  Computing exact recall@picked for MLP-HardCE seed {seed}...")
        r = compute_recall(dec, f"MLP-seed{seed}")
        mlp_recalls.append(r)
    if mlp_recalls:
        results["mlp_hard_ce"] = {
            "recall_at_picked": float(np.mean(mlp_recalls)),
            "recall_at_picked_std": float(np.std(mlp_recalls, ddof=1) if len(mlp_recalls)>1 else 0),
            "per_seed": mlp_recalls,
            "source": "exact (FAISS)"
        }
except Exception as e:
    print(f"  MLP failed: {e}")

# ── print and save ────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("EXACT RECALL@PICKED — COMPLETE TABLE")
print("="*60)
order = ["random","majority","bge_confidence","prefrag_conf",
         "logistic_regression","mlp_hard_ce","sg_phase3","sg_phase4","oracle"]
for m in order:
    if m not in results:
        continue
    v = results[m]
    src = v.get("source","")
    r = v["recall_at_picked"]
    print(f"  {m:<25} {r:.4f}  [{src}]")

out = RESULTS_DIR / "exact_recall_at_picked_faiss.json"
with open(out, "w") as f:
    json.dump({k: {kk: float(vv) if isinstance(vv, (np.floating,float)) else vv
                   for kk,vv in v.items()} for k,v in results.items()},
              f, indent=2)
print(f"\nSaved → {out}")
print("Upload this file to update Table 4 with exact values.")
