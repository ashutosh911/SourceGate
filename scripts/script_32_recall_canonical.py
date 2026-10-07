"""
script_32_recall_canonical.py  —  recall@picked for every routing method,
                                  computed by script_7's own retriever

WHY THIS EXISTS
  Table 4's recall@picked column was assembled from two code paths that
  disagree on how many chunks a source type may return:

    script_7_evaluation.py :: MultiSourceRetriever.search_type()
        searches each constituent index, merges by score, keeps the global
        top-10. Exactly 10 chunks. Produced the published values for
        SourceGate, oracle, random and majority.

    script_24_exact_recall_faiss.py :: retrieve_chunk_ids()
        unions top-10 from EACH constituent index: up to 20 chunks for text
        (nq+triviaqa) and table (ott+tat), but 10 for kg (one index).
        Produced the published values for every other method.

  The second is a bug: it contradicts the paper's own definition of
  recall@picked ("top-10 chunks from the routed source") and the reader's
  actual 10-chunk context, and it inflates asymmetrically in favour of
  text/table-heavy (KG-averse) routers.

  script_31_recall_convention_audit.py established the split with a
  reimplementation. This script removes any dependence on that
  reimplementation by importing script_7's MultiSourceRetriever and calling
  search_type() directly -- the same object that produced the published
  SourceGate numbers. The hit rule is likewise script_7's:

      hit = bool(set(cids) & {cid : relevance > 0})

  Values emitted here are the canonical ones for Table 4.

USAGE
  conda activate chestx && python script_32_recall_canonical.py

OUTPUT
  phase5_results/recall_at_picked_canonical.json
"""

import json
import sys
import numpy as np
from pathlib import Path
import torch

# script_7 executes only cheap config at import time (paths, an LLMConfig
# dataclass); the retriever, BGE and the reader are all built inside main().
import script_7_evaluation as S7
from sourceformer import SourceFormerK3, SOURCE_TYPES

RESULTS_DIR = Path("phase5_results")
PHASE6_DIR = Path("phase6_results")
TOP_K = S7.TOP_K
SEEDS = [42, 123, 2026]
SOURCE_IDX = {s: i for i, s in enumerate(SOURCE_TYPES)}

# Published values, to see which reproduce and which move.
PUBLISHED = {
    "random": 0.385, "majority": 0.558, "oracle": 0.802,
    "bge_confidence": 0.778, "prefrag_conf": 0.300,
    "logistic_regression": 0.722, "mlp_hard_ce": 0.739,
    "r3ag_rq": 0.740, "r3ag_full": 0.354, "gpt4o_zeroshot": 0.646,
    "sg_superv": 0.714, "sg_joint": 0.725,
}
# Which code path produced each published value (from script_31's audit).
PROVENANCE = {
    "random": "strict", "majority": "strict", "oracle": "strict",
    "sg_superv": "strict", "sg_joint": "strict",
    "bge_confidence": "loose", "prefrag_conf": "loose",
    "logistic_regression": "loose", "mlp_hard_ce": "loose",
    "r3ag_rq": "loose", "r3ag_full": "loose", "gpt4o_zeroshot": "loose",
}

print("Loading test records via script_7.load_test_records ...")
records = S7.load_test_records(S7.TEST_FILE)
n = len(records)
relevant = [{c for c, s in r["relevant_chunks"].items() if s > 0} for r in records]
gold = np.array([r["oracle_label"] for r in records])

raw = json.load(open(S7.TEST_FILE))
if len(raw) != n:
    sys.exit(f"FATAL: script_7 dropped {len(raw)-n} queries; the cached "
             f"embeddings and saved decision arrays are indexed against the "
             f"full {len(raw)}-query file and would be misaligned.")

cache = Path("query_emb_cache/test_embs_k3.npy")
if not cache.exists():
    cache = Path("query_emb_cache/test_embs.npy")
embs = np.load(cache).astype(np.float32)
assert embs.shape[0] == n

print("Building script_7's MultiSourceRetriever ...")
retriever = S7.MultiSourceRetriever()

# Retrieval outcome is a property of (query, chosen source), not of the
# routing method.  Compute the n x K hit matrix once and reuse it for every
# decision array; the former implementation repeated identical FAISS searches
# for each method/seed.
print("Precomputing strict global-top-10 hit matrix ...")
hit_matrix = np.zeros((n, len(SOURCE_TYPES)), dtype=np.uint8)
for j, source_type in enumerate(SOURCE_TYPES):
    source_cids = retriever.search_type(embs, source_type, TOP_K)
    for i in range(n):
        hit_matrix[i, j] = bool(set(source_cids[i].tolist()) & relevant[i])
    print(f"  {source_type}: oracle-source hit rate={hit_matrix[:, j].mean():.4f}")


def recall_at_picked(dec):
    """script_7's own retrieval and hit rule, applied to a decision array."""
    dec = np.asarray(dec, dtype=int)
    return float(hit_matrix[np.arange(n), dec].mean())


# ── routing decisions, one array per seed ────────────────────────────────────
def sg_decisions(pattern):
    out = []
    for s in SEEDS:
        p = Path(pattern.format(seed=s))
        if not p.exists():
            return None
        ck = torch.load(p, map_location="cpu", weights_only=False)
        m = SourceFormerK3(dropout=0.2)
        m.load_state_dict(ck["state_dict"] if "state_dict" in ck else ck)
        m.eval()
        with torch.no_grad():
            out.append(m(torch.from_numpy(embs).float()).argmax(-1).numpy())
    return out


D = {}
D["oracle"] = [gold]
D["random"] = [np.random.RandomState(42).randint(0, 3, n)]
D["majority"] = [np.zeros(n, dtype=int)]
for nm, f in (("bge_confidence", "confidence_decisions_k3.npy"),
              ("prefrag_conf", "prefrag_conf_decisions_k3.npy")):
    p = RESULTS_DIR / f
    if p.exists():
        D[nm] = [np.load(p).astype(int)]
for nm, tag in (("r3ag_full", ""), ("r3ag_rq", "_rq")):
    ps = [PHASE6_DIR / f"r3ag_test_preds{tag}_seed{s}.npy" for s in SEEDS]
    if all(p.exists() for p in ps):
        D[nm] = [np.load(p).astype(int) for p in ps]
llm = RESULTS_DIR / "llm_classifier_k3_results.json"
if llm.exists():
    D["gpt4o_zeroshot"] = [np.array([SOURCE_IDX[r["pred"]]
                                     for r in json.load(open(llm))["records"]])]
for nm, pat in (("sg_superv", "checkpoints/sourceformer_k3_seed{seed}_best.pt"),
                ("sg_joint", "checkpoints_phase4/phase4_seed{seed}_best.pt"),
                ("mlp_hard_ce", "checkpoints_ablation_2x2/CE_Uniform_seed{seed}_best.pt")):
    d = sg_decisions(pat)
    if d:
        D[nm] = d
try:
    from sklearn.linear_model import LogisticRegression
    tr = json.load(open("mmrag_train.json"))
    tc = Path("query_emb_cache/train_embs_k3.npy")
    if not tc.exists():
        tc = Path("query_emb_cache/train_embs.npy")
    te = np.load(tc).astype(np.float32)
    lab, keep = [], []
    for i, it in enumerate(tr):
        s = it.get("dataset_score", {})
        by = [s.get("nq", 0) + s.get("triviaqa", 0),
              s.get("ott", 0) + s.get("tat", 0), s.get("kg", 0)]
        if max(by) > 0:
            lab.append(int(np.argmax(by)))
            keep.append(i)
    lr = LogisticRegression(C=0.1, max_iter=1000, solver="lbfgs", random_state=42)
    lr.fit(te[keep], np.array(lab))
    D["logistic_regression"] = [lr.predict(embs).astype(int)]
except Exception as e:
    print(f"  [warn] logistic regression skipped: {e}")

# ── evaluate ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 90)
print(f"{'method':<22}{'published':>10}{'canonical':>11}{'delta':>9}"
      f"{'  path that made published':>28}")
print("=" * 90)
out = {}
for nm, decs in D.items():
    vals = [recall_at_picked(d) for d in decs]
    mean = float(np.mean(vals))
    std = float(np.std(vals, ddof=1)) if len(vals) > 1 else None
    pub = PUBLISHED.get(nm)
    out[nm] = {"recall_at_picked": mean, "std": std, "per_seed": vals,
               "published": pub,
               "delta": (mean - pub) if pub is not None else None,
               "published_came_from": PROVENANCE.get(nm),
               "source": "script_7 MultiSourceRetriever.search_type"}
    d = f"{mean-pub:+.4f}" if pub is not None else "--"
    print(f"{nm:<22}{(f'{pub:.3f}' if pub else '--'):>10}{mean:>11.4f}{d:>9}"
          f"{PROVENANCE.get(nm,'?'):>28}")

json.dump(out, open(RESULTS_DIR / "recall_at_picked_canonical.json", "w"), indent=2)
print("\nSaved → phase5_results/recall_at_picked_canonical.json")
print("\nExpectation: rows whose published value came from the STRICT path "
      "reproduce (|delta| < 0.004); rows from the LOOSE path move down.")
