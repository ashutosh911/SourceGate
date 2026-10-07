"""
script_31_recall_convention_audit.py  —  Is recall@picked computed the same
                                          way for every row of Table 4?

WHY
  Two code paths in this repository compute recall@picked, and they do not
  agree on how many chunks a source type is allowed to return.

    STRICT  script_7_evaluation.py :: search_type()
            searches each constituent index, merges by score, keeps the
            global TOP-10 for the source type.  -> 10 chunks, always.
            Produced phase5_metrics.json, i.e. the published values for
            SourceGate (0.714 / 0.725), oracle (0.802), random, majority.

    LOOSE   script_24_exact_recall_faiss.py :: retrieve_chunk_ids()
            searches each constituent index and unions top-10 from EACH.
            -> up to 20 chunks for text (nq+triviaqa) and table (ott+tat),
               but only 10 for kg (single index).
            Produced the published values for BGE-confidence (0.778),
            PrefRAG-Conf (0.300), logistic regression (0.722), MLP-HardCE
            (0.739), and (via script_28) R3AG-RQ, R3AG-full, GPT-4o-mini.

  If that is right, Table 4 compares SourceGate under a 10-chunk budget
  against baselines under a budget of up to 20 -- and the inflation is
  asymmetric, favouring exactly the text/table-heavy (KG-averse) routers.

WHAT THIS DOES
  Recomputes recall@picked for every routing method under BOTH conventions
  and checks which one reproduces each published number. It changes nothing;
  it only reports.

USAGE
  conda activate chestx && python script_31_recall_convention_audit.py

OUTPUT
  phase5_results/recall_convention_audit.json
"""

import json
import sys
import numpy as np
from pathlib import Path

import faiss
import torch

RESULTS_DIR = Path("phase5_results")
PHASE6_DIR = Path("phase6_results")
FAISS_DIR = Path("faiss_indices")
SOURCE_TYPES = ["text", "table", "kg"]
SOURCE_IDX = {s: i for i, s in enumerate(SOURCE_TYPES)}
SOURCE_TO_DATASETS = {"text": ["nq", "triviaqa"], "table": ["ott", "tat"], "kg": ["kg"]}
TOP_K = 10
SEEDS = [42, 123, 2026]

PUBLISHED = {  # Table 4 recall@picked as it currently stands
    "random": 0.385, "majority": 0.558, "oracle": 0.802,
    "bge_confidence": 0.778, "prefrag_conf": 0.300,
    "logistic_regression": 0.722, "mlp_hard_ce": 0.739,
    "r3ag_rq": 0.740, "r3ag_full": 0.354, "gpt4o_zeroshot": 0.646,
    "sg_superv": 0.714, "sg_joint": 0.725,
}

test_data = json.load(open("mmrag_test.json"))
n = len(test_data)
rel = [{c for c, s in it.get("relevant_chunks", {}).items() if s > 0}
       for it in test_data]
cache = Path("query_emb_cache/test_embs_k3.npy")
if not cache.exists():
    cache = Path("query_emb_cache/test_embs.npy")
embs = np.load(cache).astype(np.float32)

print("Loading FAISS indices...")
idx, cids = {}, {}
for src, dss in SOURCE_TO_DATASETS.items():
    idx[src], cids[src] = {}, {}
    for ds in dss:
        idx[src][ds] = faiss.read_index(str(FAISS_DIR / ds / "index.faiss"))
        cids[src][ds] = list(np.load(FAISS_DIR / ds / "chunk_ids.npy",
                                     allow_pickle=True))


def hits(i, k, strict):
    """Does source type k return a relevant chunk for query i?"""
    q = embs[i].reshape(1, -1)
    src = SOURCE_TYPES[k]
    if strict:
        # merge across constituent indices, keep global top-10 by score
        pool = []
        for ds, ix in idx[src].items():
            sc, pos = ix.search(q, min(TOP_K, ix.ntotal))
            ids = cids[src][ds]
            pool += [(float(sc[0][j]), ids[p])
                     for j, p in enumerate(pos[0]) if 0 <= p < len(ids)]
        pool.sort(key=lambda t: -t[0])
        got = {c for _, c in pool[:TOP_K]}
    else:
        got = set()
        for ds, ix in idx[src].items():
            _, pos = ix.search(q, min(TOP_K, ix.ntotal))
            ids = cids[src][ds]
            got.update(ids[p] for p in pos[0] if 0 <= p < len(ids))
    return bool(got & rel[i])


print("Building hit matrices under both conventions...")
H = {}
for strict in (True, False):
    key = "strict" if strict else "loose"
    c = RESULTS_DIR / f"hit_matrix_test_{key}.npy"
    if c.exists():
        H[key] = np.load(c)
        print(f"  [{key}] cache hit")
        continue
    M = np.zeros((n, 3), dtype=np.int8)
    for i in range(n):
        for k in range(3):
            M[i, k] = hits(i, k, strict)
        if (i + 1) % 250 == 0:
            print(f"  [{key}] {i+1}/{n}")
    np.save(c, M)
    H[key] = M
for key in ("strict", "loose"):
    print(f"  [{key}] per-source hit rate = {np.round(H[key].mean(0), 4).tolist()}")


def gold_sum(it):
    s = it["dataset_score"]
    return int(np.argmax([s.get("nq", 0) + s.get("triviaqa", 0),
                          s.get("ott", 0) + s.get("tat", 0), s.get("kg", 0)]))


gold = np.array([gold_sum(it) for it in test_data])

# ── every method's per-query routing decision ────────────────────────────────
from sourceformer import SourceFormerK3

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
rng = np.random.RandomState(42)
D["random"] = [rng.randint(0, 3, n)]
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
d = sg_decisions("checkpoints/sourceformer_k3_seed{seed}_best.pt")
if d:
    D["sg_superv"] = d
d = sg_decisions("checkpoints_phase4/phase4_seed{seed}_best.pt")
if d:
    D["sg_joint"] = d
d = sg_decisions("checkpoints_ablation_2x2/CE_Uniform_seed{seed}_best.pt")
if d:
    D["mlp_hard_ce"] = d

try:  # logistic regression, refit exactly as script_24 does
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

# ── recompute under both conventions ─────────────────────────────────────────
print("\n" + "=" * 86)
print(f"{'method':<22}{'published':>10}{'STRICT':>10}{'LOOSE':>10}"
      f"{'  reproduces':>14}{'  strict-pub':>13}")
print("=" * 86)
rows = {}
for nm, decs in D.items():
    r = {}
    for key in ("strict", "loose"):
        v = [float(H[key][np.arange(n), dd].mean()) for dd in decs]
        r[key] = float(np.mean(v))
    pub = PUBLISHED.get(nm)
    which = ""
    if pub is not None:
        ds, dl = abs(r["strict"] - pub), abs(r["loose"] - pub)
        which = "STRICT" if ds < dl and ds < 0.004 else (
                "LOOSE" if dl < ds and dl < 0.004 else "neither")
    rows[nm] = {"published": pub, **r, "reproduces": which,
                "delta_strict_minus_published": (r["strict"] - pub) if pub else None}
    print(f"{nm:<22}{(f'{pub:.3f}' if pub else '--'):>10}{r['strict']:>10.4f}"
          f"{r['loose']:>10.4f}{which:>14}"
          f"{(r['strict']-pub) if pub else 0:>+13.4f}")

json.dump(rows, open(RESULTS_DIR / "recall_convention_audit.json", "w"), indent=2)
print("\nSaved → phase5_results/recall_convention_audit.json")
