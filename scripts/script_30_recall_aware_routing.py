"""
script_30_recall_aware_routing.py  —  Pareto analysis + recall-aware routing

PURPOSE
  Two questions about the macro-accuracy / recall@picked trade-off.

  (1) DOMINANCE. Four baselines exceed SourceGate on recall@picked. Does any
      of them exceed it on BOTH recall@picked and routing macro accuracy? A
      method is dominated only if another is >= on both and > on one.

  (2) OPERATING POINT. SourceGate routes by argmax of the router posterior.
      recall@picked, however, rewards routing away from sources whose
      retrieval is weak, independently of whether that is the correct source.
      We therefore sweep a one-parameter family of decision rules

          z(q) = argmax_k  p_k(q) * r_k^lambda

      where p_k is the router posterior and r_k is source k's retrieval
      reliability. lambda=0 recovers the paper's argmax rule; larger lambda
      trades label agreement for retrieval hit-rate. The question is whether
      SourceGate can be moved along the frontier to match or beat the
      KG-averse baselines, and at what macro cost.

PROTOCOL (no test-set selection)
  r_k        estimated on the TRAIN split only
  lambda     selected on the DEV split only, by a criterion fixed before
             looking at any test number (see LAMBDA_CRITERION)
  reporting  the selected lambda is then evaluated once on TEST

  Nothing on the test set informs r_k or lambda. The full test curve is
  computed only so the trade-off can be drawn; the claimed operating point is
  the dev-selected one.

  recall@picked and the hit matrices use the CANONICAL retrieval definition:
  a source type's constituent indices are searched, merged by score, and the
  global top-10 kept (10 chunks total, not 10 per index). This matches
  script_7_evaluation.py::search_type, which produced the published
  SourceGate/oracle numbers, and it is what script_32_recall_canonical.py
  established as authoritative for Table 4.

  An earlier version of this script used the per-index union of
  script_24_exact_recall_faiss.py, which returns up to 20 chunks for text and
  table and inflates recall@picked asymmetrically in favour of KG-averse
  routing. Every number produced before that fix is superseded.

USAGE
  conda activate chestx && python script_30_recall_aware_routing.py

OUTPUT
  phase5_results/recall_aware_routing.json
"""

import json
import sys
import numpy as np
from pathlib import Path

import faiss
import torch

RESULTS_DIR = Path("phase5_results")
FAISS_DIR = Path("faiss_indices")
SOURCE_TYPES = ["text", "table", "kg"]
SOURCE_TO_DATASETS = {"text": ["nq", "triviaqa"], "table": ["ott", "tat"], "kg": ["kg"]}
TOP_K = 10
SEEDS = [42, 123, 2026]
LAMBDAS = np.round(np.arange(0.0, 6.01, 0.25), 4)

# Fixed before inspecting any test number: among rules whose DEV macro stays
# at or above the strongest learned baseline's macro (MLP-HardCE, 0.702),
# take the one with the highest DEV recall@picked.
MACRO_FLOOR = 0.702
LAMBDA_CRITERION = (f"max dev recall@picked subject to dev macro >= "
                    f"{MACRO_FLOOR} (MLP-HardCE macro)")

# Published test-set operating points (Table 4).
# (routing macro, recall@picked) -- recall values are the CANONICAL ones from
# script_32_recall_canonical.py, not the superseded per-index-union figures.
BASELINES = {
    "BGE-confidence":     (0.661, 0.755),
    "R3AG-RQ":            (0.663, 0.709),
    "MLP-HardCE":         (0.702, 0.709),
    "LogisticRegression": (0.642, 0.682),
    "GPT-4o-mini":        (0.542, 0.624),
    "R3AG-full":          (0.348, 0.347),
    "PrefRAG-Conf":       (0.334, 0.299),
    "Random":             (0.337, 0.385),
    "Majority":           (0.333, 0.558),
    "SG superv":          (0.737, 0.714),
    "SG joint":           (0.717, 0.725),
}


def load_split(name):
    data = json.load(open(f"mmrag_{name}.json"))
    for cand in (f"query_emb_cache/{name}_embs_k3.npy",
                 f"query_emb_cache/{name}_embs.npy"):
        if Path(cand).exists():
            embs = np.load(cand).astype(np.float32)
            break
    else:
        sys.exit(f"FATAL: no cached embeddings for {name}")
    if embs.shape[0] != len(data):
        sys.exit(f"FATAL: {name} embeddings {embs.shape[0]} != {len(data)} queries")
    return data, embs


def gold_sum(item):
    """Canonical routing label (see script_28): argmax of summed dataset_score."""
    s = item["dataset_score"]
    return int(np.argmax([s.get("nq", 0) + s.get("triviaqa", 0),
                          s.get("ott", 0) + s.get("tat", 0),
                          s.get("kg", 0)]))


def macro_accuracy(dec, gold):
    return float(np.mean([float((dec[gold == k] == k).mean())
                          for k in range(3) if (gold == k).sum()]))


print("Loading FAISS indices...")
indices, chunk_ids = {}, {}
for src, dss in SOURCE_TO_DATASETS.items():
    indices[src], chunk_ids[src] = {}, {}
    for ds in dss:
        indices[src][ds] = faiss.read_index(str(FAISS_DIR / ds / "index.faiss"))
        chunk_ids[src][ds] = list(np.load(FAISS_DIR / ds / "chunk_ids.npy",
                                          allow_pickle=True))
    print(f"  {src}: {', '.join(dss)}")


def hit_matrix(data, embs, name):
    """H[i,k] = 1 if the canonical top-10 from source k contains a relevant chunk.

    Canonical = merge the source type's constituent indices by score and keep
    the global top-10 (script_7_evaluation.py::search_type).
    """
    cache = RESULTS_DIR / f"hit_matrix_{name}_strict.npy"
    if cache.exists():
        H = np.load(cache)
        if H.shape == (len(data), 3):
            print(f"  [{name}] hit-matrix cache hit {H.shape}")
            return H
    rel = [{c for c, s in it.get("relevant_chunks", {}).items() if s > 0}
           for it in data]
    H = np.zeros((len(data), 3), dtype=np.int8)
    for i in range(len(data)):
        q = embs[i].reshape(1, -1)
        for k, src in enumerate(SOURCE_TYPES):
            # merge the constituent indices by score, keep the global top-10
            pool = []
            for ds, idx in indices[src].items():
                sc, pos = idx.search(q, min(TOP_K, idx.ntotal))
                ids = chunk_ids[src][ds]
                pool += [(float(sc[0][j]), ids[p])
                         for j, p in enumerate(pos[0]) if 0 <= p < len(ids)]
            pool.sort(key=lambda t: -t[0])
            got = {c for _, c in pool[:TOP_K]}
            H[i, k] = 1 if (got & rel[i]) else 0
        if (i + 1) % 250 == 0:
            print(f"  [{name}] {i+1}/{len(data)}")
    np.save(cache, H)
    return H


print("\nBuilding hit matrices (this is the expensive part)...")
splits = {}
for name in ("train", "dev", "test"):
    data, embs = load_split(name)
    H = hit_matrix(data, embs, name)
    gold = np.array([gold_sum(it) for it in data])
    splits[name] = dict(data=data, embs=embs, H=H, gold=gold)
    print(f"  [{name}] n={len(data)}  per-source hit rate="
          f"{np.round(H.mean(0), 4).tolist()}")

# ── r_k from TRAIN only ───────────────────────────────────────────────────────
r = splits["train"]["H"].mean(0).astype(np.float64)
print(f"\nRetrieval reliability r_k (train split only): "
      f"{dict(zip(SOURCE_TYPES, np.round(r, 4)))}")

# ── router posteriors ─────────────────────────────────────────────────────────
from sourceformer import SourceFormerK3

def posteriors(embs, seed):
    ck = torch.load(f"checkpoints/sourceformer_k3_seed{seed}_best.pt",
                    map_location="cpu", weights_only=False)
    m = SourceFormerK3(dropout=0.2)
    m.load_state_dict(ck["state_dict"] if "state_dict" in ck else ck)
    m.eval()
    with torch.no_grad():
        return torch.softmax(m(torch.from_numpy(embs).float()), -1).numpy()


def curve(split, seed):
    p = posteriors(splits[split]["embs"], seed)
    H, gold = splits[split]["H"], splits[split]["gold"]
    out = []
    for lam in LAMBDAS:
        dec = np.argmax(p * (r ** lam), axis=1)
        out.append(dict(lam=float(lam),
                        macro=macro_accuracy(dec, gold),
                        recall=float(H[np.arange(len(dec)), dec].mean()),
                        kg_rate=float((dec == 2).mean())))
    return out


print("\nSweeping lambda on DEV (selection) and TEST (curve only)...")
dev_curves = {s: curve("dev", s) for s in SEEDS}
test_curves = {s: curve("test", s) for s in SEEDS}


def mean_curve(cs):
    return [dict(lam=cs[SEEDS[0]][i]["lam"],
                 macro=float(np.mean([cs[s][i]["macro"] for s in SEEDS])),
                 recall=float(np.mean([cs[s][i]["recall"] for s in SEEDS])),
                 recall_std=float(np.std([cs[s][i]["recall"] for s in SEEDS], ddof=1)),
                 macro_std=float(np.std([cs[s][i]["macro"] for s in SEEDS], ddof=1)),
                 kg_rate=float(np.mean([cs[s][i]["kg_rate"] for s in SEEDS])))
            for i in range(len(LAMBDAS))]


dev_mean, test_mean = mean_curve(dev_curves), mean_curve(test_curves)

eligible = [d for d in dev_mean if d["macro"] >= MACRO_FLOOR]
if not eligible:
    print(f"\nNo lambda keeps dev macro >= {MACRO_FLOOR}; selection fails.")
    chosen = None
else:
    chosen = max(eligible, key=lambda d: d["recall"])
    print(f"\nSelected on dev: lambda={chosen['lam']}  "
          f"dev macro={chosen['macro']:.4f}  dev recall={chosen['recall']:.4f}")

print("\nDEV curve (selection split):")
for d in dev_mean:
    mark = " <-- selected" if chosen and d["lam"] == chosen["lam"] else ""
    print(f"  lam={d['lam']:<5} macro={d['macro']:.4f}  recall={d['recall']:.4f}"
          f"  kg_rate={d['kg_rate']:.3f}{mark}")

result = {"protocol": {"r_k_split": "train", "lambda_split": "dev",
                       "criterion": LAMBDA_CRITERION, "macro_floor": MACRO_FLOOR},
          "r_k": dict(zip(SOURCE_TYPES, r.tolist())),
          "dev_curve": dev_mean, "test_curve": test_mean,
          "selected": chosen}

if chosen is not None:
    tp = next(d for d in test_mean if d["lam"] == chosen["lam"])
    base = next(d for d in test_mean if d["lam"] == 0.0)
    result["test_at_selected"] = tp
    result["test_at_lambda0"] = base
    print("\n" + "=" * 68)
    print("TEST at the dev-selected lambda (single evaluation)")
    print("=" * 68)
    print(f"  lambda=0   (paper rule): macro {base['macro']:.4f}  "
          f"recall {base['recall']:.4f}")
    print(f"  lambda={chosen['lam']:<4}            : macro {tp['macro']:.4f}  "
          f"recall {tp['recall']:.4f}")
    print(f"  change                 : macro {tp['macro']-base['macro']:+.4f}  "
          f"recall {tp['recall']-base['recall']:+.4f}")

    print("\n  Dominance check against published baselines:")
    dominated = []
    for nm, (bm, br) in BASELINES.items():
        if nm.startswith("SG"):
            continue
        if tp["macro"] >= bm and tp["recall"] >= br and (tp["macro"] > bm or tp["recall"] > br):
            dominated.append(nm)
            print(f"    DOMINATES {nm} (macro {bm}, recall {br})")
    result["dominates"] = dominated
    if not dominated:
        print("    dominates nothing it did not already dominate")

# ── dominance among published points ──────────────────────────────────────────
print("\n" + "=" * 68)
print("PARETO FRONTIER among published Table 4 operating points")
print("=" * 68)
front = {}
for nm, (m, rr) in BASELINES.items():
    dom = [o for o, (m2, r2) in BASELINES.items()
           if o != nm and m2 >= m and r2 >= rr and (m2 > m or r2 > rr)]
    front[nm] = {"macro": m, "recall": rr, "dominated_by": dom,
                 "on_frontier": not dom}
for nm in sorted(BASELINES, key=lambda x: -BASELINES[x][0]):
    f = front[nm]
    tag = "ON FRONTIER" if f["on_frontier"] else f"dominated by {f['dominated_by'][0]}"
    print(f"  {nm:<20} macro {f['macro']:.3f}  recall {f['recall']:.3f}   {tag}")
result["pareto"] = front

out = RESULTS_DIR / "recall_aware_routing.json"
json.dump(result, open(out, "w"), indent=2)
print(f"\nSaved → {out}")
