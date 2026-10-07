"""
script_61_nll_scale_check.py

Is Table 4's NLL column on the same scale as script_10's current compute_nll?

Table 4 reports NLL for four rows only: Random 4.713, Oracle 3.784,
SG-Superv 3.782, SG-Joint 3.808. Re-running script_10 over five other
methods produced 6.50-6.71 for every one of them -- including methods whose
routing is BETTER than random, which is incoherent against Random's 4.713.
Either the new numbers are on a different scale from the published column, or
the published column is.

Oracle routing is the clean probe: its decisions are just the gold labels, so
it needs no checkpoint and no trained router. If current compute_nll returns
~3.78 on oracle picks, the code is consistent with the published column and
PrefRAG-Conf's 6.504 is a real (very poor) value. If it returns ~6.5, the
published column came from different code and the two must not be mixed.

Random subsample -- mmrag_test.json is ordered BY DATASET, so test[:n] is
never a sample (trap #1). Realised class composition is printed.

Usage: python script_61_nll_scale_check.py [--n 200]
"""
import argparse
import json
import os

import numpy as np

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import script_10_reader_scaling as S10

SRC = ["text", "table", "kg"]

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=200)
args = ap.parse_args()

records = S10.load_test_records(S10.TEST_FILE)
n_all = len(records)
rng = np.random.default_rng(20260930)
sel = np.sort(rng.choice(n_all, size=min(args.n, n_all), replace=False))
print(f"{n_all} test records; random subsample n={len(sel)}")

oracle = np.array([r["oracle_label"] for r in records])
comp = {s: int((oracle[sel] == i).sum()) for i, s in enumerate(SRC)}
print(f"realised composition of the subsample: {comp}")

bge, bge_tok = S10.load_bge()
query_embs = S10.encode_queries(bge, bge_tok, [records[i]["query"] for i in sel])
retriever = S10.MultiSourceRetriever()
chunk_db = S10.ChunkDB("chunk_texts.db")
reader = S10.LocalLlamaReader()

nlls = []
for k, i in enumerate(sel):
    r = records[i]
    picked = SRC[int(oracle[i])]                       # ORACLE routing
    cids = retriever.search_type(query_embs[k:k+1], picked, S10.TOP_K)[0]
    chunks_dict = chunk_db.get_many(list(cids))
    context = S10.format_chunks([chunks_dict[c] for c in cids])
    gold = r["answer"][0] if isinstance(r["answer"], list) else r["answer"]
    v = reader.compute_nll(r["query"], context, gold)
    if np.isfinite(v):
        nlls.append(v)
    if (k + 1) % 50 == 0:
        print(f"  {k+1}/{len(sel)}  running mean NLL = {np.mean(nlls):.4f}", flush=True)

a = np.array(nlls)
print("\n" + "=" * 62)
print(f"oracle-routing NLL, current compute_nll : {a.mean():.4f}  "
      f"(median {np.median(a):.4f}, n={len(a)})")
print(f"Table 4's published Oracle NLL          : 3.784")
print(f"script_10 re-run, prefrag_conf          : 6.504")
print("=" * 62)
d = abs(a.mean() - 3.784)
if d < 0.35:
    print("VERDICT: current code REPRODUCES the published scale.")
    print("         -> the 6.5 values are genuine; the column can be filled.")
elif abs(a.mean() - 6.5) < 0.8:
    print("VERDICT: current code is on the ~6.5 scale, NOT the published one.")
    print("         -> published NLL column came from different code; DO NOT MIX.")
else:
    print(f"VERDICT: neither scale (mean {a.mean():.4f}); needs manual inspection.")

json.dump({"n": int(len(a)), "oracle_nll_current_code": float(a.mean()),
           "oracle_nll_published": 3.784, "prefrag_conf_rerun": 6.504,
           "subsample_composition": comp},
          open("phase5_results/nll_scale_check.json", "w"), indent=2)
print("wrote phase5_results/nll_scale_check.json")
