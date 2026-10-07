"""
script_62_nll_column.py

Fills Table 4's NLL column for the baseline methods, using the SAME function
that produced the published values.

WHY THIS SCRIPT EXISTS. The codebase contains two different functions that
both compute something called "NLL", and they are ~2.4 nats apart:

  * script_10_reader_scaling.compute_nll -- context truncated to 3000 chars,
    prompt to 1400 tokens, full to 1450. Returns ~6.15 even under ORACLE
    routing (script_61_nll_scale_check.py).
  * compute_l_ans_sequential (script_38 / script_6 validate_full) -- top-10
    chunks, max_seq_len 2048, answer-span masking. This is what produced
    Table 4: phase4_seed42_log.json's final loss_ans is 3.8082 against the
    table's 3.808 for SG (Joint).

Table 4's column is the second one, so this script uses it. Mixing the two
would put oracle and random below methods that route worse, which is what
first exposed the discrepancy.

Gold-answer NLL under the frozen reader, conditioned on the top-10 context of
whichever source each method routed to. Lower is better.

Usage:
  python script_62_nll_column.py                      # all decision arrays
  python script_62_nll_column.py --methods oracle random
"""
import argparse
import json
import os

import numpy as np

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import script_38_bridge_ablation as S38

SRC = ["text", "table", "kg"]
OUT = "phase5_results/nll_column.json"

DECISIONS = {
    "bge_confidence": "phase5_results/confidence_decisions_k3.npy",
    "logreg":         "phase5_results/lr_decisions_k3.npy",
    "majority":       "phase5_results/majority_decisions_k3.npy",
    "mlp_hardce":     "phase5_results/mlp_hardce_decisions_k3.npy",
    "prefrag_conf":   "phase5_results/prefrag_conf_decisions_k3.npy",
}

ap = argparse.ArgumentParser()
ap.add_argument("--methods", nargs="+", default=None)
args = ap.parse_args()

records = S38.load_records("mmrag_test.json")
n = len(records)
gold_lab = np.array([S38.hard_label_k3(r) if hasattr(S38, "hard_label_k3")
                     else r["hard_label"] for r in records])
print(f"{n} test records")

# oracle = gold labels; random = seeded uniform. Both are reference rows whose
# published NLL we can check this run against (Oracle 3.784, Random 4.713).
picks = {"oracle": gold_lab,
         "random": np.random.default_rng(42).integers(0, 3, n)}
for name, path in DECISIONS.items():
    if os.path.exists(path):
        a = np.load(path)
        if len(a) == n:
            picks[name] = a.astype(int)
        else:
            print(f"  [skip] {name}: {len(a)} decisions vs {n} records")
if args.methods:
    picks = {k: v for k, v in picks.items() if k in args.methods}
print("methods:", list(picks))

# ORDER MATTERS (handover trap #5): retrieve with the FAISS indices resident,
# release them, and only then load the reader. The first version of this script
# loaded the reader first and kept ~9 GiB of indices alive through scoring; it
# was killed by the OOM reaper at the retrieval->scoring transition.
bge, bge_tok = S38.load_bge_encoder()
retriever = S38.MultiSourceRetriever()
chunk_db = S38.ChunkDB("chunk_texts.db")
cfg = S38.llm_cfg

queries = [r["query"] for r in records]
answers = [r["answer"] for r in records]

# retrieve once per (query, source); reuse across methods
print("retrieving contexts for all sources...")
ctx_cache = {}
B = 64
for s0 in range(0, n, B):
    qs = queries[s0:s0 + B]
    embs = S38.encode_queries_batch(bge, bge_tok, qs).detach().cpu().numpy().astype(np.float32)
    got = retriever.search_all_types(embs, S38.TOP_K)
    for j, src in enumerate(SRC):
        for bi in range(len(qs)):
            cids = got[src][bi]
            ctx_cache[(s0 + bi, j)] = S38.format_chunks([chunk_db.get(c) for c in cids])
    if (s0 + B) % 512 == 0:
        print(f"  retrieved {min(s0+B, n)}/{n}", flush=True)

# Contexts are plain strings from here on; drop the indices and the encoder
# before the reader claims memory.
import gc
del retriever, bge, bge_tok
try:
    S38.S12._pos_cache.clear()
except Exception:
    pass
gc.collect()
torch.cuda.empty_cache()
print("released FAISS indices + BGE encoder", flush=True)
llm, llm_tok = S38.load_frozen_llm()

# Merge into any existing results rather than replacing them: this script is
# routinely run for a subset of methods (--methods), and a plain overwrite
# would discard rows computed in an earlier invocation.
res = {}
if os.path.exists(OUT):
    try:
        res = json.load(open(OUT))
        print(f"merging into existing {OUT}: {list(res)}")
    except Exception as e:
        print(f"  [warn] could not read {OUT} ({e}); starting fresh")
for name, dec in picks.items():
    print(f"\n=== {name} ===", flush=True)
    vals = []
    for i in range(n):
        ctx = ctx_cache[(i, int(dec[i]))]
        with torch.no_grad():
            v = S38.compute_l_ans_sequential(
                llm, llm_tok, [queries[i]], [ctx], [answers[i]], cfg, S38.DEVICE)
        vals.append(float(v.mean().item()))
        if (i + 1) % 250 == 0:
            print(f"  {i+1}/{n}  running mean NLL = {np.mean(vals):.4f}", flush=True)
    a = np.array(vals)
    a = a[np.isfinite(a)]
    dist = {s: int((dec == k).sum()) for k, s in enumerate(SRC)}
    res[name] = {"nll_mean": float(a.mean()), "nll_median": float(np.median(a)),
                 "n_scored": int(len(a)), "pick_dist": dist}
    print(f"  {name}: NLL = {a.mean():.4f}  (median {np.median(a):.4f}, n={len(a)})")
    json.dump(res, open(OUT, "w"), indent=2)

print("\n" + "=" * 62)
print(f"{'method':<18}{'NLL (this run)':>16}{'published':>12}")
PUB = {"random": 4.713, "oracle": 3.784}
for k, v in res.items():
    p = PUB.get(k)
    print(f"{k:<18}{v['nll_mean']:>16.4f}{(f'{p:.3f}' if p else '--'):>12}")
print("=" * 62)
print("If oracle/random reproduce their published values, the new rows are")
print("on Table 4's scale and can be inserted directly.")
print(f"wrote {OUT}")
