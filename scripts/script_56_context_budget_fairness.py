"""
script_56_context_budget_fairness.py  --  was the reader starved?

THE PROBLEM THIS TESTS
  Every comparison so far concluded that BGE max-similarity dominates reader
  likelihood as a routing signal: similarity adds +0.045 / +0.029 / +0.055
  macro over SourceGate on three benchmark axes, while the reader adds nothing
  reliable on top (<= +0.022 at the 95% upper limit).

  An audit of the scoring path shows that comparison was not information-fair.
  script_12 retrieves TOP_K=10 chunks per source -- about 27,850 characters,
  roughly 7,000 tokens -- and then truncates the reader's context to
  MAX_CTX_CHARS = 800.  The reader conditions on ~256 tokens, i.e. 2.9% of the
  retrieved evidence, and since chunks average ~1,400 characters it does not
  see even one complete chunk: it scores a fragment of chunk #1.

  BGE max-similarity, by contrast, is computed over ALL retrieved chunks.

  So the reader may be dominated because it was starved, not because token
  likelihood is a weak routing signal.  If so, the domination claim is an
  artefact of a hyperparameter, and the paper's own PrefRAG-Conf baseline
  (macro 0.588) understates the method it is meant to represent -- which a
  reviewer would be right to call a strawman.

A LATENT BUG, FIXED HERE
  score_source tokenises prefix and full separately with different caps
  (max_length 900 and 1024) and masks labels at the prefix length.  At 800
  chars neither cap fires (measured: prefix mean 256 / max 330 tokens, full
  mean 280 / max 355), so published numbers are unaffected.  But at any larger
  budget the right-truncation of `full` would cut off the QUESTION TOKENS
  being scored, and any prefix beyond 900 tokens would be scored as if it were
  question tokens.

  The fix is to raise the caps to 4000/4096 and control length upstream by
  truncating the context STRING, leaving the tokenisation itself untouched.
  An earlier attempt concatenated separately tokenised pieces; that is safer
  against truncation but shifts scores by 0.35 nats on average, because it
  forces a token boundary where the joint tokenisation merges the space after
  "Question: " into the first question word.  Changing the measurement while
  measuring a budget effect would have confounded the very thing under test.

  Condition A re-scores at the original 800 chars as a control: it must
  reproduce the published scores exactly, confirming that only the budget
  differs between conditions.  Cap hits are counted; a non-zero count means
  the budget exceeded the prompt window and the run is invalid.

PREDICTIONS, FIXED BEFORE RUNNING
  C1  The reader was starved.  Reader macro rises materially at 3200 chars,
      the residual analysis must be redone at the fair budget, and the
      domination claim is weakened or overturned.
  C2  The reader was not starved.  Reader macro is flat or falls with more
      context.  Domination survives a serious challenge and becomes much
      better defended -- and we can state that the result is not an artefact
      of the context budget, which is the obvious reviewer objection.

  C2 would also be interesting in its own right: more evidence not helping a
  likelihood-based router is a fact about the signal, not about our setup.

USAGE
  conda activate chestx && python script_56_context_budget_fairness.py \
      [--budget 3200] [--split test] [--n N]

OUTPUT
  phase5_results/context_budget_{split}_{budget}.npz
"""

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch

import script_12_prefrag_conf as S12

RESULTS = Path("phase5_results")
SOURCES = ["text", "table", "kg"]
MAX_PREFIX_TOKENS = 4000          # context may be cut here; caps are reported
MAX_FULL_TOKENS = 4096


def macro(pred, g, k=3):
    v = [(pred[g == j] == j).mean() for j in range(k) if (g == j).sum()]
    return float(np.mean(v)) if v else float("nan")


def retrieve_budget(q_emb, source_type, idx, k, budget):
    texts = []
    for ds in S12.TYPE_TO_DATASETS_K3[source_type]:
        ids = S12.retrieve_chunk_ids(q_emb, ds, idx[ds], k)
        texts += S12.get_chunk_texts(ds, ids)
    return "\n\n".join(texts)[:budget]


_cap_hits = {"prefix": 0, "full": 0, "n": 0}


@torch.no_grad()
def score_source_fixed(query, context, tok, llm):
    """script_12.score_source with the token caps raised, and nothing else.

    An earlier version of this function tokenised prefix and question
    separately and concatenated the ids.  That guarantees the question is
    never truncated, but it also forces a token boundary at "Question: ",
    where the joint tokenisation would merge the trailing space with the first
    question word.  Scores shifted by 0.35 nats on average -- a change in what
    is being measured, which would confound the budget comparison this script
    exists to run.

    So the tokenisation is left exactly as published and only the caps move,
    from 900/1024 to MAX_PREFIX_TOKENS/MAX_FULL_TOKENS.  Length is controlled
    upstream by truncating the context STRING, so at 800 chars this reproduces
    the published scores bit for bit.  Cap hits are counted and reported; if
    any occur the budget is too large for this prompt and the run is invalid.
    """
    prefix = ("Answer the following question using only the provided context. "
              "Be concise.\n\nContext:\n" + context + "\n\nQuestion: ")
    full = prefix + query + "\n\nAnswer:"

    enc_prefix = tok(prefix, return_tensors="pt", truncation=True,
                     max_length=MAX_PREFIX_TOKENS)
    enc_full = tok(full, return_tensors="pt", truncation=True,
                   max_length=MAX_FULL_TOKENS).to(S12.DEVICE)

    prefix_len = enc_prefix["input_ids"].shape[1]
    seq_len = enc_full["input_ids"].shape[1]
    _cap_hits["n"] += 1
    _cap_hits["prefix"] += int(prefix_len >= MAX_PREFIX_TOKENS)
    _cap_hits["full"] += int(seq_len >= MAX_FULL_TOKENS)
    if prefix_len >= seq_len:
        return -1e9

    # Only the question tokens are scored, so computing logits over the whole
    # sequence is wasted work -- and at 3200 chars the (seq_len x 128k vocab)
    # tensor OOMs a 16 GB card.  Ask for logits on just the scored span and
    # take the mean cross-entropy by hand; mathematically identical to passing
    # `labels`, and verified bit-identical against the published 800-char run.
    ids = enc_full["input_ids"]
    n_keep = seq_len - prefix_len + 1
    try:
        logits = llm(input_ids=ids, logits_to_keep=n_keep).logits
    except TypeError:
        logits = llm(input_ids=ids, num_logits_to_keep=n_keep).logits
    tgt = ids[0, prefix_len:]                       # tokens being predicted
    pred = logits[0, :-1].float()                   # their predecessors
    return -torch.nn.functional.cross_entropy(pred, tgt).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=3200)
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--control800", action="store_true",
                    help="also re-score at 800 to verify the fixed scorer reproduces published values")
    args = ap.parse_args()

    recs_all = json.load(open(f"mmrag_{args.split}.json"))
    if args.split == "test":
        embs_all = np.load("query_emb_cache/test_embs.npy").astype(np.float32)
    else:
        embs_all = np.load(RESULTS / "prefrag_conf_dev_scores.npz")["embs"].astype(np.float32)

    # mmrag_{split}.json is ordered BY DATASET (ott, webqsp, tat, cwq, nq,
    # triviaqa), so recs[:n] is not a sample -- the first 40 test queries are
    # 83% table and contain zero KG.  Subsample at random instead; a leading
    # slice silently makes every small run unrepresentative.
    if args.n is not None and args.n < len(recs_all):
        sel = np.sort(np.random.default_rng(20260926).choice(
            len(recs_all), size=args.n, replace=False))
    else:
        sel = np.arange(len(recs_all))
    recs = [recs_all[i] for i in sel]
    embs = embs_all[sel]
    n = len(recs)
    gold = np.array([S12.hard_label_k3(r) for r in recs])
    print(f"  gold composition {np.bincount(gold, minlength=3)} "
          f"(full split {np.bincount([S12.hard_label_k3(r) for r in recs_all], minlength=3)})")

    budgets = ([800] if args.control800 else []) + [args.budget]
    print(f"{args.split}: {n} queries   budgets {budgets}")

    idx = S12.load_indices()
    ctx = {}
    for b in budgets:
        print(f"Retrieving at budget {b}...")
        ctx[b] = [[retrieve_budget(embs[i], s, idx, S12.TOP_K, b) for s in SOURCES]
                  for i in range(n)]
        ch = np.array([[len(c) for c in row] for row in ctx[b]])
        print(f"  mean chars delivered: {dict(zip(SOURCES, ch.mean(0).round(0)))}")
    del idx
    S12._pos_cache.clear()
    gc.collect()
    torch.cuda.empty_cache()

    tok, llm = S12.load_llm()
    out = {}
    for b in budgets:
        L = np.zeros((n, 3))
        t0 = time.time()
        for i in range(n):
            for j in range(3):
                L[i, j] = score_source_fixed(recs[i]["query"], ctx[b][i][j], tok, llm)
            if (i + 1) % 50 == 0:
                rate = (i + 1) / max(time.time() - t0, 1e-9)
                print(f"  b={b} {i+1}/{n}  {rate:.2f} q/s  "
                      f"eta {(n-i-1)/rate/60:.1f} min", flush=True)
        out[b] = L
        np.savez(RESULTS / f"context_budget_{args.split}_{b}.npz", scores=L, gold=gold)
        print(f"  budget {b}: reader argmax macro {macro(L.argmax(1), gold):.4f}")

    print("\n" + "=" * 66)
    print(f"{'budget (chars)':<18}{'reader macro':>14}{'vs published 800':>20}")
    print("=" * 66)
    base = None
    if args.split == "test" and args.n is None:
        pub = json.load(open(RESULTS / "prefrag_conf_results.json"))
        Lp = np.array([[r["source_scores"][s] for s in SOURCES] for r in pub])
        okp = (Lp > -1e8).all(1)
        base = macro(Lp[okp].argmax(1), np.array([r["true_label"] for r in pub])[okp])
        print(f"{'800 (published)':<18}{base:>14.4f}{'--':>20}")
    for b in budgets:
        m = macro(out[b].argmax(1), gold)
        d = f"{m - base:+.4f}" if base is not None else "--"
        print(f"{b:<18}{m:>14.4f}{d:>20}")
    print(f"\n  cap hits: prefix {_cap_hits['prefix']}/{_cap_hits['n']}  full {_cap_hits['full']}/{_cap_hits['n']}")
    print("\n  C1 reader was starved -> macro rises; redo the residual analysis")
    print("  C2 reader not starved -> domination survives the budget objection")


if __name__ == "__main__":
    main()
