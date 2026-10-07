"""
script_37_scrambled_context_control.py  —  does generation confidence respond
                                           to evidence relevance, or to source
                                           identity?

MOTIVATION
  A variance decomposition of the routing signal (mean-token NLL over question
  tokens, conditioned on each candidate source's retrieved context) shows that
  across four regimes -- mmRAG question-span, mmRAG gold-answer-span, and
  OTT-QA at two equalised context budgets -- 75-86% of its variance is a
  query-intrinsic difficulty term that is identical across candidate sources,
  and only 14-22% is the query x source interaction that could carry
  source-specific evidence.

  That bounds how much relevance information the signal *could* carry. It does
  not establish that the interaction term carries any.

THE CONTROL
  For each query i and source s we score query i's question tokens against the
  context that source s returned for a DIFFERENT query j. Format, chunk style
  and context length are preserved exactly (all sources are capped at
  MAX_CTX_CHARS, so the token ratio stays ~1.13x). Only relevance is destroyed.

  j is drawn from a seeded derangement (no query keeps its own context), so
  every scrambled context is a real retrieved context from the right source --
  just for the wrong question.

PREDICTIONS, FIXED BEFORE RUNNING
  H1  The interaction term carries genuine relevance.
      -> scrambling destroys it. The interaction share of variance collapses
         toward 0, picks are driven by the constant per-source offset alone,
         and macro falls toward the degenerate 1/K = 0.333.
  H2  The signal is responding to source identity/format, not relevance.
      -> scrambling changes little. Interaction share stays ~20%, and macro
         stays near the real-context value of 0.588.

  Either outcome is reportable. H2 is the stronger diagnostic claim; H1 means
  the interaction is real but too weak to route on, which is the weaker claim
  the decomposition already supports.

USAGE
  conda activate chestx && python script_37_scrambled_context_control.py [--n N]

OUTPUT
  phase5_results/scrambled_context_control.json
"""

import argparse
import json
import time

import numpy as np

import script_12_prefrag_conf as S12

SOURCES = ["text", "table", "kg"]
SEED = 20260925


def decomp(nll):
    """Two-way variance decomposition: query / source / interaction."""
    n, k = nll.shape
    grand = nll.mean()
    qm = nll.mean(1, keepdims=True)
    sm = nll.mean(0, keepdims=True)
    tot = ((nll - grand) ** 2).sum()
    q = k * ((qm - grand) ** 2).sum()
    s = n * ((sm - grand) ** 2).sum()
    return q / tot, s / tot, (tot - q - s) / tot


def macro(pred, gold, k=3):
    return float(np.mean([(pred[gold == j] == j).mean()
                          for j in range(k) if (gold == j).sum()]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None, help="limit queries (smoke test)")
    args = ap.parse_args()

    test = json.load(open(S12.TEST_FILE))
    embs = np.load(S12.EMB_CACHE).astype(np.float32)
    n = len(test) if args.n is None else min(args.n, len(test))
    gold = np.array([S12.hard_label_k3(t) for t in test[:n]])

    # seeded derangement: nobody keeps their own context
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n)
    fixed = np.flatnonzero(perm == np.arange(n))
    for i in fixed:                       # swap any fixed point with its neighbour
        j = (i + 1) % n
        perm[i], perm[j] = perm[j], perm[i]
    assert not (perm == np.arange(n)).any(), "derangement has fixed points"
    print(f"{n} queries; derangement seed {SEED}, {len(fixed)} fixed points repaired")

    # Retrieve every context BEFORE the reader is loaded, then release the FAISS
    # indices (~9 GB resident) and the chunk cache. The reader is loaded fp16
    # with CPU offload, so its offloaded layers need system RAM; holding the
    # indices at the same time exhausts it and the per-forward host->device
    # transfer OOMs.
    idx = S12.load_indices()
    print("Retrieving contexts...")
    ctx = [[S12.retrieve_context(embs[i], s, idx, S12.TOP_K) for s in SOURCES]
           for i in range(n)]
    lens = np.array([[len(c) for c in row] for row in ctx])
    print(f"  mean context chars per source: "
          f"{dict(zip(SOURCES, lens.mean(0).round(1)))}")

    del idx
    S12._pos_cache.clear()
    import gc, torch
    gc.collect()
    torch.cuda.empty_cache()
    print("  released FAISS indices and chunk cache before loading the reader")

    llm_tok, llm = S12.load_llm()

    print("\nScoring scrambled contexts (question of i, context of perm[i])...")
    scores = np.zeros((n, 3))
    t0 = time.time()
    for i in range(n):
        q = test[i]["query"]
        for j, s in enumerate(SOURCES):
            scores[i, j] = S12.score_source(q, ctx[perm[i]][j], llm_tok, llm)
        if (i + 1) % 100 == 0:
            el = time.time() - t0
            print(f"  {i+1}/{n}  {(i+1)/max(el,1):.2f} q/s  eta {(n-i-1)/max((i+1)/max(el,1),1e-9)/60:.1f} min")

    ok = (scores > -1e8).all(1)
    nll = -scores[ok]
    g = gold[ok]
    pred = nll.argmin(1)

    q, s, inter = decomp(nll)
    out = {
        "n_scored": int(ok.sum()),
        "seed": SEED,
        "scrambled": {
            "variance_query": float(q), "variance_source": float(s),
            "variance_interaction": float(inter),
            "macro": macro(pred, g), "acc": float((pred == g).mean()),
            "pick_rate": {t: float((pred == j).mean()) for j, t in enumerate(SOURCES)},
            "mean_nll": {t: float(nll[:, j].mean()) for j, t in enumerate(SOURCES)},
        },
        "real_context_reference": {
            "variance_query": 0.751, "variance_source": 0.032,
            "variance_interaction": 0.217,
            "macro": 0.5883, "pick_rate": {"text": 0.502, "table": 0.199, "kg": 0.298},
        },
        "baselines": {"random_macro": 0.337, "bge_confidence": 0.661,
                      "supervised": 0.737, "base_rate_kg": 0.163},
    }
    json.dump(out, open("phase5_results/scrambled_context_control.json", "w"), indent=2)

    r = out["real_context_reference"]
    print("\n" + "=" * 72)
    print(f"{'':<26}{'real ctx':>12}{'scrambled':>12}{'change':>12}")
    print("=" * 72)
    print(f"{'variance: query':<26}{r['variance_query']:>12.1%}{q:>12.1%}{q-r['variance_query']:>+12.1%}")
    print(f"{'variance: source offset':<26}{r['variance_source']:>12.1%}{s:>12.1%}{s-r['variance_source']:>+12.1%}")
    print(f"{'variance: interaction':<26}{r['variance_interaction']:>12.1%}{inter:>12.1%}{inter-r['variance_interaction']:>+12.1%}")
    print(f"{'routing macro':<26}{r['macro']:>12.4f}{macro(pred,g):>12.4f}{macro(pred,g)-r['macro']:>+12.4f}")
    for j, t in enumerate(SOURCES):
        print(f"{'  pick rate '+t:<26}{r['pick_rate'][t]:>12.1%}{(pred==j).mean():>12.1%}"
              f"{(pred==j).mean()-r['pick_rate'][t]:>+12.1%}")
    print(f"\n  random {0.337:.3f} | BGE-conf {0.661:.3f} | supervised {0.737:.3f}")
    print("\n  H1 (interaction is relevance): macro -> ~0.333, interaction -> ~0")
    print("  H2 (signal tracks source identity): macro ~0.588, interaction ~22%")


if __name__ == "__main__":
    main()
