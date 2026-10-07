"""
script_40_contrastive_evidence_gain.py  --  does a query-conditional
                                            counterfactual cancel nuisance in
                                            the generation-confidence signal?

WHAT THIS TESTS, AND WHAT IT DOES NOT
  The source-level version of the contrastive score is already settled without
  GPU time.  Scrambling relevance while holding source and length fixed leaves
  a per-source NLL spread of 0.086 nats (0.22% of variance), so there is almost
  no format prior to subtract; and an ORACLE per-source affine recalibration of
  the raw score, fitted on the test set itself, reaches only macro 0.652 --
  below BGE-confidence (0.661) and well below SourceGate (0.737).  Any method
  whose correction is constant within a query is capped there.

  What is not settled is the QUERY-CONDITIONAL counterfactual

      G(i,s) = log p(q_i | c_{i,s}) - E_j[ log p(q_i | c_{j,s}) ]

  where the negative uses query i's own tokens against a context that source s
  returned for a different query j.  That term varies within a row, so it can
  re-rank.  It should remove whatever part of the query x source interaction is
  spurious compatibility rather than evidence.

  The real-context interaction is 21.7% of variance and the scrambled one 5.6%,
  but those are shares of different totals.  This script records both raw score
  matrices so the comparison can finally be made in nats.

NOISE, AND WHY M > 1
  A single donor estimates E_j with one draw, injecting noise of roughly the
  same scale as the nuisance being removed.  Averaging M donors scales that
  noise as 1/M while leaving the nuisance estimate unbiased.  Results are
  reported at M = 1, 2, 3 so the trend is visible rather than assumed; if the
  gain does not grow with M it is not a real cancellation.

PREDICTIONS, FIXED BEFORE RUNNING
  Q1  The interaction contains a substantial query-conditional nuisance term.
      -> macro rises with M and exceeds the 0.652 source-calibration oracle.
  Q2  The interaction is essentially all evidence (or all noise).
      -> macro at M=3 stays within a couple of points of 0.588, or falls.

  Q2 is the outcome the existing evidence points to.  Q1 is the only route by
  which the mechanism story survives.

USAGE
  conda activate chestx && python script_40_contrastive_evidence_gain.py --n 400
  (--n omitted runs the full 1286-query test set)

OUTPUT
  phase5_results/contrastive_evidence_gain.json   summary + all macros
  phase5_results/ceg_scores.npz                   raw real (N,3) and neg (N,3,M)
"""

import argparse
import gc
import json
import time

import numpy as np

import script_12_prefrag_conf as S12

SOURCES = ["text", "table", "kg"]
SEED = 20260925
M_DONORS = 3


def donor_permutations(n, m, rng):
    """m derangements that never reuse a donor for the same query.

    Independent random derangements collide, so instead lay the queries on a
    random cycle and step m different distances along it.  Every row is a
    permutation, no query donates to itself, and the m donors of any query are
    distinct by construction for m < n.
    """
    assert m < n, "need more queries than donors"
    sigma = rng.permutation(n)          # position -> query id
    pos = np.empty(n, dtype=int)
    pos[sigma] = np.arange(n)           # query id -> position
    perms = np.stack([sigma[(pos + d) % n] for d in range(1, m + 1)])
    assert not (perms == np.arange(n)).any()
    return perms


def decomp(nll):
    """Two-way variance decomposition, returned as shares AND absolute sums of
    squares, because shares of different totals are not comparable."""
    n, k = nll.shape
    g = nll.mean()
    qm = nll.mean(1, keepdims=True)
    sm = nll.mean(0, keepdims=True)
    tot = ((nll - g) ** 2).sum()
    q = k * ((qm - g) ** 2).sum()
    s = n * ((sm - g) ** 2).sum()
    inter = tot - q - s
    return {"share_query": q / tot, "share_source": s / tot,
            "share_interaction": inter / tot,
            "ss_total": tot, "ss_query": q, "ss_source": s,
            "ss_interaction": inter,
            "ms_interaction": inter / max((n - 1) * (k - 1), 1)}


def macro(pred, gold, k=3):
    return float(np.mean([(pred[gold == j] == j).mean()
                          for j in range(k) if (gold == j).sum()]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--donors", type=int, default=M_DONORS)
    args = ap.parse_args()
    M = args.donors

    test_all = json.load(open(S12.TEST_FILE))
    embs_all = np.load(S12.EMB_CACHE).astype(np.float32)

    # mmrag_test.json is ordered BY DATASET (ott, webqsp, tat, cwq, nq,
    # triviaqa).  The original --n 400 run took test[:400], which is 51% table
    # and 32% text against the split's true 28% / 56% -- so that pilot was run
    # on an unrepresentative sample.  Subsample at random instead.
    if args.n is not None and args.n < len(test_all):
        sel = np.sort(np.random.default_rng(SEED).choice(
            len(test_all), size=args.n, replace=False))
    else:
        sel = np.arange(len(test_all))
    test = [test_all[i] for i in sel]
    embs = embs_all[sel]
    n = len(test)
    gold = np.array([S12.hard_label_k3(t) for t in test])
    print(f"{n} queries, {M} donors  -> {n * 3 * (M + 1)} forward passes")
    print(f"  gold counts {dict(zip(SOURCES, [int((gold == j).sum()) for j in range(3)]))}")

    rng = np.random.default_rng(SEED)
    perms = donor_permutations(n, M, rng)
    assert all(len(set(perms[:, i])) == M for i in range(n))

    # Retrieve every context before the reader is loaded, then release the FAISS
    # indices (~9 GB resident): the reader is fp16 with CPU offload and its
    # offloaded layers need the same host RAM.
    idx = S12.load_indices()
    print("Retrieving contexts...")
    ctx = [[S12.retrieve_context(embs[i], s, idx, S12.TOP_K) for s in SOURCES]
           for i in range(n)]
    lens = np.array([[len(c) for c in row] for row in ctx])
    print(f"  mean context chars: {dict(zip(SOURCES, lens.mean(0).round(1)))}")

    del idx
    S12._pos_cache.clear()
    import torch
    gc.collect()
    torch.cuda.empty_cache()

    llm_tok, llm = S12.load_llm()

    real = np.zeros((n, 3))
    neg = np.zeros((n, 3, M))
    # Crash tolerance: the fp16 CPU-offloaded reader sits at ~14 GiB of a
    # 15.45 GiB card, so any other process on the GPU can OOM this run.  A
    # d10 attempt died at query 200/1286 and lost everything.  Checkpoint
    # every 50 queries to a donor-count-specific file and resume from it.
    ckpt_path = S12.RESULTS_DIR / f"ceg_partial_d{M}.npz"
    start_i = 0
    if ckpt_path.exists():
        cp = np.load(ckpt_path)
        if (cp["real"].shape == real.shape and cp["neg"].shape == neg.shape
                and int(cp["seed"]) == SEED and int(cp["n"]) == n):
            real, neg = cp["real"], cp["neg"]
            start_i = int(cp["done"])
            print(f"  RESUMING from {ckpt_path} at query {start_i}/{n}")
        else:
            print(f"  ignoring {ckpt_path}: shape/seed mismatch")

    # The fp16 CPU-offloaded reader sits at ~14.07 GiB of a 15.45 GiB card.
    # Twice now an unrelated process has briefly taken ~800 MiB and OOM-killed
    # this run mid-flight (at q200 and q1150). The allocation is transient, so
    # back off and retry rather than losing the run; only give up after the
    # spike has failed to clear for ~5 minutes.
    def score_retry(query, context, tries=6):
        for t in range(tries):
            try:
                return S12.score_source(query, context, llm_tok, llm)
            except torch.OutOfMemoryError:
                wait = 15 * (t + 1)
                print(f"    OOM (attempt {t+1}/{tries}) -> empty_cache, "
                      f"retry in {wait}s", flush=True)
                torch.cuda.empty_cache()
                gc.collect()
                time.sleep(wait)
        print("    OOM persisted; emitting sentinel for this item", flush=True)
        return -1e9      # downstream `ok` mask drops sentinel rows

    t0 = time.time()
    for i in range(start_i, n):
        q = test[i]["query"]
        for j, s in enumerate(SOURCES):
            real[i, j] = score_retry(q, ctx[i][j])
            for m in range(M):
                neg[i, j, m] = score_retry(q, ctx[perms[m, i]][j])
        if (i + 1) % 50 == 0:
            np.savez(ckpt_path, real=real, neg=neg, done=i + 1, seed=SEED, n=n)
        if (i + 1) % 25 == 0:
            el = time.time() - t0
            rate = (i + 1 - start_i) / max(el, 1e-9)   # only work done THIS process
            print(f"  {i+1}/{n}  {rate:.2f} q/s  eta {(n-i-1)/max(rate,1e-9)/60:.1f} min",
                  flush=True)

    np.savez(S12.RESULTS_DIR / "ceg_scores.npz",
             real=real, neg=neg, gold=gold, perms=perms, lens=lens)

    ok = (real > -1e8).all(1) & (neg > -1e8).all((1, 2))
    R, N_, g = real[ok], neg[ok], gold[ok]
    print(f"\nscored cleanly: {ok.sum()}/{n}")

    out = {"n": int(n), "n_scored": int(ok.sum()), "donors": M, "seed": SEED,
           "macro": {}, "variance": {}}

    out["macro"]["raw"] = macro(R.argmax(1), g)
    # source-constant control: subtract the mean negative for that source
    src_const = N_.reshape(-1, 3, M).mean((0, 2))
    out["macro"]["minus_source_constant"] = macro((R - src_const).argmax(1), g)
    # query-conditional contrastive score, at each M
    for m in range(1, M + 1):
        G = R - N_[:, :, :m].mean(2)
        out["macro"][f"contrastive_M{m}"] = macro(G.argmax(1), g)
        out["variance"][f"contrastive_M{m}"] = {k: float(v) for k, v in decomp(-G).items()}

    out["variance"]["real"] = {k: float(v) for k, v in decomp(-R).items()}
    out["variance"]["negative_M1"] = {k: float(v) for k, v in decomp(-N_[:, :, 0]).items()}
    out["mean_nll"] = {"real": dict(zip(SOURCES, (-R.mean(0)).round(4).tolist())),
                       "negative": dict(zip(SOURCES, (-N_.mean((0, 2))).round(4).tolist()))}
    out["reference"] = {"raw_published": 0.5883, "source_calibration_oracle": 0.652,
                        "bge_confidence": 0.661, "sourcegate": 0.737, "random": 0.337}

    json.dump(out, open(S12.RESULTS_DIR / "contrastive_evidence_gain.json", "w"),
              indent=2)

    print("\n" + "=" * 64)
    for k, v in out["macro"].items():
        print(f"  {k:<28}{v:>10.4f}")
    print("-" * 64)
    for k, v in out["reference"].items():
        print(f"  {k:<28}{v:>10.4f}")
    print("=" * 64)
    print("  interaction, absolute mean square (nats^2):")
    print(f"    real          {out['variance']['real']['ms_interaction']:.5f}")
    print(f"    negative(M=1) {out['variance']['negative_M1']['ms_interaction']:.5f}")
    print(f"    contrastive   {out['variance'][f'contrastive_M{M}']['ms_interaction']:.5f}")
    print("\n  Q1 nuisance cancellation -> macro rises with M, beats 0.652")
    print("  Q2 interaction is evidence/noise -> macro ~0.588 or falls")


if __name__ == "__main__":
    main()
