"""
script_47_ottqa_k2_clean_and_complementarity.py

TWO JOBS, ONE PASS OVER OTT-QA

  A.  FIX A MODEL-SELECTION DEFECT IN A PUBLISHED NUMBER.
      script_18's train_sourcegate_k2 early-stops on the same 2,214 queries it
      then reports: `best_macro` is chosen by evaluating test_embs_t/test_labels
      every epoch (40 epochs, patience 5, 3 seeds).  Model selection and
      evaluation are the same set, so the published Section 6.11 figure
      (macro 0.6756 +- 0.0083) is optimistically biased.

      OTT-QA has 41,469 training records, so the fix is free: carve a dev split
      from train, early-stop on it, and evaluate once on the untouched 2,214.
      Both numbers are reported so the size of the bias is visible.

  B.  REPLICATE THE COMPLEMENTARITY RESULT ON A SECOND BENCHMARK.
      On mmRAG, reader likelihood adds macro beyond SourceGate's own logits --
      the one positive finding to survive the corrections.  A single-benchmark
      version of that claim will not carry a paper.  OTT-QA is independent of
      the mmRAG retrieval stack entirely: contexts are stored in the benchmark
      file, so no FAISS and no chunk lookup, which is also why OTT-QA escaped
      the defect that invalidated the mmRAG diagnostic.

      Combiners are cross-fitted within test (fit on half A, predict half B and
      vice versa), which is clean with respect to a SourceGate that never saw
      test.  The same protocol as script_44's P2, so the two benchmarks are
      directly comparable.

PREDICTIONS, FIXED BEFORE RUNNING
  R1  Complementarity replicates: sg+likelihood beats sg_only, CI excluding 0.
  R2  It does not replicate -> the mmRAG result is benchmark-specific and the
      complementarity claim cannot be the paper's central contribution.

USAGE
  conda activate chestx && python script_47_ottqa_k2_clean_and_complementarity.py

OUTPUT
  phase8_results_ottqa/k2_clean_and_complementarity.json
  phase8_results_ottqa/prefrag_conf_k2_scores.npy   (n,2) reader scores
"""

import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import script_18_ottqa_k2_routing as S18
from sourceformer import SourceFormerK3

OUT = Path("phase8_results_ottqa")
SCORES = OUT / "prefrag_conf_k2_scores.npy"
SEEDS = [42, 123, 2026]
SEED = 20260926
B = 10000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SRC = ["table", "passage"]


def macro(pred, g, k=2):
    v = [(pred[g == j] == j).mean() for j in range(k) if (g == j).sum()]
    return float(np.mean(v)) if v else float("nan")


def boot_diff(pa, pb, g, rng, b=B):
    n = len(g)
    idx = rng.integers(0, n, size=(b, n))
    d = np.array([macro(pa[r], g[r]) - macro(pb[r], g[r]) for r in idx])
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"mean": float(d.mean()), "ci95": [float(lo), float(hi)],
            "p_two_sided": float(2 * min((d <= 0).mean(), (d >= 0).mean()))}


def train_k2(tr_emb, tr_tgt, tr_hard, va_emb, va_lab, te_emb, seed):
    """Train SourceGate-K2, early-stopping on a split the caller chooses."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    counts = Counter(tr_hard.tolist())
    w = torch.tensor([1.0 / counts[int(l)] for l in tr_hard])
    sampler = torch.utils.data.WeightedRandomSampler(w, len(w), replacement=True)
    ds = torch.utils.data.TensorDataset(torch.from_numpy(tr_emb).float(), tr_tgt, tr_hard)
    loader = torch.utils.data.DataLoader(ds, batch_size=S18.BATCH, sampler=sampler)

    model = SourceFormerK3(input_dim=768, hidden=512, mid=128, k=2).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=S18.LR, weight_decay=S18.WD)
    va_t = torch.from_numpy(va_emb).float().to(DEVICE)
    best, best_state, bad = -1.0, None, 0
    for _ in range(S18.EPOCHS):
        model.train()
        for e, t, _ in loader:
            e, t = e.to(DEVICE), t.to(DEVICE)
            opt.zero_grad()
            S18.soft_cross_entropy(model(e), t).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            m = macro(model(va_t).argmax(-1).cpu().numpy(), va_lab)
        if m > best:
            best, best_state, bad = m, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= S18.PATIENCE:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(te_emb).float().to(DEVICE)).cpu().numpy(), best


# ── data ─────────────────────────────────────────────────────────────────────
train_rec = json.load(open(OUT / "ottqa_k2_train.json"))
test_rec = json.load(open(OUT / "ottqa_k2_test.json"))
tr_emb_all = np.load(OUT / "emb_cache/train_embs.npy").astype(np.float32)
te_emb = np.load(OUT / "emb_cache/test_embs.npy").astype(np.float32)
gold = np.array([int(r["label"]) for r in test_rec])
n = len(gold)
print(f"OTT-QA: {len(train_rec)} train / {n} test   "
      f"base rates {dict(zip(SRC, [float((gold == j).mean()) for j in range(2)]))}")

tgt_all = torch.tensor([[r["dataset_score"][s] for s in SRC] for r in train_rec],
                       dtype=torch.float32)
hard_all = torch.tensor([r["label"] for r in train_rec], dtype=torch.long)

rng = np.random.default_rng(SEED)
perm = rng.permutation(len(train_rec))
n_dev = 4000
dev_i, tr_i = perm[:n_dev], perm[n_dev:]
print(f"  carved dev split of {n_dev} from train; {len(tr_i)} left to train on")

out = {"n_test": int(n), "n_dev_carved": n_dev}

# ── A. clean vs test-selected SourceGate-K2 ──────────────────────────────────
print("\n" + "=" * 76)
print("A  SourceGate-K2: model selection on a carved dev split vs on test")
print("=" * 76)
clean_logits, clean_macros = [], []
for s in SEEDS:
    lg, bestdev = train_k2(tr_emb_all[tr_i], tgt_all[tr_i], hard_all[tr_i],
                           tr_emb_all[dev_i], hard_all[dev_i].numpy(), te_emb, s)
    clean_logits.append(lg)
    m = macro(lg.argmax(1), gold)
    clean_macros.append(m)
    print(f"  seed {s}: dev-selected, test macro {m:.4f}   (best dev {bestdev:.4f})")

testsel_macros = []
for s in SEEDS:
    lg, bestte = train_k2(tr_emb_all[tr_i], tgt_all[tr_i], hard_all[tr_i],
                          te_emb, gold, te_emb, s)          # reproduces the defect
    testsel_macros.append(macro(lg.argmax(1), gold))
    print(f"  seed {s}: TEST-selected, test macro {testsel_macros[-1]:.4f}")

out["A_model_selection"] = {
    "dev_selected_mean": float(np.mean(clean_macros)),
    "dev_selected_std": float(np.std(clean_macros, ddof=1)),
    "test_selected_mean": float(np.mean(testsel_macros)),
    "test_selected_std": float(np.std(testsel_macros, ddof=1)),
    "published": 0.6756,
    "optimistic_bias": float(np.mean(testsel_macros) - np.mean(clean_macros)),
}
print(f"\n  dev-selected  {np.mean(clean_macros):.4f} +- {np.std(clean_macros, ddof=1):.4f}")
print(f"  test-selected {np.mean(testsel_macros):.4f} +- {np.std(testsel_macros, ddof=1):.4f}"
      f"   (published 0.6756)")
print(f"  optimistic bias {np.mean(testsel_macros) - np.mean(clean_macros):+.4f}")

SG = np.mean(clean_logits, 0)

# ── B. reader likelihood on OTT-QA test ──────────────────────────────────────
print("\n" + "=" * 76)
print("B  PrefRAG-Conf question-token scores on OTT-QA test")
print("=" * 76)
if SCORES.exists() and np.load(SCORES).shape == (n, 2):
    L = np.load(SCORES)
    print(f"  cache hit: {SCORES}")
else:
    tok, llm = S18.load_llm_4bit()
    L = np.zeros((n, 2))
    t0 = time.time()
    for i, r in enumerate(test_rec):
        L[i, 0] = S18.score_source(r["query"], r["table_context"], tok, llm)
        L[i, 1] = S18.score_source(r["query"], r["passage_context"], tok, llm)
        if (i + 1) % 200 == 0:
            rate = (i + 1) / max(time.time() - t0, 1e-9)
            print(f"  {i+1}/{n}  {rate:.2f} q/s  eta {(n-i-1)/rate/60:.1f} min", flush=True)
    np.save(SCORES, L)
    del llm
    torch.cuda.empty_cache()

ok = (L > -1e8).all(1)
L, g, SGb = L[ok], gold[ok], SG[ok]
print(f"  scored {ok.sum()}/{n}   likelihood argmax macro {macro(L.argmax(1), g):.4f}"
      f"   (published raw 0.389)")
print(f"  SourceGate-K2 (dev-selected) macro {macro(SGb.argmax(1), g):.4f}")

# ── complementarity, cross-fit within test (script_44's P2 protocol) ─────────
print("\n" + "=" * 76)
print("B  COMPLEMENTARITY, cross-fit within test")
print("=" * 76)


def feats(S, Lk):
    return np.hstack([S, Lk, (Lk[:, [0]] - Lk[:, [1]])])


def fit_predict(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr)
    return LogisticRegression(max_iter=5000).fit(sc.transform(Xtr), ytr).predict(sc.transform(Xte))


m = len(g)
half = rng.permutation(m)
A, Bx = half[: m // 2], half[m // 2:]
preds = {}
for name, X in (("sg_only", SGb), ("sg_plus_likelihood", feats(SGb, L))):
    pr = np.zeros(m, dtype=int)
    pr[Bx] = fit_predict(X[A], g[A], X[Bx])
    pr[A] = fit_predict(X[Bx], g[Bx], X[A])
    preds[name] = pr
    print(f"  {name:<24}{macro(pr, g):.4f}")

d = boot_diff(preds["sg_plus_likelihood"], preds["sg_only"], g, rng)
print(f"\n  likelihood adds {d['mean']:+.4f}  CI [{d['ci95'][0]:+.4f}, "
      f"{d['ci95'][1]:+.4f}]  p={d['p_two_sided']:.4f}")
out["B_complementarity"] = {
    "likelihood_argmax_macro": macro(L.argmax(1), g),
    "sourcegate_argmax_macro": macro(SGb.argmax(1), g),
    "crossfit": {k: macro(v, g) for k, v in preds.items()},
    "delta_vs_sg_only": d,
}

json.dump(out, open(OUT / "k2_clean_and_complementarity.json", "w"), indent=2)
print("\n  R1 replicates -> complementarity survives on a second benchmark")
print("  R2 does not   -> the mmRAG result is benchmark-specific")
print(f"\nSaved -> {OUT / 'k2_clean_and_complementarity.json'}")
