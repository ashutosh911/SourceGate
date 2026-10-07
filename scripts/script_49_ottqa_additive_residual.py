"""
script_49_ottqa_additive_residual.py  --  does the additive reader residual
                                          replicate on the second benchmark?

WHAT IS BEING REPLICATED
  On mmRAG, adding a standardised reader-likelihood term to SourceGate's
  log-probabilities,

      score_s(q) = log_softmax(SG(q))_s + lambda_s * z_s(q),

  raised routing macro from 0.7415 to 0.760-0.765 across four estimates
  (+0.018 to +0.023), three of which had bootstrap CIs excluding zero.  Since
  lambda = 0 reproduces SourceGate exactly, the family contains the incumbent
  and the gain cannot be an artefact of a weaker re-fitted baseline.

  A single-benchmark result cannot carry the claim, and OTT-QA is the right
  second test: its contexts are stored in the benchmark file, so it shares no
  retrieval code with mmRAG and was untouched by the chunk-lookup defect that
  invalidated the original diagnostic.

PROTOCOL
  Cross-fit within test (script_44/48's P2): lambda chosen on half A and
  applied to half B, and vice versa.  There are no dev-split reader scores for
  OTT-QA, and carving one would cost another ~8,000 reader passes; cross-fit is
  the same protocol that gave the cleanest mmRAG estimate and is honest with
  respect to SourceGate-K2, which never sees test.

  SourceGate-K2 is retrained here with model selection on a dev split carved
  from OTT-QA's 41,469 training records -- NOT on test, which is how
  script_18 trained it and which inflated the published Section 6.11 figure by
  about +0.0085 macro.

PREDICTIONS, FIXED BEFORE RUNNING
  V1  Replicates: test macro rises above dev-selected SourceGate-K2 with a
      bootstrap CI excluding zero, and lambda* is clearly non-zero.
  V2  Does not replicate: lambda* ~ 0 or the gain is within noise.  The mmRAG
      result would then be benchmark-specific and could not be the paper's
      central contribution.

USAGE
  conda activate chestx && python script_49_ottqa_additive_residual.py
  (requires phase8_results_ottqa/prefrag_conf_k2_scores.npy from script_47)

OUTPUT
  phase8_results_ottqa/additive_residual_k2.json
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import script_18_ottqa_k2_routing as S18
from sourceformer import SourceFormerK3

OUT = Path("phase8_results_ottqa")
SEEDS = [42, 123, 2026]
SEED = 20260926
B = 10000
GRID = np.round(np.arange(-1.0, 1.0001, 0.01), 3)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SRC = ["table", "passage"]


def macro(pred, g, k=2):
    v = [(pred[g == j] == j).mean() for j in range(k) if (g == j).sum()]
    return float(np.mean(v)) if v else float("nan")


def log_softmax(x):
    x = x - x.max(1, keepdims=True)
    return x - np.log(np.exp(x).sum(1, keepdims=True))


def boot_diff(pa, pb, g, rng, b=B):
    n = len(g)
    idx = rng.integers(0, n, size=(b, n))
    d = np.array([macro(pa[r], g[r]) - macro(pb[r], g[r]) for r in idx])
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"mean": float(d.mean()), "ci95": [float(lo), float(hi)],
            "p_two_sided": float(2 * min((d <= 0).mean(), (d >= 0).mean()))}


def train_k2(tr_emb, tr_tgt, tr_hard, va_emb, va_lab, te_emb, seed):
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
        return model(torch.from_numpy(te_emb).float().to(DEVICE)).cpu().numpy()


def pick_lambda(SG, Z, g):
    best_m, best_l = -1.0, 0.0
    for lam in GRID:
        m = macro((SG + lam * Z).argmax(1), g)
        if m > best_m + 1e-12 or (abs(m - best_m) <= 1e-12 and abs(lam) < abs(best_l)):
            best_m, best_l = m, lam
    return float(best_l)


def pick_lambda_per_source(SG, Z, g, rounds=3):
    lam = np.zeros(2)
    for _ in range(rounds):
        for j in range(2):
            best_m, best_v = -1.0, lam[j]
            for v in GRID:
                t = lam.copy()
                t[j] = v
                m = macro((SG + Z * t).argmax(1), g)
                if m > best_m + 1e-12 or (abs(m - best_m) <= 1e-12 and abs(v) < abs(best_v)):
                    best_m, best_v = m, v
            lam[j] = best_v
    return lam


# ── data ─────────────────────────────────────────────────────────────────────
train_rec = json.load(open(OUT / "ottqa_k2_train.json"))
test_rec = json.load(open(OUT / "ottqa_k2_test.json"))
tr_emb_all = np.load(OUT / "emb_cache/train_embs.npy").astype(np.float32)
te_emb = np.load(OUT / "emb_cache/test_embs.npy").astype(np.float32)
gold = np.array([int(r["label"]) for r in test_rec])
L = np.load(OUT / "prefrag_conf_k2_scores.npy")
assert len(L) == len(gold)

tgt = torch.tensor([[r["dataset_score"][s] for s in SRC] for r in train_rec],
                   dtype=torch.float32)
hard = torch.tensor([r["label"] for r in train_rec], dtype=torch.long)

rng = np.random.default_rng(SEED)
perm = rng.permutation(len(train_rec))
dev_i, tr_i = perm[:4000], perm[4000:]

SG = log_softmax(np.mean([train_k2(tr_emb_all[tr_i], tgt[tr_i], hard[tr_i],
                                   tr_emb_all[dev_i], hard[dev_i].numpy(), te_emb, s)
                          for s in SEEDS], 0))

ok = (L > -1e8).all(1)
L, g, SG = L[ok], gold[ok], SG[ok]
n = len(g)
sg_arg = SG.argmax(1)
print(f"OTT-QA test n={n}")
print(f"  SourceGate-K2 (dev-selected) macro {macro(sg_arg, g):.4f}   "
      f"(published, test-selected: 0.6756)")
print(f"  reader likelihood argmax     macro {macro(L.argmax(1), g):.4f}   "
      f"(published raw: 0.389)")

out = {"n": int(n), "sourcegate_k2_macro": macro(sg_arg, g),
       "likelihood_argmax_macro": macro(L.argmax(1), g)}

# ── cross-fit additive residual ──────────────────────────────────────────────
half = rng.permutation(n)
A, Bx = half[: n // 2], half[n // 2:]
print("\n" + "=" * 74)
print("CROSS-FIT ADDITIVE RESIDUAL")
print("=" * 74)
for tag, fn in (("scalar", pick_lambda), ("per_source", pick_lambda_per_source)):
    pred = np.zeros(n, dtype=int)
    lams = {}
    for name, fit, app in (("A->B", A, Bx), ("B->A", Bx, A)):
        # standardise using the FITTING half only
        mu, sd = L[fit].mean(0), L[fit].std(0)
        lam = fn(SG[fit], (L[fit] - mu) / sd, g[fit])
        lams[name] = float(lam) if np.isscalar(lam) else np.round(lam, 3).tolist()
        pred[app] = (SG[app] + ((L[app] - mu) / sd) * lam).argmax(1)
    d = boot_diff(pred, sg_arg, g, rng)
    out[tag] = {"lambdas": lams, "test_macro": macro(pred, g), **d}
    print(f"  {tag:<12}lambdas {lams}")
    print(f"  {'':12}macro {macro(pred, g):.4f}  vs SG {d['mean']:+.4f}  "
          f"CI [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]  p={d['p_two_sided']:.4f}")

json.dump(out, open(OUT / "additive_residual_k2.json", "w"), indent=2)
print("\n  V1 replicates  -> the residual is not benchmark-specific")
print("  V2 does not    -> mmRAG-only; cannot be the central claim")
print(f"\nSaved -> {OUT / 'additive_residual_k2.json'}")
