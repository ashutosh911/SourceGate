"""
script_51_residual_controls.py  --  is the additive residual about the READER,
                                    or would any second signal do?

THE RESULT BEING CONTROLLED
  Adding a standardised reader-likelihood term to SourceGate's log-probabilities
  raises mmRAG K=3 macro from 0.7365 +- 0.0049 to 0.7599 +- 0.0100 (every seed
  improves; lambda chosen on dev).  The natural objection is that this is not
  about the reader at all: bolting ANY second opinion onto a router might buy
  two macro points, in which case the finding is "ensembling helps" and says
  nothing about generation confidence.

  This runs the identical machinery -- same standardisation, same lambda grid,
  same dev-fit/test-frozen and cross-fit protocols, same bootstrap -- with the
  reader term replaced by other signals.

  CONTROL SIGNALS
    bge_confidence   max cosine similarity per source type, recomputed from the
                     FAISS indices.  Retrieval-based like the reader: it looks
                     at the corpus, not only the query.  The sharpest control,
                     because it shares the reader's "inspect what is there"
                     character while costing no forward passes.
    logreg           logits of a multinomial logistic probe on the frozen query
                     embedding, trained on the train split.
    mlp_hardce       logits of a hard-label MLP on the same embedding.
                     Both are query-only, like SourceGate itself, so they test
                     whether a merely CORRELATED second router helps.
    gaussian_noise   standardised noise, seeded.  Calibrates how much apparent
                     gain the lambda search manufactures from nothing; whatever
                     this scores is the floor any real signal must clear.

INTERPRETATION
  X1  Reader >> all controls
      -> the contribution is specific to conditioning on retrieved evidence
         with a language model, and the complementarity claim stands.
  X2  bge_confidence matches the reader
      -> the useful signal is "inspect the retrieved content", obtainable
         without any reader passes.  That is a BETTER practical result and a
         weaker scientific one, and it must be reported as such.
  X3  Query-only routers or noise match the reader
      -> the gain is an artefact of the lambda search, and the residual result
         must be withdrawn.

USAGE
  conda activate chestx && python script_51_residual_controls.py

OUTPUT
  phase5_results/residual_controls.json
"""

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

import script_12_prefrag_conf as S12
from sourceformer import SourceFormerK3

RESULTS = Path("phase5_results")
SOURCES = ["text", "table", "kg"]
SEEDS = [42, 123, 2026]
SEED = 20260926
B = 10000
GRID = np.round(np.arange(-3, 5.001, 0.02), 3)
SIM_CACHE = RESULTS / "bge_maxsim_scores.npz"


def macro(pred, g, k=3):
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


def sg_logits(emb):
    outs = []
    for s in SEEDS:
        ck = torch.load(Path("checkpoints") / f"sourceformer_k3_seed{s}_best.pt",
                        map_location="cpu", weights_only=False)
        m = SourceFormerK3(dropout=0.2)
        m.load_state_dict(ck["state_dict"] if "state_dict" in ck else ck)
        m.eval()
        with torch.no_grad():
            outs.append(m(torch.from_numpy(emb).float()).numpy())
    return np.mean(outs, 0)


def pick_lambda(SG, Z, g):
    best = max(((l, macro((SG + l * Z).argmax(1), g)) for l in GRID),
               key=lambda x: (x[1], -abs(x[0])))
    return float(best[0])


# ── data ─────────────────────────────────────────────────────────────────────
res = json.load(open(RESULTS / "prefrag_conf_results.json"))
Lt_all = np.array([[r["source_scores"][s] for s in SOURCES] for r in res])
gt_all = np.array([r["true_label"] for r in res])
ok_t = (Lt_all > -1e8).all(1)
Lt, gt = Lt_all[ok_t], gt_all[ok_t]
emb_t = np.load("query_emb_cache/test_embs.npy").astype(np.float32)[ok_t]

dev = np.load(RESULTS / "prefrag_conf_dev_scores.npz")
ok_d = (dev["scores"] > -1e8).all(1)
Ld, gd = dev["scores"][ok_d], dev["gold"][ok_d]
emb_d = dev["embs"][ok_d].astype(np.float32)

SGd, SGt = log_softmax(sg_logits(emb_d)), log_softmax(sg_logits(emb_t))
sg_arg = SGt.argmax(1)
print(f"dev n={len(gd)}  test n={len(gt)}   SourceGate test macro {macro(sg_arg, gt):.4f}")

# ── BGE max-similarity per source type ───────────────────────────────────────
if SIM_CACHE.exists():
    z = np.load(SIM_CACHE)
    Sd, St = z["dev"], z["test"]
    print(f"  bge max-sim cache hit: {SIM_CACHE}")
else:
    import faiss  # noqa: F401
    idx = S12.load_indices()

    def maxsim(embs):
        out = np.zeros((len(embs), 3))
        for j, st in enumerate(SOURCES):
            best = np.full(len(embs), -np.inf)
            for ds in S12.TYPE_TO_DATASETS_K3[st]:
                D, _ = idx[ds].search(embs.astype(np.float32), 1)
                best = np.maximum(best, D[:, 0])
            out[:, j] = best
        return out

    Sd, St = maxsim(emb_d), maxsim(emb_t)
    np.savez(SIM_CACHE, dev=Sd, test=St)
    del idx
print(f"  BGE-confidence argmax test macro {macro(St.argmax(1), gt):.4f}  (published 0.661)")

# ── query-only baselines, trained on the train split ─────────────────────────
tr_emb = np.load("query_emb_cache/train_embs_k3.npy").astype(np.float32)
train = json.load(open("mmrag_train.json"))
tr_gold = np.array([S12.hard_label_k3(t) for t in train])
assert len(tr_gold) == len(tr_emb)
sc = StandardScaler().fit(tr_emb)
lr = LogisticRegression(max_iter=3000).fit(sc.transform(tr_emb), tr_gold)
mlp = MLPClassifier(hidden_layer_sizes=(512, 128), max_iter=300,
                    random_state=SEED).fit(sc.transform(tr_emb), tr_gold)

rng = np.random.default_rng(SEED)
signals = {
    "reader_likelihood": (Ld, Lt),
    "bge_confidence": (Sd, St),
    "logreg_logits": (lr.decision_function(sc.transform(emb_d)),
                      lr.decision_function(sc.transform(emb_t))),
    "mlp_hardce_logits": (np.log(mlp.predict_proba(sc.transform(emb_d)) + 1e-9),
                          np.log(mlp.predict_proba(sc.transform(emb_t)) + 1e-9)),
    "gaussian_noise": (rng.standard_normal((len(gd), 3)),
                       rng.standard_normal((len(gt), 3))),
}

print("\n" + "=" * 88)
print(f"{'residual signal':<22}{'lam*':>7}{'P1 test':>10}{'d vs SG':>10}{'p':>8}"
      f"{'P2 test':>10}{'d vs SG':>10}{'p':>8}")
print("=" * 88)
out = {"sourcegate_test_macro": macro(sg_arg, gt), "controls": {}}

half = rng.permutation(len(gt))
A, Bx = half[: len(gt) // 2], half[len(gt) // 2:]

for name, (Xd, Xt) in signals.items():
    mu, sd_ = Xd.mean(0), Xd.std(0)
    Zd, Zt = (Xd - mu) / sd_, (Xt - mu) / sd_
    lam = pick_lambda(SGd, Zd, gd)
    p1 = (SGt + lam * Zt).argmax(1)
    d1 = boot_diff(p1, sg_arg, gt, rng)

    p2 = np.zeros(len(gt), dtype=int)
    for fit, app in ((A, Bx), (Bx, A)):
        m_, s_ = Xt[fit].mean(0), Xt[fit].std(0)
        l2 = pick_lambda(SGt[fit], (Xt[fit] - m_) / s_, gt[fit])
        p2[app] = (SGt[app] + l2 * ((Xt[app] - m_) / s_)).argmax(1)
    d2 = boot_diff(p2, sg_arg, gt, rng)

    out["controls"][name] = {"lambda_dev": lam, "P1_test_macro": macro(p1, gt),
                             "P1_vs_sg": d1, "P2_test_macro": macro(p2, gt),
                             "P2_vs_sg": d2}
    print(f"{name:<22}{lam:>+7.2f}{macro(p1, gt):>10.4f}{d1['mean']:>+10.4f}"
          f"{d1['p_two_sided']:>8.3f}{macro(p2, gt):>10.4f}{d2['mean']:>+10.4f}"
          f"{d2['p_two_sided']:>8.3f}")

json.dump(out, open(RESULTS / "residual_controls.json", "w"), indent=2)
print("\n  X1 reader >> controls        -> complementarity is reader-specific")
print("  X2 bge matches reader        -> 'inspect retrieved content', no reader needed")
print("  X3 query-only or noise match -> lambda search artefact; withdraw the result")
print(f"\nSaved -> {RESULTS / 'residual_controls.json'}")
