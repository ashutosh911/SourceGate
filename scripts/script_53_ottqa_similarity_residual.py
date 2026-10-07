"""
script_53_ottqa_similarity_residual.py  --  does the subsumption result
                                            replicate on OTT-QA?

THE mmRAG RESULT BEING REPLICATED
  Residuals added to SourceGate's log-probabilities, lambda chosen on held-out
  data, mmRAG K=3 test (per-seed mean over 3 seeds):

      SourceGate                      0.7365 +- 0.0049
      SG + BGE similarity             0.7849 +- 0.0017     sim beyond SG  p=0.0002
      SG + reader likelihood          0.7599 +- 0.0100
      SG + similarity + reader        0.7952 +- 0.0047     reader beyond sim p=0.26

  So the reader is subsumed: its contribution to routing is already available
  from the embedding's similarity to the retrieved content, which PrefRAG-style
  methods have computed anyway by the time they run their K forward passes.

  Both halves of that need a second benchmark: the similarity gain, and the
  reader's failure to add on top of it.

SIMILARITY ON OTT-QA
  OTT-QA stores its table and passage contexts in the benchmark file, so there
  is no FAISS index to read a max-similarity off.  The analogue is the cosine
  between the BGE query embedding and the BGE embedding of each stored
  context, using the same encoder, the same query prefix and the same L2
  normalisation as everywhere else.  This is a weaker proxy than mmRAG's
  max-over-retrieved-chunks, so if anything it understates the similarity
  signal -- which makes it a conservative test of subsumption.

PROTOCOL
  Cross-fit within test, matching script_49 and script_52's P2.
  SourceGate-K2 is retrained with model selection on a dev split carved from
  OTT-QA's training records, not on test (script_18 selected on test, which
  inflated the published Section 6.11 figure by about +0.0085).

PREDICTIONS, FIXED BEFORE RUNNING
  W1  Replicates: similarity beats SourceGate-K2 with a CI excluding zero, and
      the reader adds nothing significant on top.
  W2  Reader adds beyond similarity here -> subsumption is mmRAG-specific and
      cannot be stated as a general claim.
  W3  Similarity does not help here -> the mmRAG similarity gain may be an
      artefact of max-over-chunks retrieval rather than relevance.

USAGE
  conda activate chestx && python script_53_ottqa_similarity_residual.py

OUTPUT
  phase8_results_ottqa/similarity_residual_k2.json
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

import script_12_prefrag_conf as S12
import script_18_ottqa_k2_routing as S18
from sourceformer import SourceFormerK3

OUT = Path("phase8_results_ottqa")
SIM_CACHE = OUT / "ctx_similarity_k2.npy"
SEEDS = [42, 123, 2026]
SEED = 20260926
B = 10000
GRID = np.round(np.arange(-3, 5.001, 0.02), 3)
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


def fit_lambdas(SG, Zs, g, rounds=4):
    lam = np.zeros(len(Zs))
    for _ in range(rounds):
        for j in range(len(Zs)):
            bm, bv = -1.0, lam[j]
            for v in GRID:
                t = lam.copy()
                t[j] = v
                m = macro((SG + sum(t[k] * Zs[k] for k in range(len(Zs)))).argmax(1), g)
                if m > bm + 1e-12 or (abs(m - bm) <= 1e-12 and abs(v) < abs(bv)):
                    bm, bv = m, v
            lam[j] = bv
    return lam


# ── data ─────────────────────────────────────────────────────────────────────
train_rec = json.load(open(OUT / "ottqa_k2_train.json"))
test_rec = json.load(open(OUT / "ottqa_k2_test.json"))
tr_emb_all = np.load(OUT / "emb_cache/train_embs.npy").astype(np.float32)
te_emb = np.load(OUT / "emb_cache/test_embs.npy").astype(np.float32)
gold = np.array([int(r["label"]) for r in test_rec])
L = np.load(OUT / "prefrag_conf_k2_scores.npy")
n = len(gold)

# ── context similarity ───────────────────────────────────────────────────────
if SIM_CACHE.exists() and np.load(SIM_CACHE).shape == (n, 2):
    S = np.load(SIM_CACHE)
    print(f"  similarity cache hit: {SIM_CACHE}")
else:
    tok = AutoTokenizer.from_pretrained(S12.BGE_NAME)
    bge = AutoModel.from_pretrained(S12.BGE_NAME, torch_dtype=torch.float16).to(DEVICE).eval()

    def enc(texts, bs=64):
        out = np.empty((len(texts), 768), dtype=np.float32)
        with torch.inference_mode():
            for i in range(0, len(texts), bs):
                e = tok(texts[i:i + bs], padding=True, truncation=True, max_length=512,
                        return_tensors="pt").to(DEVICE)
                out[i:i + bs] = F.normalize(
                    bge(**e).last_hidden_state[:, 0].float(), p=2, dim=1).cpu().numpy()
        return out

    # contexts are documents, so they are encoded WITHOUT the query prefix,
    # which BGE applies to queries only
    S = np.zeros((n, 2))
    for j, field in enumerate(["table_context", "passage_context"]):
        C = enc([r[field] for r in test_rec])
        S[:, j] = (te_emb * C).sum(1)
        print(f"  encoded {field}: mean cos {S[:, j].mean():.4f}")
    np.save(SIM_CACHE, S)
    del bge, tok
    torch.cuda.empty_cache()

# ── SourceGate-K2, dev-selected ──────────────────────────────────────────────
tgt = torch.tensor([[r["dataset_score"][s] for s in SRC] for r in train_rec],
                   dtype=torch.float32)
hard = torch.tensor([r["label"] for r in train_rec], dtype=torch.long)
rng = np.random.default_rng(SEED)
perm = rng.permutation(len(train_rec))
dev_i, tr_i = perm[:4000], perm[4000:]
SG = log_softmax(np.mean([train_k2(tr_emb_all[tr_i], tgt[tr_i], hard[tr_i],
                                   tr_emb_all[dev_i], hard[dev_i].numpy(), te_emb, s)
                          for s in SEEDS], 0))

np.save(OUT / "sourcegate_k2_logits_devselected.npy", SG)

ok = (L > -1e8).all(1)
L, S, SG, g = L[ok], S[ok], SG[ok], gold[ok]
m = len(g)
sg_arg = SG.argmax(1)
print(f"\nOTT-QA test n={m}")
print(f"  SourceGate-K2 (dev-selected) {macro(sg_arg, g):.4f}")
print(f"  similarity argmax            {macro(S.argmax(1), g):.4f}")
print(f"  reader argmax                {macro(L.argmax(1), g):.4f}   (chance 0.484)")

MODELS = {"SG": [], "SG+sim": ["sim"], "SG+reader": ["reader"],
          "SG+sim+reader": ["sim", "reader"]}
half = rng.permutation(m)
A, Bx = half[: m // 2], half[m // 2:]
preds = {}
print("\n" + "=" * 74)
print("CROSS-FIT WITHIN TEST")
print("=" * 74)
for name, keys in MODELS.items():
    if not keys:
        preds[name] = sg_arg
    else:
        pr = np.zeros(m, dtype=int)
        for fit, app in ((A, Bx), (Bx, A)):
            Zf, Za = [], []
            for k in keys:
                X = {"sim": S, "reader": L}[k]
                mu, sd = X[fit].mean(0), X[fit].std(0)
                Zf.append((X[fit] - mu) / sd)
                Za.append((X[app] - mu) / sd)
            lam = fit_lambdas(SG[fit], Zf, g[fit])
            pr[app] = (SG[app] + sum(lam[k] * Za[k] for k in range(len(keys)))).argmax(1)
        preds[name] = pr
    print(f"  {name:<18}{macro(preds[name], g):.4f}")

out = {"n": int(m), "macros": {k: macro(v, g) for k, v in preds.items()},
       "likelihood_argmax": macro(L.argmax(1), g),
       "similarity_argmax": macro(S.argmax(1), g)}
key = boot_diff(preds["SG+sim+reader"], preds["SG+sim"], g, rng)
dsim = boot_diff(preds["SG+sim"], preds["SG"], g, rng)
print(f"\n  KEY  (SG+sim+reader) - (SG+sim) = {key['mean']:+.4f}  "
      f"CI [{key['ci95'][0]:+.4f}, {key['ci95'][1]:+.4f}]  p={key['p_two_sided']:.4f}")
print(f"  (SG+sim) - SG                   = {dsim['mean']:+.4f}  "
      f"CI [{dsim['ci95'][0]:+.4f}, {dsim['ci95'][1]:+.4f}]  p={dsim['p_two_sided']:.4f}")
out["key_reader_beyond_sim"] = key
out["sim_beyond_sg"] = dsim

json.dump(out, open(OUT / "similarity_residual_k2.json", "w"), indent=2)
print("\n  W1 replicates -> subsumption is general")
print("  W2 reader adds here -> subsumption is mmRAG-specific")
print(f"\nSaved -> {OUT / 'similarity_residual_k2.json'}")
