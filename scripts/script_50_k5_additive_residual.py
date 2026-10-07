"""
script_50_k5_additive_residual.py  --  third replication axis for the additive
                                       reader residual: 5-way routing

WHY K=5
  The residual  score_s(q) = log_softmax(SG(q))_s + lambda_s * z_s(q)  raises
  mmRAG K=3 routing macro from 0.7365 +- 0.0049 to 0.7599 +- 0.0100, improving
  every seed, with lambda chosen on dev.  OTT-QA supplies a second benchmark
  and a second reader configuration (4-bit NF4 rather than fp16).

  K=5 supplies something neither of those does: a harder decision.  The five
  source types are the individual datasets (nq, triviaqa, ott, tat, kg) rather
  than the three merged families, so chance falls from 0.333 to 0.200 and the
  router must separate nq from triviaqa and ott from tat -- pairs that are the
  same format and differ only in content.  If the reader's contribution were a
  format effect it should vanish exactly there.  If it is evidence
  compatibility, it should survive or grow.

  K=5 also removes a retrieval-budget asymmetry present at K=3: each source
  type is a single FAISS index, so every candidate gets exactly TOP_K chunks
  with no merging.

PROTOCOL
  Reader scores are computed for dev and test.  lambda is chosen on dev and
  applied once to frozen test, and a cross-fit-within-test estimate is reported
  alongside, matching scripts 44 and 48 so all three axes are comparable.

  Gold labels use script_35's convention: argmax over the five per-dataset
  relevance scores, dropping queries whose scores are all zero.

PREDICTIONS, FIXED BEFORE RUNNING
  K1  Replicates -> lambda* clearly non-zero, test macro above SourceGate-K5
      with a bootstrap CI excluding zero.
  K2  Fails at K=5 -> the K=3 gain came from separating the three FORMAT
      families, not from evidence, and the claim weakens to a format-family
      effect.

USAGE
  conda activate chestx && python script_50_k5_additive_residual.py [--n N]

OUTPUT
  phase5_results/k5_additive_residual.json
  phase5_results/prefrag_conf_k5_scores.npz
"""

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch

import script_12_prefrag_conf as S12
from sourceformer import SourceFormerK5, SOURCE_TYPES_K5

RESULTS = Path("phase5_results")
SCORES = RESULTS / "prefrag_conf_k5_scores.npz"
SEEDS = [7, 99, 314]
SEED = 20260926
B = 10000
GRID = np.round(np.arange(-3, 5.001, 0.02), 3)
K = 5


def macro(pred, g):
    v = [(pred[g == j] == j).mean() for j in range(K) if (g == j).sum()]
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


def soft_target_k5(item):
    v = np.array([float(item["dataset_score"].get(t, 0.0)) for t in SOURCE_TYPES_K5],
                 dtype=np.float32)
    return None if v.sum() == 0 else v / v.sum()


def sg_logits_k5(emb, seed=None):
    seeds = SEEDS if seed is None else [seed]
    outs = []
    for s in seeds:
        p = Path("checkpoints") / f"sourceformer_k5_seed{s}_best.pt"
        ck = torch.load(p, map_location="cpu", weights_only=False)
        m = SourceFormerK5() if "SourceFormerK5" in str(type(SourceFormerK5)) else SourceFormerK5()
        m.load_state_dict(ck["state_dict"] if "state_dict" in ck else ck)
        m.eval()
        with torch.no_grad():
            outs.append(m(torch.from_numpy(emb).float()).numpy())
    return np.mean(outs, 0)


def retrieve_ds(q_emb, ds, idx, k):
    """K=5: each source type IS one index, so no merging and no budget asymmetry.

    Also returns the top-1 inner product, which is the BGE-confidence signal.
    Capturing it here is free -- the search has already been done -- and it is
    the residual that subsumed the reader at K=3 and on OTT-QA.
    """
    D, ids = idx[ds].search(q_emb[None].astype(np.float32), k)
    ctx = "\n\n".join(S12.get_chunk_texts(ds, ids[0].tolist()))[:S12.MAX_CTX_CHARS]
    return ctx, float(D[0, 0])


def encode(queries, cache):
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModel
    from tqdm import tqdm
    cache = Path(cache)
    if cache.exists():
        e = np.load(cache)
        if e.shape[0] == len(queries):
            return e.astype(np.float32)
    tok = AutoTokenizer.from_pretrained(S12.BGE_NAME)
    bge = AutoModel.from_pretrained(S12.BGE_NAME, torch_dtype=torch.float16).to(S12.DEVICE).eval()
    qs = [S12.QUERY_PREFIX + q for q in queries]
    out = np.empty((len(qs), 768), dtype=np.float32)
    with torch.inference_mode():
        for s in tqdm(range(0, len(qs), S12.ENCODE_BATCH), desc=f"BGE {cache.name}"):
            e = min(s + S12.ENCODE_BATCH, len(qs))
            enc = tok(qs[s:e], padding=True, truncation=True, max_length=512,
                      return_tensors="pt").to(S12.DEVICE)
            out[s:e] = F.normalize(bge(**enc).last_hidden_state[:, 0].float(),
                                   p=2, dim=1).cpu().numpy()
    del bge, tok
    torch.cuda.empty_cache()
    np.save(cache, out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None)
    args = ap.parse_args()

    splits = {}
    for name in ("dev", "test"):
        data = json.load(open(f"mmrag_{name}.json"))
        keep, lab = [], []
        for i, it in enumerate(data):
            s = soft_target_k5(it)
            if s is not None:
                keep.append(i)
                lab.append(int(np.argmax(s)))
        if args.n:
            keep, lab = keep[: args.n], lab[: args.n]
        splits[name] = {"recs": [data[i] for i in keep], "gold": np.array(lab)}
        print(f"{name}: {len(keep)} queries with non-zero relevance   "
              f"counts {dict(zip(SOURCE_TYPES_K5, np.bincount(lab, minlength=K).tolist()))}")

    # embeddings, to split-specific caches (never the shared test_embs.npy)
    for name in ("dev", "test"):
        splits[name]["emb"] = encode([r["query"] for r in splits[name]["recs"]],
                                     f"query_emb_cache/k5_{name}_embs_prefrag.npy")

    if SCORES.exists():
        z = np.load(SCORES)
        if z["test"].shape[0] == len(splits["test"]["gold"]):
            for nm in ("dev", "test"):
                splits[nm]["L"] = z[nm]
                splits[nm]["S"] = z[f"sim_{nm}"]
            print(f"  score cache hit: {SCORES}")

    if "L" not in splits["test"]:
        idx = S12.load_indices()
        ctx = {}
        for name in ("dev", "test"):
            print(f"Retrieving {name} contexts (5 sources)...")
            rows = [[retrieve_ds(e, ds, idx, S12.TOP_K) for ds in SOURCE_TYPES_K5]
                    for e in splits[name]["emb"]]
            ctx[name] = [[c for c, _ in r] for r in rows]
            splits[name]["S"] = np.array([[s for _, s in r] for r in rows])
        del idx
        S12._pos_cache.clear()
        gc.collect()
        torch.cuda.empty_cache()

        tok, llm = S12.load_llm()
        for name in ("dev", "test"):
            recs = splits[name]["recs"]
            L = np.zeros((len(recs), K))
            t0 = time.time()
            for i, r in enumerate(recs):
                for j in range(K):
                    L[i, j] = S12.score_source(r["query"], ctx[name][i][j], tok, llm)
                if (i + 1) % 100 == 0:
                    rate = (i + 1) / max(time.time() - t0, 1e-9)
                    print(f"  {name} {i+1}/{len(recs)}  {rate:.2f} q/s  "
                          f"eta {(len(recs)-i-1)/rate/60:.1f} min", flush=True)
            splits[name]["L"] = L
        np.savez(SCORES, dev=splits["dev"]["L"], test=splits["test"]["L"],
                 sim_dev=splits["dev"]["S"], sim_test=splits["test"]["S"])
        del llm
        torch.cuda.empty_cache()

    # ── analysis ─────────────────────────────────────────────────────────────
    d, t = splits["dev"], splits["test"]
    okd, okt = (d["L"] > -1e8).all(1), (t["L"] > -1e8).all(1)
    Ld, gd = d["L"][okd], d["gold"][okd]
    Lt, gt = t["L"][okt], t["gold"][okt]
    SGd = log_softmax(sg_logits_k5(d["emb"][okd]))
    SGt = log_softmax(sg_logits_k5(t["emb"][okt]))
    mu, sd = Ld.mean(0), Ld.std(0)
    Zd, Zt = (Ld - mu) / sd, (Lt - mu) / sd
    sg_arg = SGt.argmax(1)
    rng = np.random.default_rng(SEED)

    print(f"\nK=5  chance {1/K:.3f}")
    print(f"  likelihood argmax   dev {macro(Ld.argmax(1), gd):.4f}  "
          f"test {macro(Lt.argmax(1), gt):.4f}")
    print(f"  SourceGate-K5       dev {macro(SGd.argmax(1), gd):.4f}  "
          f"test {macro(sg_arg, gt):.4f}")

    out = {"n_dev": int(len(gd)), "n_test": int(len(gt)),
           "likelihood_macro_test": macro(Lt.argmax(1), gt),
           "sourcegate_k5_macro_test": macro(sg_arg, gt)}

    Sd, St = d["S"][okd], t["S"][okt]
    musd, sdsd = Sd.mean(0), Sd.std(0)
    ZSd, ZSt = (Sd - musd) / sdsd, (St - musd) / sdsd
    print(f"  BGE similarity argmax dev {macro(Sd.argmax(1), gd):.4f}  "
          f"test {macro(St.argmax(1), gt):.4f}")
    out["similarity_macro_test"] = macro(St.argmax(1), gt)

    def fit_lambdas(SG, Zs, g, rounds=4):
        lam = np.zeros(len(Zs))
        for _ in range(rounds):
            for j in range(len(Zs)):
                bm, bv = -1.0, lam[j]
                for v in GRID:
                    tr = lam.copy()
                    tr[j] = v
                    m = macro((SG + sum(tr[k] * Zs[k] for k in range(len(Zs)))).argmax(1), g)
                    if m > bm + 1e-12 or (abs(m - bm) <= 1e-12 and abs(v) < abs(bv)):
                        bm, bv = m, v
                lam[j] = bv
        return lam

    MODELS = {"SG": [], "SG+sim": ["sim"], "SG+reader": ["reader"],
              "SG+sim+reader": ["sim", "reader"]}
    BD = {"sim": ZSd, "reader": Zd}
    BT = {"sim": ZSt, "reader": Zt}

    print("\n" + "=" * 74)
    print("P1  lambdas on DEV -> frozen TEST")
    print("=" * 74)
    pr1 = {}
    for name, keys in MODELS.items():
        if not keys:
            pr1[name] = sg_arg
        else:
            lam = fit_lambdas(SGd, [BD[k] for k in keys], gd)
            pr1[name] = (SGt + sum(lam[i] * BT[k] for i, k in enumerate(keys))).argmax(1)
        print(f"  {name:<18}{macro(pr1[name], gt):.4f}")
    k1 = boot_diff(pr1["SG+sim+reader"], pr1["SG+sim"], gt, rng)
    s1 = boot_diff(pr1["SG+sim"], pr1["SG"], gt, rng)
    print(f"\n  KEY  (SG+sim+reader)-(SG+sim) = {k1['mean']:+.4f}  "
          f"CI [{k1['ci95'][0]:+.4f}, {k1['ci95'][1]:+.4f}]  p={k1['p_two_sided']:.4f}")
    print(f"  (SG+sim)-SG                   = {s1['mean']:+.4f}  "
          f"CI [{s1['ci95'][0]:+.4f}, {s1['ci95'][1]:+.4f}]  p={s1['p_two_sided']:.4f}")
    out["P1"] = {"macros": {k: macro(v, gt) for k, v in pr1.items()},
                 "key_reader_beyond_sim": k1, "sim_beyond_sg": s1}

    print("\n" + "=" * 74)
    print("P2  cross-fit within TEST")
    print("=" * 74)
    half = rng.permutation(len(gt))
    A, Bx = half[: len(gt) // 2], half[len(gt) // 2:]
    pr2 = {}
    for name, keys in MODELS.items():
        if not keys:
            pr2[name] = sg_arg
        else:
            p = np.zeros(len(gt), dtype=int)
            for fit, app in ((A, Bx), (Bx, A)):
                Zf, Za = [], []
                for k in keys:
                    X = {"sim": St, "reader": Lt}[k]
                    mu_, sd_ = X[fit].mean(0), X[fit].std(0)
                    Zf.append((X[fit] - mu_) / sd_)
                    Za.append((X[app] - mu_) / sd_)
                lam = fit_lambdas(SGt[fit], Zf, gt[fit])
                p[app] = (SGt[app] + sum(lam[i] * Za[i] for i in range(len(keys)))).argmax(1)
            pr2[name] = p
        print(f"  {name:<18}{macro(pr2[name], gt):.4f}")
    k2 = boot_diff(pr2["SG+sim+reader"], pr2["SG+sim"], gt, rng)
    s2 = boot_diff(pr2["SG+sim"], pr2["SG"], gt, rng)
    print(f"\n  KEY  (SG+sim+reader)-(SG+sim) = {k2['mean']:+.4f}  "
          f"CI [{k2['ci95'][0]:+.4f}, {k2['ci95'][1]:+.4f}]  p={k2['p_two_sided']:.4f}")
    print(f"  (SG+sim)-SG                   = {s2['mean']:+.4f}  "
          f"CI [{s2['ci95'][0]:+.4f}, {s2['ci95'][1]:+.4f}]  p={s2['p_two_sided']:.4f}")
    out["P2"] = {"macros": {k: macro(v, gt) for k, v in pr2.items()},
                 "key_reader_beyond_sim": k2, "sim_beyond_sg": s2}

    json.dump(out, open(RESULTS / "k5_additive_residual.json", "w"), indent=2)
    print("\n  K1 subsumption replicates at K=5 -> claim holds where nq/triviaqa and")
    print("     ott/tat must be separated, i.e. where format cannot be the cue")
    print("  K2 reader adds at K=5 -> subsumption is a K=3 format-family artefact")
    print(f"\nSaved -> {RESULTS / 'k5_additive_residual.json'}")


if __name__ == "__main__":
    main()
