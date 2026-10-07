"""
script_36_verify_2x2_and_bootstrap.py  —  independent re-derivation of the
                                          2x2 factorial and the paired
                                          bootstrap confidence intervals

WHY
  Several published numbers turned out to depend on evaluation paths that
  were individually inconsistent (retrieval budget, gold-label convention,
  query-encoding prefix, chunk-text lookup). This script re-derives two
  results that have not yet been checked, from checkpoints and saved
  predictions, using the canonical conventions established elsewhere:

    gold label   argmax over SUMMED per-source dataset_score
                 (717 text / 359 table / 210 kg on test)
    embeddings   query_emb_cache/test_embs.npy -- BGE with the training
                 QUERY_PREFIX and L2 normalisation

  It reports only; it writes no manuscript values.

WHAT IT CHECKS
  1. Table 6, the 2x2 factorial (loss type x sampler type), from the twelve
     checkpoints in checkpoints_ablation_2x2/, against the stored
     phase7_results/ablation_2x2_results.json and the values in the paper.

  2. Section 6.3's paired non-parametric bootstrap of the macro-accuracy
     difference between each SourceGate configuration and BGE-confidence:
     10,000 replicates, resampling test queries with replacement, pairing
     predictions by query identity.

     Published: supervised pretraining  +0.082, 95% CI [+0.046, +0.117]
                joint training          +0.056, 95% CI [+0.041, +0.072]
     The paper states the supervised figure was computed on seed-42
     predictions (macro 0.7425), so that is reproduced specifically, and
     the three-seed mean is reported alongside.

USAGE
  conda activate chestx && python script_36_verify_2x2_and_bootstrap.py

OUTPUT
  phase5_results/verify_2x2_and_bootstrap.json
"""

import json
from pathlib import Path

import numpy as np
import torch

from sourceformer import SourceFormerK3

RESULTS_DIR = Path("phase5_results")
SEEDS = [42, 123, 2026]
CELLS = ["CE_Uniform", "CE_Weighted", "KL_Uniform", "KL_Weighted"]
SOURCE_TYPES = ["text", "table", "kg"]

PAPER_2X2 = {  # Table 6: macro, std, text, table, kg
    "CE_Uniform":  (0.683, 0.028, 0.790, 0.783, 0.476),
    "CE_Weighted": (0.715, 0.008, 0.656, 0.813, 0.676),
    "KL_Uniform":  (0.714, 0.027, 0.747, 0.785, 0.611),
    "KL_Weighted": (0.717, 0.005, 0.679, 0.808, 0.663),
}
BGE_MACRO_PUBLISHED = 0.661


def gold_sum(it):
    s = it["dataset_score"]
    return int(np.argmax([s.get("nq", 0) + s.get("triviaqa", 0),
                          s.get("ott", 0) + s.get("tat", 0), s.get("kg", 0)]))


def macro(pred, gold):
    return float(np.mean([float((pred[gold == k] == k).mean())
                          for k in range(3) if (gold == k).sum()]))


def per_type(pred, gold):
    return {t: float((pred[gold == k] == k).mean())
            for k, t in enumerate(SOURCE_TYPES) if (gold == k).sum()}


test = json.load(open("mmrag_test.json"))
gold = np.array([gold_sum(it) for it in test])
n = len(gold)
embs = np.load("query_emb_cache/test_embs.npy").astype(np.float32)
assert embs.shape[0] == n
print(f"test queries: {n}   class counts: "
      f"{ {t: int((gold==k).sum()) for k,t in enumerate(SOURCE_TYPES)} }")


def decisions(ckpt_path):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    m = SourceFormerK3(dropout=0.2)
    m.load_state_dict(ck["state_dict"] if "state_dict" in ck else ck)
    m.eval()
    with torch.no_grad():
        return m(torch.from_numpy(embs).float()).argmax(-1).numpy()


out = {}

# ── 1. 2x2 factorial ─────────────────────────────────────────────────────────
print("\n" + "=" * 78)
print("1. 2x2 FACTORIAL (Table 6) -- recomputed from checkpoints")
print("=" * 78)
print(f"{'cell':<13}{'paper':>8}{'stored':>9}{'recomputed':>12}{'sd':>8}"
      f"{'text':>8}{'table':>8}{'kg':>8}")
stored = json.load(open("phase7_results/ablation_2x2_results.json"))
stored_by = {c["cell_label"]: c for c in stored["cells"]}
cell_rows = {}
for cell in CELLS:
    ms, pts = [], []
    for s in SEEDS:
        p = Path("checkpoints_ablation_2x2") / f"{cell}_seed{s}_best.pt"
        if not p.exists():
            print(f"  MISSING {p}")
            break
        d = decisions(p)
        ms.append(macro(d, gold))
        pts.append(per_type(d, gold))
    if len(ms) != len(SEEDS):
        continue
    pt = {t: float(np.mean([x[t] for x in pts])) for t in SOURCE_TYPES}
    pm = PAPER_2X2[cell][0]
    sm = stored_by[cell]["test_macro"]
    rm = float(np.mean(ms))
    cell_rows[cell] = {"paper_macro": pm, "stored_macro": sm,
                       "recomputed_macro": rm,
                       "recomputed_sd": float(np.std(ms, ddof=1)),
                       "per_seed": ms, "per_type": pt,
                       "delta_vs_paper": rm - pm}
    print(f"{cell:<13}{pm:>8.3f}{sm:>9.4f}{rm:>12.4f}"
          f"{np.std(ms, ddof=1):>8.4f}{pt['text']:>8.3f}{pt['table']:>8.3f}"
          f"{pt['kg']:>8.3f}")
out["factorial"] = cell_rows

mx = max(abs(v["recomputed_macro"] - v["paper_macro"]) for v in cell_rows.values())
print(f"\n  largest |recomputed - paper| across the four cells: {mx:.4f}")
print("  main effects (recomputed):")
ce = np.mean([cell_rows[c]["recomputed_macro"] for c in ("CE_Uniform", "CE_Weighted")])
kl = np.mean([cell_rows[c]["recomputed_macro"] for c in ("KL_Uniform", "KL_Weighted")])
un = np.mean([cell_rows[c]["recomputed_macro"] for c in ("CE_Uniform", "KL_Uniform")])
we = np.mean([cell_rows[c]["recomputed_macro"] for c in ("CE_Weighted", "KL_Weighted")])
print(f"    loss:    KL - CE       = {kl-ce:+.4f}")
print(f"    sampler: Weighted - Uniform = {we-un:+.4f}")
out["main_effects"] = {"kl_minus_ce": float(kl - ce),
                       "weighted_minus_uniform": float(we - un)}

# ── 2. paired bootstrap vs BGE-confidence ────────────────────────────────────
print("\n" + "=" * 78)
print("2. PAIRED BOOTSTRAP vs BGE-confidence (Section 6.3)")
print("=" * 78)
conf_p = RESULTS_DIR / "confidence_decisions_k3.npy"
if not conf_p.exists():
    raise SystemExit("confidence_decisions_k3.npy not found")
conf = np.load(conf_p).astype(int)
print(f"  BGE-confidence macro (recomputed): {macro(conf, gold):.4f}"
      f"   published {BGE_MACRO_PUBLISHED}")

CFG = {
    "supervised_pretraining": ("checkpoints/sourceformer_k3_seed{s}_best.pt",
                               (0.082, 0.046, 0.117)),
    "joint_training": ("checkpoints_phase4/phase4_seed{s}_best.pt",
                       (0.056, 0.041, 0.072)),
}
rng = np.random.default_rng(0)
B = 10000
idx = rng.integers(0, n, size=(B, n))          # shared resamples across configs

boot = {}
for name, (pat, pub) in CFG.items():
    decs = {s: decisions(pat.format(s=s)) for s in SEEDS
            if Path(pat.format(s=s)).exists()}
    if not decs:
        print(f"  {name}: no checkpoints found")
        continue
    macros = {s: macro(d, gold) for s, d in decs.items()}
    entry = {"per_seed_macro": macros,
             "mean_macro": float(np.mean(list(macros.values())))}
    for tag, d in (("seed42", decs.get(42)), ):
        if d is None:
            continue
        diffs = np.empty(B)
        for b in range(B):
            r = idx[b]
            diffs[b] = macro(d[r], gold[r]) - macro(conf[r], gold[r])
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        pm, plo, phi = pub
        entry[tag] = {"point_diff": float(macros[42] - macro(conf, gold)),
                      "bootstrap_mean": float(diffs.mean()),
                      "ci95": [float(lo), float(hi)],
                      "p_one_sided": float((diffs <= 0).mean()),
                      "published": {"mean": pm, "ci95": [plo, phi]}}
        print(f"\n  {name} (seed 42, macro {macros[42]:.4f}):")
        print(f"    point difference   {entry[tag]['point_diff']:+.4f}")
        print(f"    bootstrap mean     {diffs.mean():+.4f}   published {pm:+.3f}")
        print(f"    95% CI            [{lo:+.4f}, {hi:+.4f}]  "
              f"published [{plo:+.3f}, {phi:+.3f}]")
        print(f"    one-sided p        {(diffs<=0).mean():.5f}")
    boot[name] = entry
out["bootstrap"] = boot

json.dump(out, open(RESULTS_DIR / "verify_2x2_and_bootstrap.json", "w"), indent=2)
print(f"\nSaved → {RESULTS_DIR/'verify_2x2_and_bootstrap.json'}")
