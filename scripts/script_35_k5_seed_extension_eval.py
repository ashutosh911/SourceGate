"""
script_35_k5_seed_extension_eval.py  —  uniform K=5 evaluation over all seeds

WHY
  Section 7.3 lists "Three seeds at K=5" as a threat to validity, stating that
  three seeds give only limited protection against random-initialisation
  sensitivity. Supervised pretraining at K=5 costs ~20 s per seed (frozen
  embeddings, a 461K MLP, no LLM), so the sample size is cheap to enlarge.

  This script does NOT retrain. It evaluates every K=5 supervised-pretraining
  checkpoint on disk -- the three original seeds plus the seeds added by
  script_5b_k5.py --seeds ... -- through one code path, on both dev and test,
  so the resulting spread is comparable across all of them.

  Evaluating rather than reading stored summaries matters here: the K=5
  summary JSON is rewritten by each run with only that run's seeds, so the
  original three seeds' supervised-pretraining macros were no longer
  recoverable from it.

LABELS
  K=5 routes to individual datasets, so the label is the argmax of the
  per-dataset soft target (soft_target_k5 in script_5b_k5.py): dataset_score
  normalised over the five sources, queries with all-zero score dropped.

EMBEDDINGS
  BGE query embeddings depend only on the query text and prefix, not on K, so
  the K=3 dev/test caches are the same vectors the K=5 router consumes.

USAGE
  conda activate chestx && python script_35_k5_seed_extension_eval.py

OUTPUT
  phase5_results/k5_seed_extension.json
"""

import json
from pathlib import Path

import numpy as np
import torch

from sourceformer import SourceFormerK5, SOURCE_TYPES_K5

CKPT_DIR = Path("checkpoints")
RESULTS_DIR = Path("phase5_results")
ORIGINAL_SEEDS = [7, 99, 314]


def soft_target_k5(item):
    v = np.array([float(item["dataset_score"].get(t, 0.0))
                  for t in SOURCE_TYPES_K5], dtype=np.float32)
    return None if v.sum() == 0 else v / v.sum()


def load_split(name, emb_path):
    data = json.load(open(f"mmrag_{name}.json"))
    embs = np.load(emb_path).astype(np.float32)
    assert embs.shape[0] == len(data), f"{name}: {embs.shape[0]} vs {len(data)}"
    keep, labels = [], []
    for i, it in enumerate(data):
        s = soft_target_k5(it)
        if s is None:
            continue
        keep.append(i)
        labels.append(int(np.argmax(s)))
    return embs[keep], np.array(labels)


def macro(pred, gold):
    return float(np.mean([float((pred[gold == k] == k).mean())
                          for k in range(len(SOURCE_TYPES_K5))
                          if (gold == k).sum()]))


def main():
    splits = {
        "dev":  load_split("dev",  "query_emb_cache/dev_embs.npy"),
        "test": load_split("test", "query_emb_cache/test_embs.npy"),
    }
    for k, (e, g) in splits.items():
        print(f"  {k}: {len(g)} labelled queries")

    ckpts = sorted(CKPT_DIR.glob("sourceformer_k5_seed*_best.pt"),
                   key=lambda p: int(p.stem.split("seed")[1].split("_")[0]))
    if not ckpts:
        raise SystemExit("no K=5 checkpoints found")

    rows = []
    for p in ckpts:
        seed = int(p.stem.split("seed")[1].split("_")[0])
        ck = torch.load(p, map_location="cpu", weights_only=False)
        m = SourceFormerK5(dropout=0.2)
        m.load_state_dict(ck["state_dict"] if "state_dict" in ck else ck)
        m.eval()
        r = {"seed": seed, "original": seed in ORIGINAL_SEEDS}
        with torch.no_grad():
            for split, (e, g) in splits.items():
                pred = m(torch.from_numpy(e).float()).argmax(-1).numpy()
                r[f"{split}_macro"] = macro(pred, g)
                r[f"{split}_acc"] = float((pred == g).mean())
        rows.append(r)

    print(f"\n{'seed':>9} {'set':>9} {'dev macro':>10} {'test macro':>11}")
    for r in rows:
        print(f"{r['seed']:>9} {'original' if r['original'] else 'added':>9}"
              f" {r['dev_macro']:>10.4f} {r['test_macro']:>11.4f}")

    out = {"n_seeds": len(rows), "per_seed": rows}
    print(f"\n{'subset':<26}{'n':>3}{'dev mean':>10}{'dev sd':>9}"
          f"{'test mean':>11}{'test sd':>9}")
    for tag, sub in (("original three seeds", [r for r in rows if r["original"]]),
                     ("all seeds", rows)):
        if len(sub) < 2:
            continue
        d = [r["dev_macro"] for r in sub]
        t = [r["test_macro"] for r in sub]
        print(f"{tag:<26}{len(sub):>3}{np.mean(d):>10.4f}{np.std(d, ddof=1):>9.4f}"
              f"{np.mean(t):>11.4f}{np.std(t, ddof=1):>9.4f}")
        out[tag.replace(" ", "_")] = {
            "n": len(sub),
            "dev_mean": float(np.mean(d)), "dev_sd": float(np.std(d, ddof=1)),
            "test_mean": float(np.mean(t)), "test_sd": float(np.std(t, ddof=1)),
            "test_ci95_halfwidth": float(1.96 * np.std(t, ddof=1) / np.sqrt(len(t))),
        }

    a = out["all_seeds"]
    print(f"\n  all seeds, test macro 95% CI on the mean: "
          f"[{a['test_mean']-a['test_ci95_halfwidth']:.4f}, "
          f"{a['test_mean']+a['test_ci95_halfwidth']:.4f}]")
    json.dump(out, open(RESULTS_DIR / "k5_seed_extension.json", "w"), indent=2)
    print(f"\nSaved → {RESULTS_DIR/'k5_seed_extension.json'}")


if __name__ == "__main__":
    main()
