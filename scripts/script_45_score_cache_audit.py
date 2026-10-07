"""Forensic integrity and score-range audit for null-evidence caches."""

import argparse
import json
from pathlib import Path

import numpy as np


SOURCES = ("text", "table", "kg")


def quantiles(values):
    a = np.asarray(values, dtype=float)
    return {str(q): float(np.quantile(a, q)) for q in (0, .5, .9, .95, .99, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    path = Path("phase9_results") / f"null_evidence_cache_{args.tag}_m5.json"
    obj = json.load(open(path))
    rows = list(obj.values())
    report = {"path": str(path), "rows": len(rows), "sources": {}}
    report["unique_query_indices"] = len({r["query_idx"] for r in rows})
    report["key_matches_query_idx"] = all(str(r["query_idx"]) in obj for r in rows)
    for source in SOURCES:
        raw = [r["raw"][source] for r in rows]
        null = [v for r in rows for v in r["negative_scores"][source]]
        lengths = [r["context_tokens"][source] for r in rows]
        flags = [v for r in rows for v in r["negative_required_padding"][source]]
        report["sources"][source] = {
            "raw_finite": bool(np.isfinite(raw).all()),
            "null_finite": bool(np.isfinite(null).all()),
            "raw_exact_zero": int(np.sum(np.asarray(raw) == 0)),
            "null_exact_zero": int(np.sum(np.asarray(null) == 0)),
            "raw_quantiles": quantiles(raw),
            "null_quantiles": quantiles(null),
            "context_token_quantiles": quantiles(lengths),
            "contexts_ge_900_tokens": int(np.sum(np.asarray(lengths) >= 900)),
            "padding_rate": float(np.mean(flags)),
            "all_have_five_nulls": all(len(r["negative_scores"][source]) == 5 for r in rows),
        }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
