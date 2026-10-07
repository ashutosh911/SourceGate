"""
script_59_ottqa_length_format_split.py

Does the abstract's OTT-QA "length closes 71%, leaving ~29% to format"
decomposition reproduce?

Context: the mmRAG counterpart ("equalising to 85 tokens narrows the KG NLL
advantage by 47%, leaving ~53% to format") does NOT reproduce -- the v2 audit
found the text-minus-KG gap actually WIDENED under equalisation (-0.1957 ->
-0.2329) rather than narrowing. So the OTT-QA figure cannot be assumed sound
just because its routing-accuracy cells verify; those cells
(verify_ottqa_equal_length.json) are accuracies, not the NLL decomposition.

Decomposition as stated: of the original per-source NLL advantage, the share
removed by equalising context length is attributed to LENGTH; the surviving
share is attributed to FORMAT.

    share_length = 1 - |gap_equalised| / |gap_original|
    share_format =     |gap_equalised| / |gap_original|

WARNING ON CONVENTIONS: ottqa_equal_length_summary.json reports the original
gap as negative values (log-likelihood, higher = better) but the per-condition
mean_nll as positive, and its per-condition macro is NaN. The two are not on a
common scale, so nothing here is taken from that file -- every number is
recomputed from the per-query caches, which are internally consistent
(both negative, log-likelihood convention).

OTT-QA K=2: table (long, 235.1 tok) vs passage (short, 84.7 tok).
On mmRAG the favoured source is the SHORT one (KG); here it is the LONG one
(table) -- the "reversed polarity" the paper claims.

CPU only.
"""
import json
import numpy as np

rng = np.random.default_rng(20260929)
D = "phase8_results_ottqa"
OUT = f"{D}/ottqa_length_format_split.json"

# ---------------------------------------------------------- original scores
orig = np.load(f"{D}/prefrag_conf_k2_scores.npy")   # (N,2) col0=table col1=passage
print(f"original scores {orig.shape}  (log-likelihood convention, higher=better)")


def cache_to_array(path, n):
    c = json.load(open(path))
    a = np.full((n, 2), np.nan)
    for k, v in c.items():
        a[int(k), 0] = v["table"]
        a[int(k), 1] = v["passage"]
    return a


N = orig.shape[0]
condA = cache_to_array(f"{D}/ottqa_equal_length_cache_A.json", N)   # 85 tokens
condB = cache_to_array(f"{D}/ottqa_equal_length_cache_B.json", N)   # 235 tokens

res = {"n": int(N), "convention": "log-likelihood, higher=better; "
                                  "gap = mean(table) - mean(passage)"}


def gap(a, label):
    ok = np.isfinite(a).all(1)
    g = float(a[ok, 0].mean() - a[ok, 1].mean())
    # paired bootstrap on the per-query difference
    d = a[ok, 0] - a[ok, 1]
    bs = np.array([d[rng.integers(0, d.sum() * 0 + len(d), len(d))].mean()
                   for _ in range(4000)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print(f"  {label:<26} n={ok.sum():>5}  table-minus-passage gap = "
          f"{g:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
    return g, float(lo), float(hi), int(ok.sum())


print("\nper-source gaps:")
g0, l0, h0, n0 = gap(orig, "original (variable length)")
gA, lA, hA, nA = gap(condA, "equalised @ 85 tokens")
gB, lB, hB, nB = gap(condB, "equalised @ 235 tokens")

res["gap_original"] = {"gap": g0, "ci95": [l0, h0], "n": n0}
res["gap_equalised_85"] = {"gap": gA, "ci95": [lA, hA], "n": nA}
res["gap_equalised_235"] = {"gap": gB, "ci95": [lB, hB], "n": nB}

print("\ndecomposition (share of the original advantage removed by equalising "
      "length):")
for label, g in [("@ 85 tokens", gA), ("@ 235 tokens", gB)]:
    share_len = 1.0 - abs(g) / abs(g0)
    share_fmt = abs(g) / abs(g0)
    widened = abs(g) > abs(g0)
    flipped = np.sign(g) != np.sign(g0)
    print(f"  {label:<14} length {100*share_len:6.1f}%   format {100*share_fmt:6.1f}%"
          + ("   <-- GAP WIDENED, decomposition undefined" if widened else "")
          + ("   <-- SIGN FLIPPED" if flipped else ""))
    res[f"decomposition_{label.strip('@ ').replace(' ', '_')}"] = {
        "share_length": float(share_len), "share_format": float(share_fmt),
        "gap_widened": bool(widened), "sign_flipped": bool(flipped)}

print("\n" + "=" * 68)
print("PAPER CLAIMS (abstract): OTT-QA length closes 71%, leaving ~29% to format")
print("                          mmRAG  length closes 47%, leaving ~53% to format")
print("                          (the mmRAG figure already failed to reproduce)")
print("=" * 68)

with open(OUT, "w") as f:
    json.dump(res, f, indent=2)
print(f"wrote {OUT}")
