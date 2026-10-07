"""
script_39_ottqa_scrambled_control.py  —  the scrambled-context control on the
                                         second benchmark

WHY
  On mmRAG the control established that the routing signal's query x source
  interaction term is genuinely relevance-driven: destroying relevance while
  holding source format and context length fixed collapses that term from
  21.7% of variance to 5.6% and drops routing to chance (0.588 -> 0.357).

  OTT-QA is the benchmark where the diagnostic result survives intact, and it
  behaves differently: there the equal-length residual is not merely
  uninformative but anti-informative (macro 0.346 and 0.334 against a random
  baseline of 0.484). Whether that anti-informative residual is relevance-
  driven or an artifact of source identity has not been tested.

  This runs the same control on OTT-QA, using the benchmark's own stored
  table/passage contexts rather than FAISS retrieval, so it is independent of
  the mmRAG retrieval stack entirely.

DESIGN
  For each query i and source s, score query i's question tokens against the
  context source s supplied for a different query j, drawn from a seeded
  derangement. Format, style and length distribution are preserved; only
  relevance is destroyed.

PREDICTIONS, FIXED BEFORE RUNNING
  P1  The anti-informative residual is relevance-driven (mirrors mmRAG).
      -> interaction share collapses, macro moves UP toward chance (0.484)
         from its real-context value of 0.389, because the systematic wrong-
         source preference is destroyed along with the relevance signal.
  P2  The anti-informative residual reflects source identity, not relevance.
      -> interaction share and macro both stay near their real-context values
         (~0.389, below random), because the wrong-source preference does not
         depend on the context actually matching the query.

  P2 would be the stronger diagnostic result: it would mean the signal does
  not merely fail to find the right source, it prefers the wrong one for
  reasons unrelated to evidence.

USAGE
  conda activate chestx && python script_39_ottqa_scrambled_control.py [--n N]

OUTPUT
  phase5_results/ottqa_scrambled_control.json
"""

import argparse
import json
import time

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

SOURCES = ["table", "passage"]          # label 0 = table, 1 = passage
SEED = 20260925
LLM = "meta-llama/Llama-3.1-8B-Instruct"
DEVICE = "cuda"
PROMPT = ("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
          "Answer the question using only the provided context. Be concise."
          "<|eot_id|>\n<|start_header_id|>user<|end_header_id|>\n"
          "Context:\n{context}\n\nQuestion: {query}<|eot_id|>\n"
          "<|start_header_id|>assistant<|end_header_id|>\n")


@torch.no_grad()
def score_question_nll(model, tok, query, context):
    """Mean log-prob of question tokens given context (PrefRAG-Conf style)."""
    ctx_ids = tok(PROMPT.format(context=context, query=""),
                  return_tensors="pt", add_special_tokens=False)["input_ids"].to(DEVICE)
    full_ids = tok(PROMPT.format(context=context, query=query),
                   return_tensors="pt", add_special_tokens=False)["input_ids"].to(DEVICE)
    if full_ids.shape[1] > 1500 or ctx_ids.shape[1] >= full_ids.shape[1]:
        return float("nan")
    labels = full_ids.clone()
    labels[:, :ctx_ids.shape[1]] = -100
    return -model(input_ids=full_ids, labels=labels).loss.item()


def decomp(nll):
    n, k = nll.shape
    g = nll.mean(); qm = nll.mean(1, keepdims=True); sm = nll.mean(0, keepdims=True)
    tot = ((nll - g) ** 2).sum()
    q = k * ((qm - g) ** 2).sum(); s = n * ((sm - g) ** 2).sum()
    return q / tot, s / tot, (tot - q - s) / tot


def macro(pred, gold, k=2):
    return float(np.mean([(pred[gold == j] == j).mean()
                          for j in range(k) if (gold == j).sum()]))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=None)
    args = ap.parse_args()

    test = json.load(open("phase8_results_ottqa/ottqa_k2_test.json"))
    n = len(test) if args.n is None else min(args.n, len(test))
    test = test[:n]
    gold = np.array([int(r["label"]) for r in test])

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n)
    for i in np.flatnonzero(perm == np.arange(n)):
        j = (i + 1) % n
        perm[i], perm[j] = perm[j], perm[i]
    assert not (perm == np.arange(n)).any()
    print(f"{n} queries; derangement seed {SEED}")
    print(f"  base rate: table {(gold==0).mean():.1%}  passage {(gold==1).mean():.1%}")

    tok = AutoTokenizer.from_pretrained(LLM)
    model = AutoModelForCausalLM.from_pretrained(
        LLM, dtype=torch.float16, device_map="auto",
        max_memory={0: "14GiB", "cpu": "4GiB"}, low_cpu_mem_usage=True).eval()

    scores = np.full((n, 2), np.nan)
    t0 = time.time()
    for i, r in enumerate(test):
        donor = test[perm[i]]
        scores[i, 0] = score_question_nll(model, tok, r["query"], donor["table_context"])
        scores[i, 1] = score_question_nll(model, tok, r["query"], donor["passage_context"])
        if (i + 1) % 200 == 0:
            el = time.time() - t0
            print(f"  {i+1}/{n}  {(i+1)/max(el,1):.2f} q/s  eta {(n-i-1)/max((i+1)/max(el,1),1e-9)/60:.1f} min")

    ok = ~np.isnan(scores).any(1)
    nll = -scores[ok]; g = gold[ok]; pred = nll.argmin(1)
    q, s, inter = decomp(nll)

    real = {"macro": 0.389, "table_acc": 0.428, "passage_acc": 0.350,
            "table_pick": 0.598, "random_macro": 0.484}
    out = {"n_scored": int(ok.sum()), "seed": SEED,
           "scrambled": {"variance_query": float(q), "variance_source": float(s),
                         "variance_interaction": float(inter),
                         "macro": macro(pred, g),
                         "table_acc": float((pred[g == 0] == 0).mean()),
                         "passage_acc": float((pred[g == 1] == 1).mean()),
                         "table_pick": float((pred == 0).mean())},
           "real_context_reference": real}
    json.dump(out, open("phase5_results/ottqa_scrambled_control.json", "w"), indent=2)

    sc = out["scrambled"]
    print("\n" + "=" * 62)
    print(f"{'':<22}{'real ctx':>12}{'scrambled':>12}")
    print("=" * 62)
    print(f"{'variance interaction':<22}{'--':>12}{inter:>12.1%}")
    print(f"{'routing macro':<22}{real['macro']:>12.3f}{sc['macro']:>12.3f}")
    print(f"{'table acc':<22}{real['table_acc']:>12.3f}{sc['table_acc']:>12.3f}")
    print(f"{'passage acc':<22}{real['passage_acc']:>12.3f}{sc['passage_acc']:>12.3f}")
    print(f"{'table pick-rate':<22}{real['table_pick']:>12.1%}{sc['table_pick']:>12.1%}")
    print(f"\n  random macro {real['random_macro']:.3f}")
    print("  P1 relevance-driven  -> macro moves UP toward 0.484")
    print("  P2 source-identity   -> macro stays ~0.389, below random")


if __name__ == "__main__":
    main()
