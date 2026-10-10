"""
mcnemar_order_swap.py — compute the McNemar exact test p-value for the
order-swap experiment (qwen, same 88 cases, answer-first vs reason-first).

Data used:
  reinference_qwen_reason.csv            - answer-first run
                                            (Answer: <X> / Reason: <text>)
  reinference_qwen_reasonfirst_reason.csv - reason-first run
                                            (Reason: <text> / Answer: <X>)

Both files have one row per case_id (the same 88 sampled cases in both),
with a `new_answer` column holding the model's answer under that prompt
ordering. This script pairs the two files by case_id and tests whether the
abstention rate (new_answer == "Uncertain") differs between the two
orderings, using a paired (McNemar) test since it is the same items and
same model, just a different prompt ordering.

Usage:
    python mcnemar_order_swap.py
"""

import numpy as np
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar

ANSWER_FIRST_FILE = "results/reinference/reinference_qwen_reason.csv"
REASON_FIRST_FILE = "results/reinference/reinference_qwen_reasonfirst_reason.csv"


def main():
    af = pd.read_csv(ANSWER_FIRST_FILE)[["case_id", "new_answer"]]
    rf = pd.read_csv(REASON_FIRST_FILE)[["case_id", "new_answer"]]

    merged = af.merge(rf, on="case_id", suffixes=("_af", "_rf"))
    print(f"Paired cases: {len(merged)}")

    before = (merged["new_answer_af"] == "Uncertain").to_numpy()  # answer-first
    after = (merged["new_answer_rf"] == "Uncertain").to_numpy()   # reason-first

    both_no = int(np.sum((~before) & (~after)))
    zero_to_one = int(np.sum((~before) & after))   # started answering, now abstains
    one_to_zero = int(np.sum(before & (~after)))   # started abstaining, now answers
    both_yes = int(np.sum(before & after))

    table = [[both_no, zero_to_one], [one_to_zero, both_yes]]
    discordant = zero_to_one + one_to_zero
    exact = discordant < 25

    result = mcnemar(table, exact=exact, correction=not exact)

    print(f"\nAbstention rate, answer-first:  {before.mean():.3f}")
    print(f"Abstention rate, reason-first:   {after.mean():.3f}")
    print(f"Discordant pairs: {discordant} "
          f"(started answering -> now abstains: {zero_to_one}; "
          f"started abstaining -> now answers: {one_to_zero})")
    print(f"Test used: {'exact binomial' if exact else 'chi2 with continuity correction'}")
    print(f"\nMcNemar p-value: {result.pvalue:.4g}")

    if discordant < 10:
        print("[CAUTION] fewer than 10 discordant pairs -- test may be underpowered")


if __name__ == "__main__":
    main()
