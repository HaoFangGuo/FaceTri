"""
case_study_sampler.py - Sample representative items for qualitative inspection
of InternVL3's abstention behaviour on the three wrinkle attributes.

Four case types, defined by whether the model abstained and whether the three
annotators agreed:

    A  abstained, annotators unanimous   - model hesitated where humans did not
    B  answered wrongly, annotators split - model should have abstained but did not
    C  abstained, annotators split        - well-aligned abstention
    D  answered correctly, annotators split - model resolved a case humans could not

Requires:
    wrinkle_rows_meta.csv
    prediction_internvl3_uncertain.csv

Usage:
    python case_study_sampler.py [--n 5] [--seed 42]
"""

import argparse
import re
import pandas as pd

META_FILE = "data/wrinkle_rows_meta.csv"
PRED_FILE = "results/predictions/prediction_internvl3_uncertain.csv"
OUT_FILE = "results/case_studies/case_studies_internvl3.csv"

CASE_DESCRIPTIONS = {
    "A": "abstained, annotators unanimous",
    "B": "answered wrongly, annotators split",
    "C": "abstained, annotators split",
    "D": "answered correctly, annotators split",
}


def parse_answer(text):
    """Parse raw model output into 'yes' / 'no' / 'uncertain' / None."""
    if pd.isna(text):
        return None
    s = str(text).strip().lower()
    if s in {"yes", "y", "1", "true"}:
        return "yes"
    if s in {"no", "n", "0", "false"}:
        return "no"
    if s in {"uncertain", "unsure", "unknown", "cannot answer", "n/a"}:
        return "uncertain"
    if re.search(r"\buncertain\b", s):
        return "uncertain"
    m = re.search(r"\b(yes|no)\b", s)
    if m:
        return m.group(1)
    return None


def assign_case(row):
    """Label each item with its case type, or None if it fits no case of interest."""
    abstained = row["parsed"] == "uncertain"
    unanimous = row["agreement_rate"] == 1.0
    correct = row["parsed"] == row["label"].lower()

    if abstained and unanimous:
        return "A"
    if not abstained and not unanimous and not correct:
        return "B"
    if abstained and not unanimous:
        return "C"
    if not abstained and not unanimous and correct:
        return "D"
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Sample case studies for qualitative inspection."
    )
    parser.add_argument("--n", type=int, default=5, help="Items per case type")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed")
    args = parser.parse_args()

    meta = pd.read_csv(META_FILE)
    pred = pd.read_csv(PRED_FILE)
    pred["parsed"] = pred["prediction"].apply(parse_answer)

    df = meta.merge(pred[["image_name", "attribute", "prediction", "parsed"]],
                    on=["image_name", "attribute"], how="left")
    n_missing = df["parsed"].isna().sum()
    if n_missing:
        print(f"[WARN] {n_missing} item(s) unmatched or unparseable, excluded")
        df = df.dropna(subset=["parsed"])

    df["case"] = df.apply(assign_case, axis=1)
    df["votes"] = df["n_yes"].astype(str) + " Yes / " + df["n_no"].astype(str) + " No"

    print("=== Case type counts across all 600 items ===")
    for case in ["A", "B", "C", "D"]:
        n = (df["case"] == case).sum()
        print(f"  {case}  {CASE_DESCRIPTIONS[case]:38s} n={n}")
    print()

    samples = []
    for case in ["A", "B", "C", "D"]:
        pool = df[df["case"] == case]
        if pool.empty:
            print(f"[WARN] No items of case {case}")
            continue
        take = min(args.n, len(pool))
        if take < args.n:
            print(f"[WARN] Case {case} has only {len(pool)} item(s)")
        samples.append(pool.sample(take, random_state=args.seed))

    if not samples:
        raise SystemExit("[ERROR] No cases found")

    out = pd.concat(samples, ignore_index=True)
    out = out[["case", "image_name", "attribute", "votes", "label",
               "prediction", "parsed", "agreement_rate", "entropy"]]
    out = out.rename(columns={"label": "majority_label", "prediction": "raw_output"})
    out["case_description"] = out["case"].map(CASE_DESCRIPTIONS)

    out.to_csv(OUT_FILE, index=False)

    for case in ["A", "B", "C", "D"]:
        sub = out[out["case"] == case]
        if sub.empty:
            continue
        print(f"--- Case {case}: {CASE_DESCRIPTIONS[case]} ---")
        print(sub[["image_name", "attribute", "votes", "majority_label", "parsed"]]
              .to_string(index=False))
        print()

    print(f"Saved to {OUT_FILE}")
    print("\nInspect each image alongside its votes column to judge whether the")
    print("model's decision was reasonable.")


if __name__ == "__main__":
    main()
