"""
case_study_bags_under_eyes.py — Select representative individual cases for
Bags_Under_Eyes to support qualitative case-study analysis in the paper.

Categories selected:
    1. Both models correct        (sanity check / easy cases)
    2. Both models wrong          (potential annotation noise or genuinely
                                    hard cases)
    3. Models disagree            (one correct, one wrong — architecture-
                                    dependent behavior)
    4. Either model says Uncertain (candidates for "grey area" discussion)
    5. Both models say Uncertain   (strongest grey-area candidates)

Usage:
    python case_study_bags_under_eyes.py
"""

import pandas as pd

ATTRIBUTE = "Bags_Under_Eyes"
N_EXAMPLES_PER_CATEGORY = 5


def parse_answer(text):
    """Same parsing logic as evaluate.py, kept in sync manually."""
    import re
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


def main():
    gt = pd.read_csv("data/benchmark/benchmark_final.csv")
    gt = gt[gt["attribute"] == ATTRIBUTE][["image_name", "label", "prompt"]]

    qwen = pd.read_csv("results/predictions/prediction_qwen_uncertain.csv")
    qwen = qwen[qwen["attribute"] == ATTRIBUTE][["image_name", "prediction"]]
    qwen = qwen.rename(columns={"prediction": "qwen_raw"})

    internvl3 = pd.read_csv("results/predictions/prediction_internvl3_uncertain.csv")
    internvl3 = internvl3[internvl3["attribute"] == ATTRIBUTE][["image_name", "prediction"]]
    internvl3 = internvl3.rename(columns={"prediction": "internvl3_raw"})

    df = gt.merge(qwen, on="image_name", how="left").merge(internvl3, on="image_name", how="left")

    df["y_true"] = df["label"].str.lower()
    df["qwen_parsed"] = df["qwen_raw"].apply(parse_answer)
    df["internvl3_parsed"] = df["internvl3_raw"].apply(parse_answer)

    df["qwen_correct"] = (df["qwen_parsed"] == df["y_true"])
    df["internvl3_correct"] = (df["internvl3_parsed"] == df["y_true"])
    df["qwen_uncertain"] = df["qwen_parsed"] == "uncertain"
    df["internvl3_uncertain"] = df["internvl3_parsed"] == "uncertain"

    categories = {
        "both_correct": df[df["qwen_correct"] & df["internvl3_correct"]],
        "both_wrong": df[
            (~df["qwen_correct"]) & (~df["internvl3_correct"])
            & (~df["qwen_uncertain"]) & (~df["internvl3_uncertain"])
        ],
        "disagreement": df[
            (df["qwen_correct"] != df["internvl3_correct"])
            & (~df["qwen_uncertain"]) & (~df["internvl3_uncertain"])
        ],
        "either_uncertain": df[df["qwen_uncertain"] | df["internvl3_uncertain"]],
        "both_uncertain": df[df["qwen_uncertain"] & df["internvl3_uncertain"]],
    }

    cols = ["image_name", "y_true", "qwen_raw", "qwen_parsed", "internvl3_raw", "internvl3_parsed"]

    for name, sub in categories.items():
        print(f"\n===== {name} (N={len(sub)}) =====")
        sample = sub[cols].head(N_EXAMPLES_PER_CATEGORY)
        print(sample.to_string(index=False))

    # Save full categorized data for later reference
    df["category"] = "other"
    for name, sub in categories.items():
        df.loc[sub.index, "category"] = name
    df.to_csv("results/case_studies/case_study_bags_under_eyes.csv", index=False)
    print(f"\nFull categorized data saved to case_study_bags_under_eyes.csv")


if __name__ == "__main__":
    main()
