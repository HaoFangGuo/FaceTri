"""
heavy_makeup_bias_analysis.py — Group model predictions by Heavy_Makeup
status to check whether makeup is a confound behind the "grey area"
observed on Bags_Under_Eyes, Crows_Feet, and Nasolabial_Folds.

Requires: CelebAMask-HQ-attribute-anno.txt containing a Heavy_Makeup
column, indexed by image_name (e.g. "0.jpg", "1.jpg", ...).

Usage:
    python heavy_makeup_bias_analysis.py
"""

import re
import pandas as pd

FLAGGED_ATTRS = ["Bags_Under_Eyes", "Crows_Feet", "Nasolabial_Folds"]

MODEL_FILES = {
    "qwen": "results/predictions/prediction_qwen_uncertain.csv",
    "internvl3": "results/predictions/prediction_internvl3_uncertain.csv",
    "molmo2o": "results/predictions/prediction_molmo2o_uncertain.csv",
}


def parse_answer(text):
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


def load_makeup_labels():
    """Load Heavy_Makeup ground truth, indexed by image_name
    (e.g. '0.jpg'), as in the CelebAMask-HQ annotation file."""
    anno = pd.read_csv(
        "data/CelebAMask-HQ-attribute-anno.txt", sep=r"\s+", skiprows=1, engine="python"
    )
    anno.index = anno.index.astype(str)
    makeup = anno["Heavy_Makeup"].map({1: "yes", -1: "no"})
    return makeup


def group_metrics(sub):
    n_total = len(sub)
    answered = sub[sub["parsed"] != "uncertain"]
    n_answered = len(answered)
    n_correct = (answered["parsed"] == answered["y_true"]).sum()
    n_incorrect = n_answered - n_correct

    coverage = n_answered / n_total if n_total else float("nan")
    risk = n_incorrect / n_answered if n_answered else float("nan")
    accuracy = n_correct / n_answered if n_answered else float("nan")

    return {
        "N": n_total,
        "Coverage": coverage,
        "Risk": risk,
        "Accuracy_answered": accuracy,
        "Uncertain_rate": 1 - coverage if n_total else float("nan"),
    }


def main():
    gt = pd.read_csv("data/benchmark/benchmark_final.csv")
    gt = gt[gt["attribute"].isin(FLAGGED_ATTRS)][["image_name", "attribute", "label"]]

    makeup = load_makeup_labels()
    gt["heavy_makeup"] = gt["image_name"].map(makeup)

    missing_makeup = gt["heavy_makeup"].isna().sum()
    if missing_makeup:
        print(f"Warning: {missing_makeup} rows have no Heavy_Makeup label, will be dropped")
    gt = gt.dropna(subset=["heavy_makeup"])

    all_results = []

    for model_name, pred_file in MODEL_FILES.items():
        pred = pd.read_csv(pred_file)
        pred = pred[pred["attribute"].isin(FLAGGED_ATTRS)][["image_name", "attribute", "prediction"]]

        df = gt.merge(pred, on=["image_name", "attribute"], how="left")
        df["y_true"] = df["label"].str.lower()
        df["parsed"] = df["prediction"].apply(parse_answer)
        df = df.dropna(subset=["parsed"])

        for attr in FLAGGED_ATTRS:
            for makeup_status in ["yes", "no"]:
                sub = df[(df["attribute"] == attr) & (df["heavy_makeup"] == makeup_status)]
                if sub.empty:
                    continue
                metrics = group_metrics(sub)
                all_results.append({
                    "model": model_name,
                    "attribute": attr,
                    "heavy_makeup": makeup_status,
                    **metrics,
                })

    report = pd.DataFrame(all_results)
    report.to_csv("results/case_studies/heavy_makeup_bias_analysis.csv", index=False)

    print(report.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nSaved to heavy_makeup_bias_analysis.csv")


if __name__ == "__main__":
    main()
