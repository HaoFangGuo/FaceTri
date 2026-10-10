"""
Find disagreement samples, with proper normalization of raw model output
strings.

Raw predictions contain trailing punctuation and casing variants, e.g.
"Uncertain.", "Yes.", "No.". A naive string equality check against the
ground truth label silently counts these as incorrect, which corrupts any
downstream accuracy / coverage / risk metric.

This script:
  1. Normalizes every prediction to the canonical set {Yes, No, Uncertain}
  2. Reports any value that could not be mapped (so nothing is silently
     dropped or miscounted)
  3. Recomputes Case A (all three models wrong) and Case B (majority
     Uncertain while the label is definite) on normalized values
  4. Reports per-model confidently-wrong rate on the disagreement subset

Scope: Bags_Under_Eyes, Crows_Feet, Nasolabial_Folds, Big_Lips
"""

import re

import pandas as pd

TARGET_ATTRIBUTES = ["Bags_Under_Eyes", "Crows_Feet", "Nasolabial_Folds", "Big_Lips"]

BENCHMARK_FILE = "data/benchmark/benchmark_final.csv"
PREDICTION_FILES = {
    "qwen": "results/predictions/prediction_qwen_uncertain.csv",
    "internvl3": "results/predictions/prediction_internvl3_uncertain.csv",
    "molmo2o": "results/predictions/prediction_molmo2o_uncertain.csv",
}

CANONICAL = {"yes": "Yes", "no": "No", "uncertain": "Uncertain"}


def normalize_prediction(raw):
    """Map a raw model output string to {Yes, No, Uncertain}, or None.

    Strips surrounding whitespace and trailing punctuation, lowercases,
    then looks up the canonical form. Returns None when the value cannot
    be mapped, so unmappable outputs can be audited rather than silently
    treated as incorrect.
    """
    if pd.isna(raw):
        return None
    text = str(raw).strip()
    text = re.sub(r"[\s.,!;:]+$", "", text)
    return CANONICAL.get(text.lower())


def load_and_merge():
    bench = pd.read_csv(BENCHMARK_FILE)
    bench = bench[bench["attribute"].isin(TARGET_ATTRIBUTES)].copy()
    merged = bench[["image_name", "attribute", "label", "prompt"]].copy()

    audit = {}

    for model_name, path in PREDICTION_FILES.items():
        pred = pd.read_csv(path)
        pred = pred[pred["attribute"].isin(TARGET_ATTRIBUTES)].copy()

        raw_col = f"raw_{model_name}"
        norm_col = f"pred_{model_name}"
        pred[raw_col] = pred["prediction"]
        pred[norm_col] = pred["prediction"].apply(normalize_prediction)

        unmappable = pred.loc[pred[norm_col].isna(), raw_col]
        audit[model_name] = unmappable

        merged = merged.merge(
            pred[["image_name", "attribute", raw_col, norm_col]],
            on=["image_name", "attribute"],
            how="left",
        )

    return merged, audit


def report_audit(audit):
    print("--- Normalization audit ---")
    for model_name, unmappable in audit.items():
        if len(unmappable) == 0:
            print(f"  {model_name}: all values mapped cleanly")
        else:
            print(f"  {model_name}: {len(unmappable)} UNMAPPABLE values")
            print(f"    unique: {sorted(unmappable.unique().tolist())}")
    print()


def report_value_counts(df):
    print("--- Normalized prediction distribution (800 in-scope rows) ---")
    for model_name in PREDICTION_FILES:
        col = f"pred_{model_name}"
        counts = df[col].value_counts(dropna=False).to_dict()
        print(f"  {model_name}: {counts}")
    print()


def main():
    merged, audit = load_and_merge()
    report_audit(audit)
    report_value_counts(merged)

    pred_cols = [f"pred_{m}" for m in PREDICTION_FILES]

    # Case A: all three models predict incorrectly
    all_wrong = pd.Series(True, index=merged.index)
    for col in pred_cols:
        all_wrong &= merged[col] != merged["label"]

    # Case B: majority of models answer Uncertain while label is definite
    uncertain_count = sum((merged[col] == "Uncertain").astype(int) for col in pred_cols)
    case_b = (uncertain_count >= 2) & merged["label"].isin(["Yes", "No"])

    subset = merged[all_wrong | case_b].copy()

    print("--- Disagreement counts (after normalization) ---")
    print(f"  In-scope rows: {len(merged)}")
    print(f"  Case A (all three wrong): {all_wrong.sum()}")
    print(f"  Case B (majority Uncertain, definite label): {case_b.sum()}")
    print(f"  Union subset size: {len(subset)}")
    print()

    print("--- Per-model accuracy on all 800 in-scope rows ---")
    for model_name in PREDICTION_FILES:
        col = f"pred_{model_name}"
        acc = (merged[col] == merged["label"]).mean()
        print(f"  {model_name}: {acc:.3f}")
    print()

    if len(subset) > 0:
        print("--- Confidently-wrong rate within disagreement subset ---")
        rates = {}
        for model_name in PREDICTION_FILES:
            col = f"pred_{model_name}"
            confidently_wrong = subset[col].isin(["Yes", "No"]) & (subset[col] != subset["label"])
            rates[model_name] = confidently_wrong.mean()
            print(f"  {model_name}: {confidently_wrong.sum()}/{len(subset)} ({confidently_wrong.mean():.3f})")

        top_model = max(rates, key=rates.get)
        print(f"\nModel with highest confidently-wrong rate: {top_model}")

    subset.to_csv("results/cells/disagreement_samples.csv", index=False)
    print(f"\nSaved {len(subset)} rows to disagreement_samples.csv")


if __name__ == "__main__":
    main()
