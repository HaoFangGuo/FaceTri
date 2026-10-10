"""
presentation_analysis.py - Produce analysis tables for the five focus
attributes, using an intuitive contingency framing rather than correlation
coefficients.

Focus attributes and their roles:
    Sideburns          (CelebA)     - clear-cut baseline, validates the setup
    Bags_Under_Eyes    (CelebA)     - hard attribute, no multi-annotator data
    Crows_Feet         (own labels) - highest annotator disagreement
    Nasolabial_Folds   (own labels) - counter-example, abstention unrelated to ambiguity
    Forehead_Wrinkles  (own labels) - highest annotator agreement among the three

Outputs:
    Table 1 - per-model, per-attribute Coverage / Risk / F1
    Table 2 - abstention rate split by whether annotators agreed

Requires:
    benchmark_final.csv
    wrinkle_rows_meta.csv
    prediction_{model}_uncertain.csv

Usage:
    python presentation_analysis.py
"""

import re
import pandas as pd

BENCHMARK_FILE = "data/benchmark/benchmark_final.csv"
META_FILE = "data/wrinkle_rows_meta.csv"

MODEL_FILES = {
    "Qwen2.5-VL-7B": "results/predictions/prediction_qwen_uncertain.csv",
    "InternVL3-8B": "results/predictions/prediction_internvl3_uncertain.csv",
    "Molmo2-O-7B": "results/predictions/prediction_molmo2o_uncertain.csv",
}

# Attribute -> (label source, role in the narrative)
FOCUS_ATTRS = {
    "Sideburns": ("CelebA", "clear-cut baseline"),
    "Bags_Under_Eyes": ("CelebA", "hard, single-annotator"),
    "Crows_Feet": ("ours", "highest disagreement"),
    "Nasolabial_Folds": ("ours", "counter-example"),
    "Forehead_Wrinkles": ("ours", "highest agreement"),
}

WRINKLE_ATTRS = [a for a, (src, _) in FOCUS_ATTRS.items() if src == "ours"]


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


def compute_metrics(sub):
    """Coverage, Risk, and F1 over the answered subset."""
    n = len(sub)
    answered = sub[sub["parsed"].isin(["yes", "no"])]
    coverage = len(answered) / n if n else float("nan")

    nan = float("nan")
    if len(answered) == 0:
        return {"n": n, "coverage": coverage, "risk": nan, "accuracy": nan,
                "precision": nan, "recall": nan, "f1": nan}

    gt = answered["label"].str.lower()
    pred = answered["parsed"]
    correct = (gt == pred).sum()

    tp = ((gt == "yes") & (pred == "yes")).sum()
    fp = ((gt == "no") & (pred == "yes")).sum()
    fn = ((gt == "yes") & (pred == "no")).sum()

    precision = tp / (tp + fp) if (tp + fp) else nan
    recall = tp / (tp + fn) if (tp + fn) else nan
    if precision == precision and recall == recall and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = nan

    return {
        "n": n,
        "coverage": coverage,
        "risk": 1 - correct / len(answered),
        "accuracy": correct / len(answered),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def table_one(bench, predictions):
    """Per-model, per-attribute Coverage / Risk / F1 for the focus attributes."""
    rows = []
    for model, pred in predictions.items():
        df = bench.merge(pred[["image_name", "attribute", "parsed"]],
                         on=["image_name", "attribute"], how="left")
        for attr, (source, role) in FOCUS_ATTRS.items():
            sub = df[df["attribute"] == attr]
            if sub.empty:
                print(f"[WARN] {attr} not found in benchmark, skipped")
                continue
            m = compute_metrics(sub)
            rows.append({
                "model": model,
                "attribute": attr,
                "labels": source,
                "coverage": round(m["coverage"], 3),
                "risk": round(m["risk"], 3),
                "precision": round(m["precision"], 3),
                "recall": round(m["recall"], 3),
                "f1": round(m["f1"], 3),
            })
    return pd.DataFrame(rows)


def table_two(meta, predictions):
    """Abstention rate split by whether the three annotators agreed."""
    rows = []
    for model, pred in predictions.items():
        df = meta.merge(pred[["image_name", "attribute", "parsed"]],
                        on=["image_name", "attribute"], how="left")
        df = df.dropna(subset=["parsed"])
        df["abstained"] = df["parsed"] == "uncertain"
        # With three annotators, agreement_rate is 1.0 when unanimous and 0.667 when split 2:1
        df["unanimous"] = df["agreement_rate"] == 1.0

        for attr in WRINKLE_ATTRS + ["ALL"]:
            sub = df if attr == "ALL" else df[df["attribute"] == attr]
            split = sub[~sub["unanimous"]]
            unanimous = sub[sub["unanimous"]]

            rate_split = split["abstained"].mean() if len(split) else float("nan")
            rate_unanimous = unanimous["abstained"].mean() if len(unanimous) else float("nan")

            rows.append({
                "model": model,
                "attribute": attr,
                "n_split": len(split),
                "n_unanimous": len(unanimous),
                "abstain_when_split": round(rate_split, 3),
                "abstain_when_unanimous": round(rate_unanimous, 3),
                "gap_pp": round(100 * (rate_split - rate_unanimous), 1),
                "ratio": (round(rate_split / rate_unanimous, 2)
                          if rate_unanimous and rate_unanimous == rate_unanimous
                          else float("nan")),
            })
    return pd.DataFrame(rows)


def main():
    bench = pd.read_csv(BENCHMARK_FILE)
    meta = pd.read_csv(META_FILE)

    predictions = {}
    for model, path in MODEL_FILES.items():
        try:
            pred = pd.read_csv(path)
        except FileNotFoundError:
            print(f"[WARN] {path} not found, skipping {model}")
            continue
        pred["parsed"] = pred["prediction"].apply(parse_answer)
        predictions[model] = pred

    if not predictions:
        raise SystemExit("[ERROR] No prediction files found")

    print("=" * 76)
    print("TABLE 1 - Performance on the five focus attributes")
    print("=" * 76)
    t1 = table_one(bench, predictions)
    for model in t1["model"].unique():
        print(f"\n{model}")
        print(t1[t1["model"] == model].drop(columns=["model"]).to_string(index=False))
    t1.to_csv("results/tables/table1_focus_attributes.csv", index=False)

    print("\n" + "=" * 76)
    print("TABLE 2 - Abstention rate by annotator agreement")
    print("=" * 76)
    print("Reads as: when the three annotators disagreed, how often did the model")
    print("abstain, compared with items where all three agreed?\n")

    t2 = table_two(meta, predictions)
    for model in t2["model"].unique():
        sub = t2[t2["model"] == model]
        never_abstained = (sub["abstain_when_split"].fillna(0).eq(0).all()
                           and sub["abstain_when_unanimous"].fillna(0).eq(0).all())
        if never_abstained:
            print(f"{model}: never abstained on these attributes, comparison not applicable\n")
            continue
        print(model)
        print(sub.drop(columns=["model"]).to_string(index=False))
        print()

    t2.to_csv("results/tables/table2_abstention_by_agreement.csv", index=False)
    print("Saved: table1_focus_attributes.csv, table2_abstention_by_agreement.csv")


if __name__ == "__main__":
    main()
