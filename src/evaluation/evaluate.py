"""
evaluate.py — FaceVLM-bench lite evaluation script

Usage:
    python evaluate.py [--gt benchmark.csv] [--pred prediction.csv] [--out evaluation_results.csv]

Requires prediction.csv to contain at least: image_name, attribute, prediction
The prediction column can be raw model output text; this script parses it
into Yes / No / Uncertain automatically.

Metrics reported, per attribute and overall:
    - Precision / Recall / F1 / Accuracy, computed only on the "answered"
      subset (predictions parsed as Yes or No, excluding Uncertain and
      unparseable outputs)
    - Coverage: fraction of valid samples the model answered (Yes/No)
      rather than abstaining (Uncertain)
    - Risk: error rate among answered samples (1 - accuracy on answered)
    - Effective Reliability (Phi_c): scalar reliability metric from
      Whitehead et al., "Reliable Visual Question Answering: Abstain
      Rather Than Answer Incorrectly" (ECCV 2022). +1 per correct answer,
      -cost per incorrect answer, 0 per abstention, averaged over ALL
      valid samples (including abstained ones).
"""

import argparse
import re
import sys

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

EFFECTIVE_RELIABILITY_COST = 1.0  # cost c applied to incorrect answers


def parse_answer(text):
    """Parse raw model output into 'yes' / 'no' / 'uncertain' / None
    (unparseable)."""
    if pd.isna(text):
        return None
    s = str(text).strip().lower()

    # Exact match
    if s in {"yes", "y", "1", "true"}:
        return "yes"
    if s in {"no", "n", "0", "false"}:
        return "no"
    if s in {"uncertain", "unsure", "unknown", "cannot answer", "n/a"}:
        return "uncertain"

    # Word-boundary search within a longer response.
    # Check 'uncertain' before 'no' so phrases like "not certain" or
    # "I'm uncertain, no clear evidence" don't get mis-parsed as "no".
    if re.search(r"\buncertain\b", s):
        return "uncertain"
    m = re.search(r"\b(yes|no)\b", s)
    if m:
        return m.group(1)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default="data/benchmark/benchmark.csv")
    ap.add_argument("--pred", default="results/predictions/prediction.csv")
    ap.add_argument("--out", default="results/tables/evaluation_results.csv")
    args = ap.parse_args()

    gt = pd.read_csv(args.gt)
    pred = pd.read_csv(args.pred)

    # Accept common variants of the prediction column name
    pred_col = next(
        (c for c in ["prediction", "pred", "answer", "output", "response"] if c in pred.columns),
        None,
    )
    if pred_col is None:
        sys.exit(f"Error: no prediction column found in {args.pred} (prediction/pred/answer/output/response)")

    df = gt.merge(
        pred[["image_name", "attribute", pred_col]],
        on=["image_name", "attribute"],
        how="left",
    )

    missing = df[pred_col].isna().sum()
    if missing:
        print(f"Warning: {missing}/{len(df)} samples have no matching prediction, will be skipped")

    df["y_true"] = df["label"].map({"Yes": 1, "No": 0})
    df["parsed"] = df[pred_col].apply(parse_answer)

    unparsed = df["parsed"].isna() & df[pred_col].notna()
    if unparsed.sum():
        print(f"Warning: {unparsed.sum()} predictions could not be parsed into Yes/No/Uncertain, skipped. Examples:")
        print(df.loc[unparsed, pred_col].head(3).to_string(index=False))

    # 'valid' = every sample with a ground-truth label and a parseable
    # prediction (yes / no / uncertain). This is the denominator for
    # Coverage and Effective Reliability.
    valid = df.dropna(subset=["y_true"]).copy()
    valid = valid[valid["parsed"].notna()]
    if valid.empty:
        sys.exit("Error: no evaluable samples")

    valid["y_true"] = valid["y_true"].astype(int)
    valid["y_pred"] = valid["parsed"].map({"yes": 1, "no": 0})
    valid["answered"] = valid["parsed"] != "uncertain"
    valid["correct"] = (valid["y_pred"] == valid["y_true"]) & valid["answered"]

    def classification_metrics(sub):
        """Precision/Recall/F1/Accuracy computed on the answered
        (non-Uncertain) subset only."""
        answered_sub = sub[sub["answered"]]
        if answered_sub.empty:
            return {
                "N_answered": 0,
                "Accuracy": float("nan"),
                "Precision": float("nan"),
                "Recall": float("nan"),
                "F1": float("nan"),
            }
        y_true = answered_sub["y_true"]
        y_pred = answered_sub["y_pred"].astype(int)
        return {
            "N_answered": len(answered_sub),
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall": recall_score(y_true, y_pred, zero_division=0),
            "F1": f1_score(y_true, y_pred, zero_division=0),
        }

    def reliability_metrics(sub):
        """Coverage, Risk, and Effective Reliability (Whitehead et al.,
        ECCV 2022), computed over ALL valid samples including
        abstentions."""
        n_total = len(sub)
        n_answered = int(sub["answered"].sum())
        n_correct = int(sub["correct"].sum())
        n_incorrect = n_answered - n_correct

        coverage = n_answered / n_total if n_total else 0.0
        risk = n_incorrect / n_answered if n_answered else 0.0
        phi_c = (
            (n_correct - EFFECTIVE_RELIABILITY_COST * n_incorrect) / n_total
            if n_total else 0.0
        )
        return {
            "N": n_total,
            "Coverage": coverage,
            "Risk": risk,
            "EffectiveReliability": phi_c,
        }

    def all_metrics(sub):
        return {**reliability_metrics(sub), **classification_metrics(sub)}

    # Overall metrics
    overall = all_metrics(valid)
    print("\n===== Overall =====")
    for k, v in overall.items():
        print(f"{k:<20}: {v:.4f}" if isinstance(v, float) else f"{k:<20}: {v}")

    # Per-attribute metrics
    rows = []
    for attr, sub in valid.groupby("attribute"):
        rows.append({"attribute": attr, **all_metrics(sub)})
    rows.append({"attribute": "OVERALL", **overall})
    report = pd.DataFrame(rows)

    print("\n===== Per-attribute =====")
    print(report.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    report.to_csv(args.out, index=False)
    print(f"\nResults saved to {args.out}")


if __name__ == "__main__":
    main()

