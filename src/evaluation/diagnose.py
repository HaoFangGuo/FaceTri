"""
Diagnostic checks for FaceVLM-Bench Lite prediction files.

Checks four things that could produce the suspicious Precision/Recall = 1.0000
cells seen in the per-attribute results:
  1. Raw prediction string distribution (what the model literally emitted)
  2. Parse failure rate (outputs that map to neither Yes nor No)
  3. Per-attribute yes-rate (a rate near 0.0 or 1.0 means degenerate predictions)
  4. Join integrity against the benchmark file (row loss or duplicate keys)

It then recomputes per-attribute Precision/Recall/F1 from scratch, reporting
the raw confusion-matrix counts (TP/FP/FN/TN) so that metrics resting on a
handful of samples are immediately visible.

Usage:
    python diagnose.py --pred-dir results --bench benchmark.csv
"""

import argparse
import glob
import os

import pandas as pd


def parse_answer(x):
    """Map a raw model output string to 1 (Yes), 0 (No), or None (unparseable)."""
    if not isinstance(x, str):
        return None
    s = x.strip().lower().strip(".!,\"' \n")
    if s.startswith("yes"):
        return 1
    if s.startswith("no"):
        return 0
    return None


def parse_label(x):
    """Map a ground-truth label string to 1/0."""
    if not isinstance(x, str):
        return None
    s = x.strip().lower().strip(".!,\"' \n")
    if s.startswith("yes"):
        return 1
    if s.startswith("no"):
        return 0
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", default="results",
                    help="Directory containing per-model prediction CSVs")
    ap.add_argument("--bench", default="data/benchmark/benchmark.csv",
                    help="Benchmark CSV with ground-truth labels")
    args = ap.parse_args()

    bench = pd.read_csv(args.bench)
    print("=" * 70)
    print("BENCHMARK FILE:", args.bench)
    print("=" * 70)
    print("columns :", bench.columns.tolist())
    print("rows    :", len(bench))
    print("dup keys:", bench.duplicated(["image_name", "attribute"]).sum())
    print("\nlabel distribution per attribute (should be 50/50):")
    bench["gt"] = bench["label"].apply(parse_label)
    print(
        bench.groupby("attribute")["gt"]
        .agg(n="size", n_pos="sum", pos_rate="mean")
        .to_string(float_format=lambda v: f"{v:.3f}")
    )

    bench_keys = bench[["image_name", "attribute", "gt"]]

    pred_files = sorted(glob.glob(os.path.join(args.pred_dir, "*.csv")))
    if not pred_files:
        print(f"\nNo CSV files found in {args.pred_dir!r}")
        return

    for path in pred_files:
        pred = pd.read_csv(path)
        print("\n\n" + "=" * 70)
        print("MODEL FILE:", os.path.basename(path))
        print("=" * 70)

        # --- 1. raw prediction strings -----------------------------------
        print("\n--- raw prediction strings (top 10 exact values) ---")
        print(pred["prediction"].value_counts(dropna=False).head(10).to_string())

        # --- 2. parse failures -------------------------------------------
        pred["pred"] = pred["prediction"].apply(parse_answer)
        n_fail = int(pred["pred"].isna().sum())
        print(f"\n--- parse failures: {n_fail} / {len(pred)} "
              f"({n_fail / len(pred):.1%}) ---")
        if n_fail:
            print("unparseable raw outputs:")
            print(pred.loc[pred["pred"].isna(), "prediction"]
                  .value_counts().head(10).to_string())

        # --- 3. join integrity -------------------------------------------
        merged = pred.merge(bench_keys, on=["image_name", "attribute"],
                            how="left", indicator=True)
        unmatched = int((merged["_merge"] == "left_only").sum())
        print("\n--- join integrity ---")
        print("prediction rows :", len(pred))
        print("merged rows     :", len(merged),
              "  <-- must equal prediction rows")
        print("unmatched rows  :", unmatched,
              "  <-- must be 0")

        # --- 4. per-attribute yes-rate + recomputed metrics ---------------
        print("\n--- per-attribute diagnostics ---")
        print("yes_rate near 0.0 or 1.0 = degenerate (model always answers "
              "one way)")
        print("TP/FP/FN/TN show how few samples each metric actually rests on\n")

        rows = []
        for attr, g in merged.groupby("attribute"):
            valid = g.dropna(subset=["pred", "gt"])
            tp = int(((valid["pred"] == 1) & (valid["gt"] == 1)).sum())
            fp = int(((valid["pred"] == 1) & (valid["gt"] == 0)).sum())
            fn = int(((valid["pred"] == 0) & (valid["gt"] == 1)).sum())
            tn = int(((valid["pred"] == 0) & (valid["gt"] == 0)).sum())

            n_pred_yes = tp + fp
            precision = tp / n_pred_yes if n_pred_yes else float("nan")
            recall = tp / (tp + fn) if (tp + fn) else float("nan")
            f1 = (2 * precision * recall / (precision + recall)
                  if precision and recall and (precision + recall) > 0
                  else float("nan"))
            acc = (tp + tn) / len(valid) if len(valid) else float("nan")

            rows.append({
                "attribute": attr,
                "n": len(g),
                "parsed": len(valid),
                "yes_rate": n_pred_yes / len(valid) if len(valid) else float("nan"),
                "TP": tp, "FP": fp, "FN": fn, "TN": tn,
                "Acc": acc, "P": precision, "R": recall, "F1": f1,
            })

        df = pd.DataFrame(rows).sort_values("yes_rate")
        print(df.to_string(index=False,
                           float_format=lambda v: f"{v:.3f}"))

        # --- 5. flag the specific pathologies ----------------------------
        print("\n--- FLAGS ---")
        flagged = False
        for _, r in df.iterrows():
            msgs = []
            if r["TP"] + r["FP"] <= 5:
                msgs.append(f"precision rests on only {int(r['TP'] + r['FP'])} "
                            f"predicted positives")
            if r["yes_rate"] >= 0.95:
                msgs.append(f"model answers Yes {r['yes_rate']:.0%} of the time")
            if r["yes_rate"] <= 0.05:
                msgs.append(f"model answers Yes only {r['yes_rate']:.0%} "
                            f"of the time")
            if r["parsed"] < r["n"]:
                msgs.append(f"{int(r['n'] - r['parsed'])} samples dropped "
                            f"(unparsed or unmatched)")
            if msgs:
                flagged = True
                print(f"  {r['attribute']:<20} " + "; ".join(msgs))
        if not flagged:
            print("  none")


if __name__ == "__main__":
    main()
