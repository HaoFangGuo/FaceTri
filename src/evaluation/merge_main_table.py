"""
merge_main_table.py — merge the four per-model evaluation CSVs
(eval_qwen.csv, eval_internvl3.csv, eval_molmo2o.csv, eval_minicpm.csv)
into a single 12-attribute main table for the paper.

Each input file is produced by evaluate.py and has columns:
    attribute, N, Coverage, Risk, EffectiveReliability, N_answered,
    Accuracy, Precision, Recall, F1
with a final row where attribute == "OVERALL".

Outputs:
    main_table_long.csv           - one row per (model, attribute), all metrics
    main_table_accuracy_wide.csv  - attribute x model, Accuracy
    main_table_coverage_wide.csv  - attribute x model, Coverage
    main_table_ER_wide.csv        - attribute x model, EffectiveReliability

Also prints the OVERALL row for each model side by side, so the ranking
reversal (best on Accuracy vs best on ER) is visible at a glance.
"""

import os

import pandas as pd

EVAL_FILES = {
    "Qwen2.5-VL-7B": "results/evals/eval_qwen.csv",
    "InternVL3-8B": "results/evals/eval_internvl3.csv",
    # Molmo2-O has two versions: the original three-option prompt (eval_molmo2o.csv)
    # drives it to near-total abstention (Coverage ~0.008), while dropping one hint
    # sentence from the prompt (eval_molmo2o_nohint.csv) brings it back to normal
    # answering behavior comparable to the other three models. The main table uses
    # the nohint version for cross-model comparability; the original version is a
    # separate data point supporting the prompt-sensitivity findings, not part of
    # this table.
    "Molmo2-O-7B": "results/evals/eval_molmo2o_nohint.csv",
    "MiniCPM-V-4": "results/evals/eval_minicpm.csv",
}

METRIC_COLS = ["N", "Coverage", "Risk", "EffectiveReliability", "N_answered",
               "Accuracy", "Precision", "Recall", "F1"]


def load_all():
    frames = []
    for model_name, path in EVAL_FILES.items():
        if not os.path.exists(path):
            print(f"[SKIP] {model_name}: {path} not found")
            continue
        df = pd.read_csv(path)
        missing_cols = [c for c in METRIC_COLS if c not in df.columns]
        if missing_cols:
            print(f"[WARN] {model_name}: missing columns {missing_cols}")
        df.insert(0, "model", model_name)
        frames.append(df)
    if not frames:
        raise SystemExit("No evaluation files could be loaded")
    return pd.concat(frames, ignore_index=True)


def make_wide(long_df, metric):
    per_attr = long_df[long_df["attribute"] != "OVERALL"]
    wide = per_attr.pivot(index="attribute", columns="model", values=metric)
    ordered = [m for m in EVAL_FILES if m in wide.columns]
    wide = wide[ordered]
    wide = wide.round(3)
    return wide


def main():
    long_df = load_all()
    long_df.to_csv("results/tables/main_table_long.csv", index=False)
    print(f"Long-format table: {len(long_df)} rows -> main_table_long.csv\n")

    for metric, out_name in [
        ("Accuracy", "results/tables/main_table_accuracy_wide.csv"),
        ("Coverage", "results/tables/main_table_coverage_wide.csv"),
        ("EffectiveReliability", "results/tables/main_table_ER_wide.csv"),
    ]:
        wide = make_wide(long_df, metric)
        wide.to_csv(out_name)
        print(f"--- {metric} (attribute x model) ---")
        print(wide.to_string())
        print(f"-> {out_name}\n")

    print("=" * 70)
    print("OVERALL row, all models side by side")
    print("=" * 70)
    overall = long_df[long_df["attribute"] == "OVERALL"].set_index("model")
    cols = ["Coverage", "Risk", "EffectiveReliability", "Accuracy", "Precision", "Recall", "F1"]
    cols = [c for c in cols if c in overall.columns]
    print(overall[cols].round(3).to_string())

    if "Accuracy" in overall.columns and "EffectiveReliability" in overall.columns:
        best_acc = overall["Accuracy"].idxmax()
        best_er = overall["EffectiveReliability"].idxmax()
        print(f"\nBest on Accuracy: {best_acc}  |  Best on Effective Reliability: {best_er}")
        if best_acc != best_er:
            print("-> ranking reversal confirmed")


if __name__ == "__main__":
    main()
