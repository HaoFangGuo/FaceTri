"""
abstention_entropy_analysis.py - Test whether model abstention aligns with
human annotator disagreement.

For each wrinkle attribute, this compares the annotation entropy of items the
model abstained on against items it answered. If a model's uncertainty is well
calibrated to genuine visual ambiguity, abstained items should carry higher
entropy, since entropy here measures how much three independent annotators
disagreed on that item.

Reports, per model and attribute:
  - mean annotation entropy for abstained vs answered items
  - point-biserial correlation between abstention and entropy
  - Mann-Whitney U test (no normality assumption)

Requires:
    wrinkle_rows_meta.csv        - per-item annotation statistics
    prediction_{model}_uncertain.csv

Usage:
    python abstention_entropy_analysis.py
"""

import re
import pandas as pd
from scipy import stats

META_FILE = "data/wrinkle_rows_meta.csv"
OUT_FILE = "results/case_studies/abstention_entropy_analysis.csv"

MODEL_FILES = {
    "qwen": "results/predictions/prediction_qwen_uncertain.csv",
    "internvl3": "results/predictions/prediction_internvl3_uncertain.csv",
    "molmo2o": "results/predictions/prediction_molmo2o_uncertain.csv",
}

MIN_GROUP_SIZE = 10


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


def main():
    meta = pd.read_csv(META_FILE)
    attributes = sorted(meta["attribute"].unique())
    rows = []

    for model, path in MODEL_FILES.items():
        try:
            pred = pd.read_csv(path)
        except FileNotFoundError:
            print(f"[WARN] {path} not found, skipping {model}\n")
            continue

        pred["parsed"] = pred["prediction"].apply(parse_answer)
        df = meta.merge(pred[["image_name", "attribute", "parsed"]],
                        on=["image_name", "attribute"], how="left")

        n_missing = df["parsed"].isna().sum()
        if n_missing:
            print(f"[WARN] {model}: {n_missing} item(s) unmatched or unparseable, excluded")
            df = df.dropna(subset=["parsed"])

        df["abstained"] = (df["parsed"] == "uncertain").astype(int)

        print(f"===== {model} =====")

        for attr in attributes + ["ALL"]:
            sub = df if attr == "ALL" else df[df["attribute"] == attr]
            abstained = sub[sub["abstained"] == 1]["entropy"]
            answered = sub[sub["abstained"] == 0]["entropy"]

            row = {
                "model": model,
                "attribute": attr,
                "abstention_rate": round(sub["abstained"].mean(), 4),
                "n_abstained": len(abstained),
                "n_answered": len(answered),
                "entropy_abstained": abstained.mean() if len(abstained) else float("nan"),
                "entropy_answered": answered.mean() if len(answered) else float("nan"),
            }
            row["entropy_gap"] = row["entropy_abstained"] - row["entropy_answered"]

            if len(abstained) < MIN_GROUP_SIZE or len(answered) < MIN_GROUP_SIZE:
                print(f"{attr:20s} abstention={row['abstention_rate']:.3f}  "
                      f"(n_abstained={len(abstained)}, n_answered={len(answered)}) "
                      f"- group too small for testing")
                row.update({"r_pb": None, "p_pb": None, "p_mannwhitney": None})
                rows.append(row)
                continue

            r_pb, p_pb = stats.pointbiserialr(sub["abstained"], sub["entropy"])
            _, p_mw = stats.mannwhitneyu(abstained, answered, alternative="greater")

            row.update({"r_pb": round(r_pb, 4), "p_pb": p_pb, "p_mannwhitney": p_mw})
            rows.append(row)

            print(f"{attr:20s} abstention={row['abstention_rate']:.3f}  "
                  f"entropy: abstained={row['entropy_abstained']:.3f} "
                  f"vs answered={row['entropy_answered']:.3f} "
                  f"(gap={row['entropy_gap']:+.3f})  "
                  f"r_pb={r_pb:+.3f} (p={p_pb:.2e})  "
                  f"MW p={p_mw:.2e}")

        print()

    if not rows:
        raise SystemExit("[ERROR] No model results available")

    out = pd.DataFrame(rows)
    for col in ("entropy_abstained", "entropy_answered", "entropy_gap"):
        out[col] = out[col].round(4)
    out.to_csv(OUT_FILE, index=False)
    print(f"Results saved to {OUT_FILE}")

    print("\nInterpretation: a positive entropy gap and positive r_pb mean the model")
    print("abstains more often on items where human annotators disagreed more,")
    print("i.e. its uncertainty tracks genuine annotation ambiguity.")


if __name__ == "__main__":
    main()
