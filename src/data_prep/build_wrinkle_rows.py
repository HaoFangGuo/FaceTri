"""
Build 600 Yes/No benchmark rows for the three wrinkle attributes.

Input : wrinkles_annotations_aggregated.csv (majority vote over three annotators)
Output: wrinkle_rows.csv, 100 Yes + 100 No per attribute, two-option prompt format
        matching benchmark.csv. Run add_uncertain_to_prompts.py afterwards to
        produce the three-option version.

Usage:
    python build_wrinkle_rows.py --agg wrinkles_annotations_aggregated.csv --start-id 1801
"""

import argparse
import csv
import os
import random
from collections import defaultdict

ATTRS = ["Forehead_Wrinkles", "Crows_Feet", "Nasolabial_Folds"]

QUESTIONS = {
    "Forehead_Wrinkles": "Does this person have forehead wrinkles?",
    "Crows_Feet": "Does this person have crow's feet?",
    "Nasolabial_Folds": "Does this person have nasolabial folds?",
}

PROMPT_TEMPLATE = (
    "Carefully examine the facial image.\n"
    "Question: {question}\n"
    "Answer with only one word: Yes or No."
)

N_PER_CLASS = 100


def image_index(name):
    """Numeric sort key for image filenames such as 21767.jpg."""
    return int(name.split(".")[0])


def load_aggregated(path):
    """Load the aggregated annotation CSV into {(image_name, attribute): stats}."""
    agg = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            attr = row["attribute"]
            if attr not in ATTRS:
                continue
            label = row["majority_label"]
            if label not in ("Yes", "No"):
                print(f"[WARN] Skipping {row['image_name']} / {attr}: majority_label={label!r}")
                continue
            agg[(row["image_name"], attr)] = {
                "majority_label": label,
                "n_annotators": row["n_annotators"],
                "n_yes": row["n_yes"],
                "n_no": row["n_no"],
                "agreement_rate": row["agreement_rate"],
                "entropy": row["entropy"],
            }
    return agg


def sample_balanced(agg, seed):
    """Sample N_PER_CLASS positives and N_PER_CLASS negatives per attribute."""
    rng = random.Random(seed)
    selected = {}

    for attr in ATTRS:
        pos = sorted([img for (img, a), v in agg.items()
                      if a == attr and v["majority_label"] == "Yes"], key=image_index)
        neg = sorted([img for (img, a), v in agg.items()
                      if a == attr and v["majority_label"] == "No"], key=image_index)
        print(f"\n{attr}: available Yes={len(pos)}, No={len(neg)}")

        if len(pos) < N_PER_CLASS or len(neg) < N_PER_CLASS:
            raise SystemExit(
                f"[ERROR] {attr} cannot reach {N_PER_CLASS}/{N_PER_CLASS} "
                f"(Yes={len(pos)}, No={len(neg)}). Annotate more images first."
            )

        selected[attr] = (
            sorted(rng.sample(pos, N_PER_CLASS), key=image_index)
            + sorted(rng.sample(neg, N_PER_CLASS), key=image_index)
        )
        print(f"  selected {N_PER_CLASS} Yes + {N_PER_CLASS} No")

    return selected


def main():
    parser = argparse.ArgumentParser(
        description="Generate 600 Yes/No wrinkle-attribute rows from aggregated annotations."
    )
    parser.add_argument("--agg", default="data/wrinkles_annotations_aggregated.csv",
                        help="Path to the aggregated annotation CSV")
    parser.add_argument("--start-id", type=int, default=1801, help="First id value")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed")
    parser.add_argument("--out", default="data/wrinkle_rows.csv", help="Output CSV path")
    parser.add_argument("--meta-out", default="data/wrinkle_rows_meta.csv",
                        help="Output path for per-item annotation statistics")
    parser.add_argument("--no-header", action="store_true",
                        help="Omit the header row, for direct concatenation")
    args = parser.parse_args()

    if not os.path.exists(args.agg):
        raise SystemExit(f"[ERROR] File not found: {args.agg}")

    agg = load_aggregated(args.agg)
    print(f"Loaded {len(agg)} annotated items from {args.agg}")

    selected = sample_balanced(agg, args.seed)

    rows = []
    current_id = args.start_id
    for attr in ATTRS:
        for image_name in selected[attr]:
            rows.append({
                "id": current_id,
                "image_name": image_name,
                "attribute": attr,
                "label": agg[(image_name, attr)]["majority_label"],
                "prompt": PROMPT_TEMPLATE.format(question=QUESTIONS[attr]),
            })
            current_id += 1

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "image_name", "attribute", "label", "prompt"])
        if not args.no_header:
            writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {args.out}: {len(rows)} rows, id {args.start_id}-{current_id - 1}")

    with open(args.meta_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "id", "image_name", "attribute", "label",
            "n_annotators", "n_yes", "n_no", "agreement_rate", "entropy",
        ])
        writer.writeheader()
        for row in rows:
            stats = agg[(row["image_name"], row["attribute"])]
            writer.writerow({
                "id": row["id"],
                "image_name": row["image_name"],
                "attribute": row["attribute"],
                "label": row["label"],
                "n_annotators": stats["n_annotators"],
                "n_yes": stats["n_yes"],
                "n_no": stats["n_no"],
                "agreement_rate": stats["agreement_rate"],
                "entropy": stats["entropy"],
            })
    print(f"Wrote {args.meta_out}: annotation statistics for the selected items")

    print("\n=== Annotator agreement on selected items ===")
    by_attr = defaultdict(list)
    for row in rows:
        by_attr[row["attribute"]].append(agg[(row["image_name"], row["attribute"])])
    for attr in ATTRS:
        stats = by_attr[attr]
        n_unanimous = sum(1 for s in stats if float(s["agreement_rate"]) == 1.0)
        mean_entropy = sum(float(s["entropy"]) for s in stats) / len(stats)
        print(f"{attr}: n={len(stats)}, unanimous={n_unanimous} "
              f"({100 * n_unanimous / len(stats):.1f}%), mean entropy={mean_entropy:.3f}")

    print(f"\nSeed: {args.seed}")
    print("Next: append these rows to benchmark.csv, then run add_uncertain_to_prompts.py")


if __name__ == "__main__":
    main()
