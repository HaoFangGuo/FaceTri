"""
Draw the case-analysis sample from the four cells produced by
stage0_kappa_and_cells.py.

Sampling is stratified by attribute within each cell, because the raw
cells are heavily skewed (cell A is dominated by Crows_Feet). Stratifying
lets the resulting failure-mode taxonomy describe all three attributes
rather than just the dominant one.

Cell A can be drawn from either the strict definition (all three
non-abstaining models wrong) or the relaxed one (at least two wrong).
The relaxed variant is the default because the strict cell is too small
to support a per-attribute breakdown.

Outputs one CSV per cell plus a combined file with a stable case_id, so
that re-inference results and human codes can be joined back later.
"""

import argparse
import os

import pandas as pd

CELL_FILES = {
    "A": "results/cells/cell_A.csv",
    "A_relaxed": "results/cells/cell_A_relaxed.csv",
    "B": "results/cells/cell_B.csv",
    "C": "results/cells/cell_C.csv",
    "D": "results/cells/cell_D.csv",
    "E": "results/cells/cell_E.csv",
}

# Which model's output is the subject of coding for each cell.
# Cells B and C are about abstention, so only the abstaining model matters.
# Cells A, D and E are about definite errors and correct answers.
CELL_SUBJECT_MODELS = {
    "A": ["qwen", "molmo2o", "minicpm"],
    "A_relaxed": ["qwen", "molmo2o", "minicpm"],
    "B": ["internvl3"],
    "C": ["internvl3"],
    "D": ["qwen", "molmo2o", "minicpm"],
    "E": ["qwen", "molmo2o", "minicpm"],
}

OUTPUT_DIR = "case_samples"

# Per-cell sample sizes, tuned to keep the manual coding workload as small
# as the analysis allows. None means take every row in the cell.
#
# Cells B and C carry the central claim and are compared against each other,
# so they get equal and larger samples. Cell D is a control and only needs to
# be large enough to establish a baseline rate. Cell E is the weakest
# evidence tier and stays small.
CELL_DEFAULT_N = {
    "A": None,
    "A_relaxed": 40,
    "B": 40,
    "C": 40,
    "D": 25,
    "E": 20,
}


def stratified_sample(df, n_total, seed):
    """Sample n_total rows, spread as evenly as possible across attributes."""
    attributes = sorted(df["attribute"].unique())
    if not attributes:
        return df.head(0)

    base = n_total // len(attributes)
    remainder = n_total % len(attributes)

    parts = []
    shortfall = 0
    for i, attribute in enumerate(attributes):
        quota = base + (1 if i < remainder else 0)
        pool = df[df["attribute"] == attribute]
        take = min(quota, len(pool))
        shortfall += quota - take
        parts.append(pool.sample(n=take, random_state=seed))

    sampled = pd.concat(parts) if parts else df.head(0)

    # Backfill from whatever is left if some attribute could not meet quota
    if shortfall > 0:
        leftover = df.drop(sampled.index)
        if len(leftover) > 0:
            sampled = pd.concat([sampled, leftover.sample(n=min(shortfall, len(leftover)), random_state=seed)])

    return sampled.sort_values(["attribute", "image_name"]).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Sample cases from the analysis cells")
    parser.add_argument(
        "--n-per-cell",
        type=int,
        default=None,
        help="Override the per-cell default sample size for every cell",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--cells", default="A,B,C,D,E", help="Comma-separated cells to sample")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    wanted = [c.strip() for c in args.cells.split(",")]

    combined = []
    for cell in wanted:
        path = CELL_FILES.get(cell)
        if path is None:
            print(f"[SKIP] {cell}: unknown cell")
            continue
        if not os.path.exists(path):
            print(f"[SKIP] {cell}: {path} not found")
            continue

        df = pd.read_csv(path)
        n_target = args.n_per_cell if args.n_per_cell is not None else CELL_DEFAULT_N.get(cell)

        if n_target is None or n_target >= len(df):
            sampled = df.sort_values(["attribute", "image_name"]).reset_index(drop=True)
            note = "all rows"
        else:
            sampled = stratified_sample(df, n_target, args.seed)
            note = f"stratified from {len(df)}"

        sampled.insert(0, "cell", cell)
        sampled.insert(0, "case_id", [f"{cell}{i + 1:03d}" for i in range(len(sampled))])
        sampled["subject_models"] = ";".join(CELL_SUBJECT_MODELS[cell])

        out_path = os.path.join(OUTPUT_DIR, f"sample_cell_{cell}.csv")
        sampled.to_csv(out_path, index=False)
        combined.append(sampled)

        by_attr = sampled["attribute"].value_counts().to_dict()
        print(f"{cell}: {len(sampled)} cases ({note})  {by_attr}")

    if combined:
        all_cases = pd.concat(combined, ignore_index=True)
        all_path = os.path.join(OUTPUT_DIR, "results/cells/case_samples/all_cases.csv")
        all_cases.to_csv(all_path, index=False)

        n_cases = len(all_cases)
        print(f"\nTotal cases to code: {n_cases}")
        print(f"Unique images to inspect: {all_cases['image_name'].nunique()}")
        print(f"Estimated coding time at 2-3 min per case: {n_cases * 2 / 60:.1f}-{n_cases * 3 / 60:.1f} hours")
        print(f"Double-coding subset (20%): {round(n_cases * 0.2)} cases")
        print(f"\nWritten to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
