"""
build_molmo2o_prompt.py — Create a Molmo2-O-specific version of
benchmark_final.csv with the "If the image does not provide enough visual
evidence..." abstention-hint sentence removed from every prompt.

Molmo2-O was found to treat that sentence as a near-mandatory instruction,
causing it to answer "Uncertain" on almost every sample regardless of image
content. All other fields (image_name, attribute, label) are unchanged, only
the prompt column is rewritten.

Input:  benchmark_final.csv
Output: benchmark_final_molmo2o.csv
"""

import pandas as pd

SRC = "data/benchmark/benchmark_final.csv"
OUT = "data/benchmark/benchmark_final_molmo2o.csv"


def strip_abstention_hint(prompt):
    marker = "If the image does not provide enough visual evidence"
    idx = prompt.find(marker)
    if idx == -1:
        raise ValueError(f"Marker not found in prompt: {prompt!r}")
    return prompt[:idx].rstrip()


def main():
    df = pd.read_csv(SRC)
    df["prompt"] = df["prompt"].apply(strip_abstention_hint)
    df.to_csv(OUT, index=False)

    print(f"Wrote {len(df)} rows to {OUT}")
    print("\nSample prompt:")
    print(df.iloc[0]["prompt"])


if __name__ == "__main__":
    main()
