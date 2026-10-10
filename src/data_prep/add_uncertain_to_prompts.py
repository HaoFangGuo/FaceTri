"""
add_uncertain_to_prompts.py — Rewrite all prompts in benchmark_final.csv
from Yes/No to Yes/No/Uncertain format, in place.

Input:  benchmark_final.csv (12 attributes x 200 rows, Yes/No prompts)
Output: benchmark_final.csv (overwritten, same rows, Yes/No/Uncertain prompts)
A backup of the original is saved automatically.
"""
import pandas as pd
import shutil

SRC = "data/benchmark/benchmark_final.csv"
BACKUP = "archive/binary_results/benchmark_final_binary.csv"


def add_uncertain_option(prompt):
    old_tail = "Answer with only one word: Yes or No."
    new_tail = (
        "Answer with only one word: Yes, No, or Uncertain.\n"
        "If the image does not provide enough visual evidence to "
        "determine this attribute, answer Uncertain."
    )
    if old_tail not in prompt:
        raise ValueError(f"Expected tail not found in prompt: {prompt!r}")
    return prompt.replace(old_tail, new_tail)


def main():
    shutil.copy(SRC, BACKUP)
    print(f"Backed up original to {BACKUP}")

    df = pd.read_csv(SRC)
    df["prompt"] = df["prompt"].apply(add_uncertain_option)
    df.to_csv(SRC, index=False)

    print(f"Rewrote {len(df)} rows in {SRC}")
    print(df.groupby("attribute").size())
    print("\nSample prompt:")
    print(df.iloc[0]["prompt"])


if __name__ == "__main__":
    main()

