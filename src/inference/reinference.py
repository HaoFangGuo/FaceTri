"""
reinference_reason_then_answer.py — re-run InternVL3-8B on sampled cases,
asking the model to state a reason alongside its answer.

Usage (on a GPU server):
    python reinference_reason_then_answer.py \
        --cases case_samples/sample_cell_B.csv,case_samples/sample_cell_C.csv \
        --gt benchmark_final.csv \
        --img-dir benchmark_images \
        --out reinference_internvl3_reason.csv

The original benchmark prompt constrains the model to a single word, which
leaves no room for a justification. This script rebuilds the prompt from the
same question but requests a fixed two-line format:

    Answer: <Yes/No/Uncertain>
    Reason: <one sentence>

Because the prompt is not identical to the original run, the answer itself
may change. That is measured rather than ignored: the script reports how
often the new answer differs from the stored prediction, which is a result
in its own right (whether demanding a justification suppresses abstention).

Output columns:
    case_id, cell, image_name, attribute, majority_label,
    original_prediction, new_answer, reason, parse_ok, raw_output

Supports resume: re-running skips case_ids already present in the output.
"""

import argparse
import os
import re
import time

import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

MODEL_ID = "OpenGVLab/InternVL3-8B"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Longer than the original run (15) because the response now carries a
# one-sentence justification in addition to the answer token.
MAX_NEW_TOKENS = 80

CANONICAL = {"yes": "Yes", "no": "No", "uncertain": "Uncertain"}

REASON_INSTRUCTION = (
    "Respond in exactly this format, with no extra text:\n"
    "Answer: <Yes, No, or Uncertain>\n"
    "Reason: <one sentence describing the visual evidence for your answer>"
)


def build_transform(input_size=448):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, use_fast=False)
    model = AutoModel.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=False,      # Turing GPU - must be False
        trust_remote_code=True,
    ).eval().cuda()
    return model, tokenizer


def build_reason_prompt(original_prompt):
    """Rebuild the benchmark prompt so it asks for an answer and a reason.

    The original prompt ends with a single-word constraint that contradicts
    a request for justification, so the question line is extracted and the
    instructions are rewritten around it. If the question cannot be located,
    the single-word constraint is stripped and the format block appended.
    """
    match = re.search(r"^Question:\s*(.+?)\s*$", original_prompt, flags=re.MULTILINE)
    if match:
        question = match.group(1)
        return (
            "Carefully examine the facial image.\n"
            f"Question: {question}\n"
            "If the image does not provide enough visual evidence to determine "
            "this attribute, answer Uncertain.\n\n"
            f"{REASON_INSTRUCTION}"
        )

    stripped = re.sub(
        r"Answer with only one word[^\n]*\n?", "", original_prompt, flags=re.IGNORECASE
    ).strip()
    return f"{stripped}\n\n{REASON_INSTRUCTION}"


def parse_response(raw):
    """Split a raw response into a canonical answer and a reason sentence."""
    if not raw:
        return None, "", False

    answer_match = re.search(r"Answer\s*:\s*([A-Za-z]+)", raw)
    reason_match = re.search(r"Reason\s*:\s*(.+)", raw, flags=re.DOTALL)

    answer = CANONICAL.get(answer_match.group(1).lower()) if answer_match else None
    reason = " ".join(reason_match.group(1).split()) if reason_match else ""

    # Fall back to a bare leading answer token when the format was ignored
    if answer is None:
        first = re.match(r"\s*([A-Za-z]+)", raw)
        if first:
            answer = CANONICAL.get(first.group(1).lower())

    return answer, reason, bool(answer_match and reason_match)


@torch.inference_mode()
def ask(model, tokenizer, image_path, prompt):
    transform = build_transform()
    image = Image.open(image_path)
    pixel_values = transform(image).unsqueeze(0).to(torch.bfloat16).cuda()

    question = f"<image>\n{prompt}"
    generation_config = dict(max_new_tokens=MAX_NEW_TOKENS, do_sample=False)

    response = model.chat(tokenizer, pixel_values, question, generation_config)
    return response.strip()


def load_cases(case_paths, gt_path, model_key):
    """Load sampled cases and attach the benchmark prompt for each row."""
    frames = []
    for path in case_paths:
        if not os.path.exists(path):
            print(f"[SKIP] case file not found: {path}")
            continue
        frames.append(pd.read_csv(path))
    if not frames:
        raise SystemExit("No case files could be loaded")

    cases = pd.concat(frames, ignore_index=True)

    bench = pd.read_csv(gt_path)
    cases = cases.merge(
        bench[["image_name", "attribute", "prompt"]],
        on=["image_name", "attribute"],
        how="left",
    )

    missing_prompt = cases["prompt"].isna().sum()
    if missing_prompt:
        print(f"[WARN] {missing_prompt} cases have no prompt in {gt_path} and will be skipped")
        cases = cases[cases["prompt"].notna()]

    pred_col = f"pred_{model_key}"
    if pred_col not in cases.columns:
        print(f"[WARN] {pred_col} not in case files; original_prediction will be blank")
        cases[pred_col] = None

    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--cases",
        default="case_samples/sample_cell_B.csv,case_samples/sample_cell_C.csv",
        help="Comma-separated sampled-case CSV files",
    )
    ap.add_argument("--gt", default="data/benchmark/benchmark_final.csv")
    ap.add_argument("--img-dir", default="benchmark_images")
    ap.add_argument("--out", default="results/reinference/reinference_internvl3_reason.csv")
    ap.add_argument("--model-key", default="internvl3", help="Which pred_* column holds the original answer")
    ap.add_argument("--limit", type=int, default=None, help="Process at most this many cases")
    ap.add_argument("--dry-run", action="store_true", help="Print the rebuilt prompt and exit")
    args = ap.parse_args()

    case_paths = [p.strip() for p in args.cases.split(",")]
    cases = load_cases(case_paths, args.gt, args.model_key)
    pred_col = f"pred_{args.model_key}"

    if args.dry_run:
        sample = cases.iloc[0]
        print("--- ORIGINAL PROMPT ---")
        print(sample["prompt"])
        print("\n--- REBUILT PROMPT ---")
        print(build_reason_prompt(sample["prompt"]))
        print(f"\nTotal cases: {len(cases)}")
        print(cases.groupby(["cell", "attribute"]).size().to_string())
        return

    # Resume: skip case_ids already written
    done = set()
    if os.path.exists(args.out):
        prev = pd.read_csv(args.out)
        done = set(prev["case_id"])
        print(f"Found {len(done)} existing results, resuming remaining cases")

    todo = cases[~cases["case_id"].isin(done)]
    if args.limit:
        todo = todo.head(args.limit)
    if todo.empty:
        print("All cases already completed")
        return

    print(f"Loading model {MODEL_ID} ...")
    model, tokenizer = load_model()
    print(f"[MEM] Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    write_header = not os.path.exists(args.out)
    t0 = time.time()
    results = []
    n_todo = len(todo)

    for i, (_, row) in enumerate(todo.iterrows(), 1):
        img_path = os.path.join(args.img_dir, str(row["image_name"]))
        raw = ""
        if not os.path.exists(img_path):
            print(f"Skipping, image not found: {img_path}")
        else:
            try:
                raw = ask(model, tokenizer, img_path, build_reason_prompt(row["prompt"]))
            except Exception as e:
                print(f"Error on {row['case_id']} ({row['image_name']}/{row['attribute']}): {e}")

        answer, reason, parse_ok = parse_response(raw)

        results.append(
            {
                "case_id": row["case_id"],
                "cell": row.get("cell", ""),
                "image_name": row["image_name"],
                "attribute": row["attribute"],
                "majority_label": row.get("majority_label", row.get("label", "")),
                "original_prediction": row.get(pred_col, ""),
                "new_answer": answer,
                "reason": reason,
                "parse_ok": parse_ok,
                "raw_output": " ".join(str(raw).split()),
            }
        )

        # Flush every 20 cases so an interruption does not lose work
        if len(results) >= 20:
            pd.DataFrame(results).to_csv(args.out, mode="a", header=write_header, index=False)
            write_header = False
            results = []

        if i % 20 == 0 or i == n_todo:
            speed = i / (time.time() - t0)
            eta = (n_todo - i) / speed if speed > 0 else 0
            print(f"[{i}/{n_todo}] {speed:.2f} it/s, ETA {eta/60:.1f} min")

    if results:
        pd.DataFrame(results).to_csv(args.out, mode="a", header=write_header, index=False)

    report(args.out)


def report(out_path):
    """Summarise parse success and how often the answer changed."""
    df = pd.read_csv(out_path)
    print(f"\nDone. {len(df)} cases saved to {out_path}")

    parsed = df["parse_ok"].sum()
    print(f"  Format followed: {parsed}/{len(df)} ({parsed / max(len(df), 1):.1%})")

    comparable = df[df["new_answer"].notna() & df["original_prediction"].notna()]
    if len(comparable) == 0:
        return

    changed = (comparable["new_answer"] != comparable["original_prediction"]).sum()
    print(f"  Answer changed vs original run: {changed}/{len(comparable)} ({changed / len(comparable):.1%})")

    still_abstain = (comparable["new_answer"] == "Uncertain").sum()
    was_abstain = (comparable["original_prediction"] == "Uncertain").sum()
    print(f"  Abstained originally: {was_abstain}  |  abstains now: {still_abstain}")

    print("\n  New answer distribution by cell:")
    print(df.groupby(["cell", "new_answer"]).size().to_string())


if __name__ == "__main__":
    main()
