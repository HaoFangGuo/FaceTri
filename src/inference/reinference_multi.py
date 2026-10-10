"""
reinference_multi.py — re-run Qwen2.5-VL, Molmo2-O or MiniCPM-V on sampled
cases, asking for an answer plus a one-sentence reason.

Usage (on a GPU server):
    python reinference_multi.py --model qwen \
        --cases case_samples/sample_cell_A.csv,case_samples/sample_cell_D.csv,case_samples/sample_cell_E.csv \
        --gt benchmark_final.csv \
        --img-dir benchmark_images \
        --out reinference_qwen_reason.csv

Model-specific loading and calling conventions are kept exactly as in the
original inference_*.py scripts, including the MiniCPM bfloat16 + sdpa +
sampling=False configuration that avoids NaN/Inf logits on this hardware.
Model libraries are imported lazily so a missing dependency for one model
does not block the others.

The prompt is rebuilt to request a fixed two-line response:

    Answer: <Yes/No/Uncertain>
    Reason: <one sentence>

Because this prompt differs from the original benchmark run, the answer may
change. That is measured, not ignored: the script reports how often the new
answer differs from the stored prediction, which is itself a result about
prompt sensitivity.

Output columns:
    case_id, cell, image_name, attribute, ground_truth,
    original_prediction, new_answer, reason, parse_ok, raw_output

Supports resume: re-running skips case_ids already present in the output.
"""

import argparse
import os
import re
import time

import pandas as pd

MAX_NEW_TOKENS = 80

MODEL_IDS = {
    "qwen": "Qwen/Qwen2.5-VL-7B-Instruct",
    "molmo2o": "allenai/Molmo2-O-7B",
    "minicpm": "openbmb/MiniCPM-V-4",
    "internvl3": "OpenGVLab/InternVL3-8B",
}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

CANONICAL = {"yes": "Yes", "no": "No", "uncertain": "Uncertain"}

REASON_INSTRUCTION = (
    "Respond in exactly this format, with no extra text:\n"
    "Answer: <Yes, No, or Uncertain>\n"
    "Reason: <one sentence describing the visual evidence for your answer>"
)

# Reason-first variant. The default ordering commits the model to an answer
# before it writes any justification, which makes the justification a
# post-hoc statement rather than a record of the decision. This variant
# reverses the order so the stated evidence precedes the answer, and exists
# as a control: if the reasons stay circular under both orderings, ordering
# is ruled out as the cause.
REASON_FIRST_INSTRUCTION = (
    "Respond in exactly this format, with no extra text:\n"
    "Reason: <one sentence describing the visual evidence you observe>\n"
    "Answer: <Yes, No, or Uncertain>"
)


# ----------------------------------------------------------------------
# Prompt handling
# ----------------------------------------------------------------------

def build_reason_prompt(original_prompt, reason_first=False):
    """Rebuild the benchmark prompt so it asks for an answer and a reason.

    The original prompt ends with a single-word constraint that contradicts a
    request for justification, so the question line is extracted and the
    instructions rewritten around it. `reason_first` swaps the order of the
    two output lines.
    """
    instruction = REASON_FIRST_INSTRUCTION if reason_first else REASON_INSTRUCTION

    match = re.search(r"^Question:\s*(.+?)\s*$", original_prompt, flags=re.MULTILINE)
    if match:
        return (
            "Carefully examine the facial image.\n"
            f"Question: {match.group(1)}\n"
            "If the image does not provide enough visual evidence to determine "
            "this attribute, answer Uncertain.\n\n"
            f"{instruction}"
        )

    stripped = re.sub(
        r"Answer with only one word[^\n]*\n?", "", original_prompt, flags=re.IGNORECASE
    ).strip()
    return f"{stripped}\n\n{instruction}"


def parse_response(raw):
    """Split a raw response into a canonical answer and a reason sentence."""
    if not raw:
        return None, "", False

    answer_match = re.search(r"Answer\s*:\s*([A-Za-z]+)", raw)
    # Stop at a following "Answer:" so the reason-first ordering does not
    # swallow the answer line into the reason text.
    reason_match = re.search(r"Reason\s*:\s*(.+?)(?=\n\s*Answer\s*:|$)", raw, flags=re.DOTALL)

    answer = CANONICAL.get(answer_match.group(1).lower()) if answer_match else None
    reason = " ".join(reason_match.group(1).split()) if reason_match else ""

    if answer is None:
        first = re.match(r"\s*([A-Za-z]+)", raw)
        if first:
            answer = CANONICAL.get(first.group(1).lower())

    return answer, reason, bool(answer_match and reason_match)


# ----------------------------------------------------------------------
# Per-model loaders and callers
# ----------------------------------------------------------------------

def load_qwen():
    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_IDS["qwen"], torch_dtype=torch.bfloat16, device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(MODEL_IDS["qwen"])
    return model, processor


def ask_qwen(model, processor, image_path, prompt):
    import torch
    from qwen_vl_utils import process_vision_info

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": prompt},
        ],
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    trimmed = generated[:, inputs.input_ids.shape[1]:]
    return processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()


def load_molmo2o():
    from transformers import AutoProcessor, AutoModelForImageTextToText

    processor = AutoProcessor.from_pretrained(
        MODEL_IDS["molmo2o"], trust_remote_code=True, dtype="auto", device_map="auto"
    )
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_IDS["molmo2o"], trust_remote_code=True, dtype="auto", device_map="auto"
    )
    return model, processor


def ask_molmo2o(model, processor, image_path, prompt):
    import torch
    from PIL import Image

    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image", "image": Image.open(image_path)},
        ],
    }]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_tensors="pt", return_dict=True,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
    generated_tokens = generated_ids[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(generated_tokens, skip_special_tokens=True)[0].strip()


def load_minicpm():
    import torch
    from transformers import AutoModel, AutoTokenizer

    # bfloat16 + sdpa. float16 produced NaN/Inf logits on this hardware, which
    # either crashed torch.multinomial or silently emitted a garbage token
    # under greedy decoding. bfloat16's wider dynamic range fixes this. Volta
    # has no native bf16 support so PyTorch falls back to a slower software
    # path, but it is numerically correct. MiniCPM-V-4 supports only "sdpa"
    # or "flash_attention_2" -- eager is NOT an option for this model.
    model = AutoModel.from_pretrained(
        MODEL_IDS["minicpm"],
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model = model.eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_IDS["minicpm"], trust_remote_code=True)
    return model, tokenizer


def ask_minicpm(model, tokenizer, image_path, prompt):
    import torch
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    msgs = [{"role": "user", "content": [image, prompt]}]
    # MiniCPM-V-4 expects `image` passed explicitly. sampling=False gives
    # greedy decoding; chat() samples by default, and sampling on this
    # hardware can hit NaN/Inf logits and crash torch.multinomial.
    with torch.inference_mode():
        answer = model.chat(msgs=msgs, image=image, tokenizer=tokenizer, sampling=False)
    return answer.strip()


def load_internvl3():
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_IDS["internvl3"], trust_remote_code=True, use_fast=False
    )
    model = AutoModel.from_pretrained(
        MODEL_IDS["internvl3"],
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=False,      # Turing GPU - must be False
        trust_remote_code=True,
    ).eval().cuda()
    return model, tokenizer


def ask_internvl3(model, tokenizer, image_path, prompt):
    import torch
    import torchvision.transforms as T
    from PIL import Image
    from torchvision.transforms.functional import InterpolationMode

    transform = T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    image = Image.open(image_path)
    pixel_values = transform(image).unsqueeze(0).to(torch.bfloat16).cuda()

    question = f"<image>\n{prompt}"
    generation_config = dict(max_new_tokens=MAX_NEW_TOKENS, do_sample=False)

    with torch.inference_mode():
        response = model.chat(tokenizer, pixel_values, question, generation_config)
    return response.strip()


LOADERS = {
    "qwen": load_qwen,
    "molmo2o": load_molmo2o,
    "minicpm": load_minicpm,
    "internvl3": load_internvl3,
}
ASKERS = {
    "qwen": ask_qwen,
    "molmo2o": ask_molmo2o,
    "minicpm": ask_minicpm,
    "internvl3": ask_internvl3,
}


# ----------------------------------------------------------------------
# Case loading
# ----------------------------------------------------------------------

def load_cases(case_paths, gt_path, model_key):
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
        on=["image_name", "attribute"], how="left",
    )

    missing = cases["prompt"].isna().sum()
    if missing:
        print(f"[WARN] {missing} cases have no prompt in {gt_path} and will be skipped")
        cases = cases[cases["prompt"].notna()]

    # Cells A and D carry the re-annotated majority label; cell E keeps the
    # original benchmark label because it had no three-annotator round.
    if "majority_label" in cases.columns:
        cases["ground_truth"] = cases["majority_label"].fillna(cases.get("label"))
    else:
        cases["ground_truth"] = cases.get("label")

    pred_col = f"pred_{model_key}"
    if pred_col not in cases.columns:
        print(f"[WARN] {pred_col} not in case files; original_prediction will be blank")
        cases[pred_col] = None

    return cases


# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODEL_IDS), help="Which model to re-run")
    ap.add_argument(
        "--cases",
        default="case_samples/sample_cell_A.csv,case_samples/sample_cell_D.csv,case_samples/sample_cell_E.csv",
        help="Comma-separated sampled-case CSV files",
    )
    ap.add_argument("--gt", default="data/benchmark/benchmark_final.csv")
    ap.add_argument("--img-dir", default="benchmark_images")
    ap.add_argument("--out", default=None, help="Defaults to reinference_<model>_reason.csv")
    ap.add_argument("--limit", type=int, default=None, help="Process at most this many cases")
    ap.add_argument(
        "--reason-first",
        action="store_true",
        help="Ask for the reason before the answer, as a control for post-hoc rationalisation",
    )
    ap.add_argument("--dry-run", action="store_true", help="Print the rebuilt prompt and exit")
    args = ap.parse_args()

    suffix = "_reasonfirst" if args.reason_first else ""
    out_path = args.out or f"reinference_{args.model}{suffix}_reason.csv"
    case_paths = [p.strip() for p in args.cases.split(",")]
    cases = load_cases(case_paths, args.gt, args.model)
    pred_col = f"pred_{args.model}"

    if args.dry_run:
        sample = cases.iloc[0]
        print("--- ORIGINAL PROMPT ---")
        print(sample["prompt"])
        print("\n--- REBUILT PROMPT ---")
        print(build_reason_prompt(sample["prompt"], args.reason_first))
        print(f"\nModel: {args.model} ({MODEL_IDS[args.model]})")
        print(f"Ordering: {'reason first' if args.reason_first else 'answer first'}")
        print(f"Output: {out_path}")
        print(f"Total cases: {len(cases)}")
        print(cases.groupby(["cell", "attribute"]).size().to_string())
        return

    done = set()
    if os.path.exists(out_path):
        prev = pd.read_csv(out_path)
        done = set(prev["case_id"])
        print(f"Found {len(done)} existing results, resuming remaining cases")

    todo = cases[~cases["case_id"].isin(done)]
    if args.limit:
        todo = todo.head(args.limit)
    if todo.empty:
        print("All cases already completed")
        return

    print(f"Loading model {MODEL_IDS[args.model]} ...")
    model, helper = LOADERS[args.model]()
    ask = ASKERS[args.model]

    write_header = not os.path.exists(out_path)
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
                raw = ask(model, helper, img_path, build_reason_prompt(row["prompt"], args.reason_first))
            except Exception as e:
                print(f"Error on {row['case_id']} ({row['image_name']}/{row['attribute']}): {e}")

        answer, reason, parse_ok = parse_response(raw)

        results.append({
            "case_id": row["case_id"],
            "cell": row.get("cell", ""),
            "image_name": row["image_name"],
            "attribute": row["attribute"],
            "ground_truth": row.get("ground_truth", ""),
            "original_prediction": row.get(pred_col, ""),
            "new_answer": answer,
            "reason": reason,
            "parse_ok": parse_ok,
            "reason_first": args.reason_first,
            "raw_output": " ".join(str(raw).split()),
        })

        # Flush every 20 cases so an interruption does not lose work
        if len(results) >= 20:
            pd.DataFrame(results).to_csv(out_path, mode="a", header=write_header, index=False)
            write_header = False
            results = []

        if i % 20 == 0 or i == n_todo:
            speed = i / (time.time() - t0)
            eta = (n_todo - i) / speed if speed > 0 else 0
            print(f"[{i}/{n_todo}] {speed:.2f} it/s, ETA {eta/60:.1f} min")

    if results:
        pd.DataFrame(results).to_csv(out_path, mode="a", header=write_header, index=False)

    report(out_path, args.model)


def report(out_path, model_key):
    df = pd.read_csv(out_path)
    print(f"\nDone. {len(df)} cases saved to {out_path}")

    parsed = df["parse_ok"].sum()
    print(f"  Format followed: {parsed}/{len(df)} ({parsed / max(len(df), 1):.1%})")

    comparable = df[df["new_answer"].notna() & df["original_prediction"].notna()]
    if len(comparable) == 0:
        return

    changed = comparable["new_answer"] != comparable["original_prediction"]
    print(f"  Answer changed vs original run: {changed.sum()}/{len(comparable)} ({changed.mean():.1%})")
    print("  Note: reasons are only diagnostic for the cases where the answer did NOT change.")

    unchanged = comparable[~changed]
    print(f"  Usable (answer unchanged): {len(unchanged)}")

    print(f"\n  New answer distribution by cell:")
    print(df.groupby(["cell", "new_answer"]).size().to_string())


if __name__ == "__main__":
    main()
