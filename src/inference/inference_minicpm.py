import argparse
import os
import time

import pandas as pd
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

MODEL_ID = "openbmb/MiniCPM-V-4"


def load_model():
    # bfloat16 + sdpa. float16 was tried first and produced NaN/Inf logits on
    # this hardware, which either crashed torch.multinomial or (under greedy
    # decoding) silently emitted a single garbage token. bfloat16's wider
    # dynamic range fixes this. Volta has no native bf16 support so PyTorch
    # falls back to a slower software path, but it is numerically correct.
    # Note: MiniCPM-V-4's model card supports only "sdpa" or
    # "flash_attention_2" -- eager is NOT an option for this model.
    model = AutoModel.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model = model.eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    return model, tokenizer


@torch.inference_mode()
def ask(model, tokenizer, image_path, prompt):
    image = Image.open(image_path).convert("RGB")
    msgs = [{"role": "user", "content": [image, prompt]}]
    # MiniCPM-V-4 expects `image` passed explicitly (unlike MiniCPM-V-2.6,
    # where image=None was correct because the image only lived in msgs).
    # sampling=False -> greedy decoding. chat() samples by default, and on
    # Volta (V100S) fp16 logits can contain NaN/Inf, which crashes
    # torch.multinomial. Greedy decoding skips that probability check, but it
    # does NOT fix the underlying NaN -- verify outputs are sensible before
    # trusting a full run.
    answer = model.chat(msgs=msgs, image=image, tokenizer=tokenizer, sampling=False)
    return answer.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default="data/benchmark/benchmark.csv")
    ap.add_argument("--img-dir", default="CelebA-HQ-img")
    ap.add_argument("--out", default="results/predictions/prediction.csv")
    ap.add_argument("--limit", type=int, default=None,
                     help="Optional: cap the number of samples, for a quick compatibility smoke test.")
    args = ap.parse_args()

    bench = pd.read_csv(args.gt)
    if args.limit is not None:
        bench = bench.head(args.limit)
        print(f"Limit set: running on {len(bench)} samples only")

    # Resume support: skip samples already present in the output file
    done = set()
    if os.path.exists(args.out):
        prev = pd.read_csv(args.out)
        done = set(zip(prev["image_name"], prev["attribute"]))
        print(f"Found {len(done)} existing results, resuming remaining samples")

    todo = bench[~bench.apply(lambda r: (r["image_name"], r["attribute"]) in done, axis=1)]
    if todo.empty:
        print("All samples already completed")
        return

    print(f"Loading model {MODEL_ID} ...")
    model, tokenizer = load_model()

    write_header = not os.path.exists(args.out)
    t0 = time.time()
    results = []

    for i, (_, row) in enumerate(todo.iterrows(), 1):
        img_path = os.path.join(args.img_dir, str(row["image_name"]))
        if not os.path.exists(img_path):
            print(f"Skipping, image not found: {img_path}")
            answer = ""
        else:
            try:
                answer = ask(model, tokenizer, img_path, row["prompt"])
            except Exception as e:
                print(f"Error on {row['image_name']}/{row['attribute']}: {e}")
                answer = ""

        results.append(
            {
                "image_name": row["image_name"],
                "attribute": row["attribute"],
                "prediction": answer,
            }
        )

        # Flush to disk every 20 samples to avoid losing results on interruption
        if len(results) >= 20:
            pd.DataFrame(results).to_csv(
                args.out, mode="a", header=write_header, index=False
            )
            write_header = False
            results = []

        if i % 20 == 0 or i == len(todo):
            speed = i / (time.time() - t0)
            eta = (len(todo) - i) / speed if speed > 0 else 0
            print(f"[{i}/{len(todo)}] {speed:.2f} it/s, ETA {eta/60:.1f} min")

    if results:
        pd.DataFrame(results).to_csv(
            args.out, mode="a", header=write_header, index=False
        )

    print(f"\nDone, results saved to {args.out}")
    print("Next step: python evaluate.py")


if __name__ == "__main__":
    main()
