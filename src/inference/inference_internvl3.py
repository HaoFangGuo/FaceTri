"""
inference_internvl.py — run FaceVLM-bench lite with InternVL3-8B

Usage (on a GPU server):
    python inference_internvl.py [--gt benchmark.csv] [--img-dir CelebA-HQ-img] [--out prediction_internvl.csv]

Dependencies:
    pip install torch transformers accelerate torchvision pandas
    (InternVL uses trust_remote_code to pull custom modeling code; qwen-vl-utils is not needed)

Note (Quadro RTX 8000 / Turing architecture):
    use_flash_attn must be set to False — this architecture does not support FlashAttention-2

Output prediction_internvl.csv: image_name, attribute, prediction (raw model output)
Format is identical to prediction.csv (Qwen), so the same evaluate.py can be reused.
Supports resume: re-running will skip samples already present in prediction_internvl.csv.
"""

import argparse
import os
import time

import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

MODEL_ID = "OpenGVLab/InternVL3-8B"   # switch to a local path if already downloaded

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


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
        use_flash_attn=False,      # Turing GPU — must be False
        trust_remote_code=True,
    ).eval().cuda()
    return model, tokenizer


@torch.inference_mode()
def ask(model, tokenizer, image_path, prompt):
    transform = build_transform()
    image = Image.open(image_path)
    pixel_values = transform(image).unsqueeze(0).to(torch.bfloat16).cuda()

    question = f"<image>\n{prompt}"
    generation_config = dict(max_new_tokens=15, do_sample=False)

    response = model.chat(tokenizer, pixel_values, question, generation_config)
    return response.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default="data/benchmark/benchmark.csv")
    ap.add_argument("--img-dir", default="CelebA-HQ-img")
    ap.add_argument("--out", default="results/predictions/prediction_internvl.csv")
    args = ap.parse_args()

    bench = pd.read_csv(args.gt)

    # Resume: skip samples already completed
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
    print(f"[MEM] Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")

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

    print(f"\nDone. Results saved to {args.out}")
    print("Next step: python evaluate.py --pred prediction_internvl.csv")


if __name__ == "__main__":
    main()
