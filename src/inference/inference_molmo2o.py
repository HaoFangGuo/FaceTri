"""
inference_molmo2o.py — Run FaceVLM-bench lite with Molmo2-O-7B (Olmo backbone)

Usage (on a GPU server):
    python inference_molmo2o.py [--gt benchmark_final.csv] [--img-dir benchmark_images] [--out prediction_molmo2o_uncertain.csv]

Dependencies:
    pip install transformers==4.57.1 torch pillow einops torchvision accelerate decord2 molmo_utils

Output: image_name, attribute, prediction (raw model output)
Supports resume: re-running will skip samples already present in the output file.
"""

import argparse
import os
import time

import pandas as pd
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

MODEL_ID = "allenai/Molmo2-O-7B"


def load_model():
    processor = AutoProcessor.from_pretrained(
        MODEL_ID, trust_remote_code=True, dtype="auto"
    )
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, trust_remote_code=True, dtype=torch.bfloat16, low_cpu_mem_usage=False
    ).to("cuda")
    return model, processor


@torch.inference_mode()
def ask(model, processor, image_path, prompt):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image", "image": Image.open(image_path)},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    inputs = {
    k: v.to("cuda:0") if torch.is_tensor(v) else v
    for k, v in inputs.items()
}

    # max_new_tokens kept small since we only need a one-word
    # Yes / No / Uncertain answer, not a long description.
    generated_ids = model.generate(**inputs, max_new_tokens=15)
    generated_tokens = generated_ids[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(
        generated_tokens, skip_special_tokens=True
    )[0].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default="data/benchmark/benchmark_final.csv")
    ap.add_argument("--img-dir", default="benchmark_images")
    ap.add_argument("--out", default="results/predictions/prediction_molmo2o_uncertain.csv")
    args = ap.parse_args()

    bench = pd.read_csv(args.gt)

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
    model, processor = load_model()

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
                answer = ask(model, processor, img_path, row["prompt"])
            except Exception as e:
                print(f"Error on {row['image_name']}/{row['attribute']}: {e}")
                answer = ""

        results.append(
            {"image_name": row["image_name"], "attribute": row["attribute"], "prediction": answer}
        )

        if len(results) >= 20:
            pd.DataFrame(results).to_csv(args.out, mode="a", header=write_header, index=False)
            write_header = False
            results = []

        if i % 20 == 0 or i == len(todo):
            speed = i / (time.time() - t0)
            eta = (len(todo) - i) / speed if speed > 0 else 0
            print(f"[{i}/{len(todo)}] {speed:.2f} it/s, ETA {eta/60:.1f} min")

    if results:
        pd.DataFrame(results).to_csv(args.out, mode="a", header=write_header, index=False)

    print(f"\nDone. Results saved to {args.out}")
    print(f"Next step: python evaluate.py --pred {args.out}")


if __name__ == "__main__":
    main()
