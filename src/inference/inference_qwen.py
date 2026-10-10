"""
inference_qwen.py — Run Qwen2.5-VL-7B-Instruct on FaceVLM-bench lite
Usage (on a GPU-equipped server):
    python inference_qwen.py [--gt benchmark.csv] [--img-dir CelebA-HQ-img] [--out prediction.csv]
Dependencies:
    pip install torch transformers accelerate qwen-vl-utils pandas
Output prediction.csv: image_name, attribute, prediction (raw model output)
Supports resume: re-running will skip samples already present in prediction.csv.
"""
import argparse
import os
import time

import pandas as pd
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"


def load_model():
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    return model, processor


@torch.inference_mode()
def ask(model, processor, image_path, prompt):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    # max_new_tokens raised from 10 to 15 to comfortably fit the
    # three-way answer set (Yes / No / Uncertain) without truncation.
    generated = model.generate(**inputs, max_new_tokens=15, do_sample=False)
    trimmed = generated[:, inputs.input_ids.shape[1]:]
    return processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default="data/benchmark/benchmark.csv")
    ap.add_argument("--img-dir", default="CelebA-HQ-img")
    ap.add_argument("--out", default="results/predictions/prediction.csv")
    args = ap.parse_args()

    bench = pd.read_csv(args.gt)

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
