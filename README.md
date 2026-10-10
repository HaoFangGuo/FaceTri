# FaceTri

**How protocol-sensitive is VLM abstention, really?**

This project is a measurement-methodology study of prompt-elicited abstention
in vision-language models (VLMs), not a conventional benchmark. We ask a
narrower question than "how accurate is model X on facial attributes": when a
VLM says *"Uncertain"*, is that response tracking genuine visual ambiguity, or
is it an artifact of how the question was phrased?

To separate the two, we compare model abstention against an **independent**
signal of ambiguity — disagreement among three human annotators — rather than
treating abstention rate alone as a sign of good calibration.

Official code and data for **“Uncertain” Is Uncertain: Prompt-Induced Abstention
in VLMs Is Neither Stable Nor Human-Aligned**, accepted as a poster at the
**NeurIPS 2026 Trust-AI-Eval Workshop**.

**Authors:** Fang Hao, Lu-Yan Wang, and Shang-Hong Lai.

[Paper on OpenReview](https://openreview.net/forum?id=yRJLG6UmYo)

FaceTri uses a three-option response protocol: **Yes / No / Uncertain**.

## Key findings

- **Prompt-elicited abstention is unstable.** On a selected diagnostic subset
  of 88 items, swapping the order of answer and reason changes abstention
  from 36.4% to 65.9% for Qwen2.5-VL, from 17.0% to 51.1% for InternVL3,
  and from 39.8% to 6.8% for MiniCPM-V-4. Molmo2-O remains at 95.5%.
  These are diagnostic-subset results, not estimates for the full benchmark.
- **Human alignment is limited in the InternVL3 analysis.** InternVL3 abstains
  more often on items with annotator disagreement than on unanimous items
  (68.2% vs 55.1%; χ² ≈ 8.84, p ≈ .003). However, 65.5% of its abstentions
  occur on unanimous items. Human disagreement is an operational proxy for
  ambiguity, rather than a definitive measure of visual uncertainty.
- **Performance depends on the attribute set.** Objective control attributes
  (Eyeglasses, Bald, Wearing_Hat, Wearing_Earrings) yield higher Effective
  Reliability than the fine-grained facial attributes studied here.

The study evaluates four small VLMs on one facial-attribute task. Its findings
should not be read as claims about all VLMs or all forms of uncertainty.

## Models and data

- **Models evaluated:** Qwen2.5-VL-7B-Instruct, InternVL3-8B, Molmo2-O-7B,
  MiniCPM-V-4
- **Dataset:** [CelebA-HQ](https://github.com/switchablenorms/CelebAMask-HQ),
  9 native attributes plus 3 custom-annotated wrinkle attributes
  (`Forehead_Wrinkles`, `Crows_Feet`, `Nasolabial_Folds`), majority-voted
  across 3 independent annotators (Fleiss' κ = 0.587)
- **Protocol:** balanced Yes/No/Uncertain visual question answering, 100
  positive / 100 negative examples per attribute (2,400 instances total)
- **Metrics:** Coverage, Risk, and Effective Reliability Φ (Whitehead et al.,
  ECCV 2022), computed per attribute and stratified by an agreement-based
  cell scheme (A–E)

See `data/README.md` for dataset provenance, licensing, and why the raw
images are not redistributed in this repository.

## Repository structure

```
data/                   Ground-truth benchmark CSVs and annotation files
  benchmark/            The Yes/No/Uncertain benchmark instances used for inference
src/
  data_prep/            Build the benchmark from raw CelebA-HQ annotations
  inference/            Per-model inference scripts + reason-augmented re-inference
  evaluation/           Score predictions into Coverage / Risk / Effective Reliability
  analysis/             Agreement-stratified cells, case studies, statistical tests
results/
  predictions/          Raw model outputs (Yes/No/Uncertain) per model and prompt variant
  reinference/          Reason-augmented re-inference outputs used for the order-swap test
  evals/                Per-model, per-attribute metric tables
  cells/                Agreement-stratified case samples (cells A–E)
  tables/               Aggregated tables reported in the paper
  case_studies/         Qualitative case selections for specific attributes
```

## Running the pipeline

Everything is run from the repository root so the default relative paths in
each script resolve correctly, e.g.:

```bash
pip install -r requirements.txt

# 1. Use the provided final benchmark:
#    data/benchmark/benchmark_final.csv
#    Download the source images separately and place them in CelebA-HQ-img/.

# 2. Run inference for a given model (requires the model's own dependencies —
#    see the docstring at the top of each inference script, GPU strongly recommended)
python src/inference/inference_qwen.py --gt data/benchmark/benchmark_final.csv

# 3. Score predictions
python src/evaluation/evaluate.py --gt data/benchmark/benchmark_final.csv --pred results/predictions/prediction.csv

# 4. Merge per-model metrics into the paper's main tables
python src/evaluation/merge_main_table.py

# 5. Downstream analysis (agreement-stratified cells, order-swap test, case studies)
python src/analysis/find_disagreement.py
python src/analysis/mcnemar_order_swap.py
```

The data-preparation scripts are retained for provenance. In particular,
`explore_dataset.py` currently samples 50 images per class; it does not by
itself recreate the final 100-per-class benchmark. For the workflow above,
use the provided final CSV rather than rebuilding or rewriting it.

The core requirements are listed in `requirements.txt`; inference dependencies
must be installed separately for each model. This release has not yet been
validated end to end in a fresh environment.

Each script's docstring provides inputs, outputs, and
model-specific setup notes (e.g. the `bfloat16` + SDPA configuration needed to
avoid NaN/Inf logits on MiniCPM-V-4, or why `use_flash_attn` must be disabled
for InternVL3 on Turing-architecture GPUs).

## License

Code in this repository is released under the MIT License (see `LICENSE`).
The CelebA-HQ images themselves are **not** redistributed here; they remain
subject to the original dataset's non-commercial research/educational use
restriction (see `data/README.md`).

## Citation

```bibtex
@inproceedings{
hao2026uncertain,
title={{\textquotedblleft}Uncertain{\textquotedblright} Is Uncertain: Prompt-Induced Abstention in {VLM}s Is Neither Stable Nor Human-Aligned},
author={Fang Hao and Lu-Yan Wang and Shang-Hong Lai},
booktitle={NeurIPS 2026 Trust-AI-Eval Workshop},
year={2026},
url={https://openreview.net/forum?id=yRJLG6UmYo}
}
```
