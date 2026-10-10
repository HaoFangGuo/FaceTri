# Data

## Source dataset

This project builds on [CelebA-HQ / CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ).
The raw high-resolution images and the original attribute annotation file
(`CelebAMask-HQ-attribute-anno.txt`) are **not redistributed in this
repository** — the dataset's license restricts use to non-commercial research
and educational purposes. See `README_CelebAMask-HQ.txt` (the dataset
maintainers' own file, kept verbatim for citation) for the official citation
and download instructions.

To reproduce the pipeline, download CelebA-HQ from the source above and place
`CelebAMask-HQ-attribute-anno.txt` in this directory (it is git-ignored).

## What's actually in this folder

- `wrinkles_anno.txt` — per-image labels for the three custom wrinkle
  attributes (Forehead_Wrinkles, Crows_Feet, Nasolabial_Folds)
- `wrinkles_annotations_long.csv` — the raw per-annotator labels behind
  `wrinkles_anno.txt`: one row per (image, attribute, annotator). Three
  independent annotators labeled each of 431 images; annotator identities are
  anonymized here as `Annotator_1`/`Annotator_2`/`Annotator_3`
- `wrinkles_annotations_aggregated.csv` — the same labels aggregated to a
  majority vote per (image, attribute), with agreement rate and entropy
  (used by the analysis scripts to compare model abstention against human
  disagreement)
- `wrinkles_annotation_summary.csv` — per-attribute agreement statistics
  across all 431 triple-annotated images (full agreement vs 2-to-1 split,
  mean entropy); overall inter-annotator agreement was Fleiss' κ = 0.587
- `wrinkle_rows.csv`, `wrinkle_rows_meta.csv` — the wrinkle attributes
  reformatted into benchmark rows, plus per-row metadata (e.g. annotator
  agreement) used by the analysis scripts
- `benchmark/` — the final Yes/No/Uncertain benchmark instances fed to each
  model, including prompt-variant versions used for the protocol-sensitivity
  experiments (2-option vs 3-option, order-swap, Molmo2-O-specific prompt
  wording, and the objective-attribute control set)

None of these files contain face images — only image filenames (referencing
the CelebA-HQ image set), attribute labels, and prompts.
