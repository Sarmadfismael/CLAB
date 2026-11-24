# Changelog

## R1 (revision for ISPRS IJGI, ijgi-4635198)

- **Fixed:** CLIP ViT-B/32 (and RN50) are now loaded as OpenCLIP `ViT-B-32-quickgelu` /
  `RN50-quickgelu` with `pretrained="openai"` (`extraction/backbones.py`, `clab/config.py`). The first
  release used `ViT-B-32`, which applies GELU to weights trained with QuickGELU (OpenCLIP warns
  about this mismatch). The cached CLIP ViT-B/32 image features and text anchors were re-extracted
  and every table recomputed. The CLIP RN50 image encoder has no GELU, so its image features are
  unchanged. The re-extraction used `r1/clipvis.py` and `r1/cliptext.py`, which reproduce OpenCLIP's
  output to float precision with either activation (checked against the first-release features
  with GELU and against `ViT-B-32-quickgelu` anchors).
- **Added:** `r1/` with every analysis added in the revision (thresholds, regularized heads,
  per-class standardization, bootstrap, centering alternatives, six-transfer and
  leave-one-dataset-out selection, source-size sweep, co-occurrence prior, synonym anchors,
  prompt-learning baselines, DDA-MLIC and DANN on CLIP features, confusion counts and examples,
  cost), with their outputs in `r1/out/`.
- **Updated:** `results/` (per-seed CSVs and `TABLES.md`) and `README.md` (table mapping for the
  revised manuscript).
