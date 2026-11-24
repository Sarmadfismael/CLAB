# CLAB: source-only multi-label remote sensing scene classification with centered vision–language anchors

Code, cached features, split definitions and per-seed results for the paper

> A. Güneş, S. F. Ismael, M. Üstüner. *Multi-Label Remote Sensing Scene Classification Without Target Data via Centered Vision–Language Anchors.* Under review, 2026.

> **Revision 1 (R1) update.** This release reproduces the revised manuscript. The CLIP ViT-B/32
> features of the first release had been extracted with OpenCLIP's `ViT-B-32` configuration, which
> pairs the OpenAI weights with the standard GELU activation; the weights were trained with
> QuickGELU (`ViT-B-32-quickgelu`). The CLIP ViT-B/32 image features and text anchors in
> `features/` were re-extracted with QuickGELU and every result in `results/` was recomputed. The
> RemoteCLIP features (loaded as in RemoteCLIP's official code) and the ImageNet, SatMAE and CLIP
> RN50 image features are unchanged. See `CHANGELOG.md`.

CLAB (CLIP Label-Anchor Blending) labels remote sensing tiles from a new acquisition without
seeing any of it during training. It combines two branches on a frozen CLIP image embedding:

1. a **semantic branch** that scores each class by cosine similarity with its class-name
   anchor, after the anchors are **mean-centered and renormalized** (this removes a dominant
   shared direction that holds about 87% of the anchors' second moment);
2. a **linear source head** trained on the labeled source dataset for a short fixed budget.

Both branches are standardized with one location and one scale estimated on **source**
scores and blended with a fixed weight `alpha = 0.30`. No target image, label or statistic is
used at any stage.

```python
from clab import CLAB
model = CLAB().fit(F_source, Y_source, anchors)   # features (N, D), labels (N, C), anchors (C, D)
scores = model.score(F_target)                     # (M, C) ranking scores, one row per tile
```

---

## 1. Quick start (CPU only, no GPU, no images needed)

```bash
git clone https://github.com/Sarmadfismael/CLAB.git
cd CLAB
pip install -r requirements.txt        # numpy, pandas, matplotlib, openpyxl
python run_all.py                      # re-runs every CPU experiment (about 30-60 min)
                                       # and writes results/TABLES.md
```

`python run_all.py --skip e03` leaves out the slow adapter sweep (Table 4).
Every script can also be run on its own, for example `python experiments/e01_main_grid.py`.

The cached CLIP ViT-B/32 and RemoteCLIP ViT-B/32 features are in `features/` (about 70 MB),
so all tables except Tables 10 and 12 are reproduced from them on a CPU. The reference
per-seed CSVs that the paper's numbers were computed from are in `results/`; running the
scripts overwrites them with your own run.

The original results were produced on Windows (Python 3.9). A clean re-run of every CPU
script on Linux (Python 3.11, NumPy 2.4, pandas 3.0) reproduces the per-seed values to within
0.03 mAP and every number printed in the paper to within its last decimal, with one
exception: CLIP-Adapter (Table 4) is trained with Adam and is more sensitive to floating-point
order, so a re-run moves its means by up to 0.1 mAP. `results/adapters*.csv` are the original
runs behind Table 4.

## 2. Where each number in the paper comes from

Table and section numbers refer to the **revised manuscript (R1)**.

| Paper item | Script | Output |
|---|---|---|
| Table 1 (datasets) | `experiments/e10_dataset_stats.py --data-root DIR` (needs the label files) | `results/datasets.csv` |
| Tables 2, 3 and 6 (main results, paired Holm, ablation); zero-shot centering gains (Section 6.2) | `experiments/e01_main_grid.py` | `results/main_grid.csv` |
| Table 4 (whole-target evaluation, two-stage bootstrap), Table A1 (per-class AP, all transfers) | `r1/a5_bootstrap_perclass.py` | `r1/out/a5_*.csv` |
| Table 5, upper block (oracle-tuned CLIP-Adapter / Tip-Adapter) | `experiments/e03_adapters.py` | `results/adapters_sweep.csv`, `results/adapters20.csv` |
| Table 5, lower block (CoOp-sigmoid, DualCoOp-global; needs torch and the CLIP text encoder) | `r1/c2_promptlearn.py sweep` then `final` | `r1/out/c2_*.csv` |
| Tables 7 and 9 (leave-one-transfer-out, sensitivity) | `experiments/e02_loto.py` | `results/loto_*.csv` |
| Table 8 (leave-one-out over six transfers), leave-one-dataset-out (Section 5.5) | `r1/a9_loto6.py`, `r1/a12_lodo.py` | `r1/out/a9_*.csv`, `r1/out/a12_lodo.csv` |
| Source-size sweep (Section 5.6) | `r1/a7_source_size.py` | `r1/out/a7_source_size.csv` |
| Table 10 (out-of-vocabulary imagery restored) | `experiments/e04_robustness.py` | `results/robust_*.csv` |
| Table 11 (hard thresholds from source data, CF1/OF1) | `r1/a1_thresholds.py` | `r1/out/a1_thresholds.csv` |
| Table 12 (published results, context only) | literature values | `supplementary/published_results.csv` |
| Table 13 (DDA-MLIC and DANN re-run on the CLIP features) | `r1/c3_uda.py` | `r1/out/c3_uda_ddamlic_dann.csv` |
| Table 14 (eight frozen backbones) | `extraction/backbone_study.py --data-root DIR` (GPU, images); CLIP ViT-B/32 row from cached features: `r1/b1_backbone_clip.py` | `results/backbone_study.csv` |
| Table 15, Figure 2 (anchor geometry), centering set | `experiments/e09_geometry.py`, `tools/make_fig_anchors.py`; raw-about-mean row: `r1/a6_centering_alternatives.py` | `results/geometry_*.csv` |
| Table 16 (alternatives to centering) | `r1/a6_centering_alternatives.py` | `r1/out/a6_centering.csv` |
| Table 17 (prompt families), DFC15 vocabulary variant | `r1/b2_prompts.py` (text encoder) | `results/prompt_ablation.csv`, `results/vocabulary_variant.csv` |
| Table 18 (fusion rule) | `experiments/e05_fusion.py` | `results/fusion.csv` |
| Table 19 (per-class AP, DFC15), Proposition 1 measurements | `experiments/e06_perclass_invariance.py` | `results/perclass.csv`, `results/invariance.csv` |
| Table 20, Table A2 (synonym anchors) | `r1/c1_synonyms.py` (text encoder) | `r1/out/c1_synonyms*.csv` |
| Table 21 (label-free estimators), oracle weights | `experiments/e08_alpha_estimators.py` | `results/alpha_*.csv` |
| Table 22 (contamination audit) | from `main_grid.csv` and `backbone_study.csv` | – |
| Source-head budget (Section 6.6); ridge / weight-decay heads; per-class standardization; whole-target means | `experiments/e07_source_budget.py`; `r1/a2_head_norm_eval.py` | `results/source_budget.csv`; `r1/out/a2_heads.csv`, `a3_perclass_z.csv`, `a4_full_target.csv` |
| Co-occurrence prior (Section 6.8) | `r1/a8_cooccurrence.py` | `r1/out/a8_cooc.csv` |
| Figures 4 and 5 (confusion, examples) | `r1/a11_examples.py`, `r1/a11b_pick.py`, `r1/f1_figures.py` (example tiles from the datasets) | `r1/out/a11_*` |
| Computational cost (Section 3.6) | `r1/a10_cost.py` | printed |
| LaTeX table bodies of the revised paper | `r1/gen_tables.py`, `r1/gen_extra.py` | `r1/out/tables_*.json` |

`python tools/make_tables.py` prints the tables of the `experiments/` scripts from `results/`.
The `r1/` scripts read `features/` and `results/` and write to `r1/out/` (override with
`CLAB_FEATURES`, `CLAB_RESULTS`, `R1_OUT`).

## 3. Protocol

**Transfers.** AID → UC Merced and UC Merced → AID in the 17-label space; AID → DFC15 and
UC Merced → DFC15 in the six-label space shared by all three datasets
(`water, grass, buildings, trees, ship, cars`; DFC15's `vegetation` is matched to `grass`
and `boat` to `ship`). In the shared-six space, images with no positive label are removed
from both domains (`shared6`); the robustness check of Table 8 uses `shared6keep`, where
the target keeps them and the source is still filtered.

**Splits and seeds.** Split seeds 0–19. For seed `k`, each dataset's rows are permuted with
`numpy.random.RandomState(k).permutation(n)`. The source head is fitted on the first
`round(0.8 n)` rows of the source permutation; evaluation uses the remaining 20% of the
target permutation (420 UC Merced, 600 AID or 438 DFC15 images). The other target images
are never used. The same seed also initializes the head weights. In the robustness check
(Table 8) the source still uses its `shared6` rows (images with at least one positive
label, in the same order), while the target is evaluated on its `shared6keep` rows.
`splits/{dataset}__{space}.csv` lists, for every image, its row index, image id, labels
and its role (`fit` / `eval`) under every seed, so the protocol can be reproduced without
this code (`python tools/export_splits.py` regenerates them from the feature files).

**Method hyperparameters** (`clab/config.py`, identical for every transfer and backbone):
linear head, full-batch gradient descent on binary cross-entropy, learning rate 0.1,
weight decay 1e-4, 50 epochs, weights from N(0, 0.01²); features standardized per
dimension with source statistics; blending weight `alpha = 0.30`.

**Prompts.** Each class anchor is the renormalized mean of the L2-normalized text
embeddings of

```
"an aerial photograph containing {}"
"a satellite image of {}"
"a remote sensing image showing {}"
```

with the class name inserted verbatim (hyphens replaced by spaces). The anchors used in
the paper are stored as `features/{backbone}__TEXT__{native,shared6}.npy`.

**Metric.** mAP is the mean over classes of average precision computed on raw scores.
CF1/OF1 at a median-score threshold are stored as secondary metrics.

**Statistics.** Paired 95% t-intervals over the 20 seeds; Holm–Bonferroni step-down
intervals over the eight CLAB-vs-probe contrasts (`clab/stats.py`): the contrast with the
i-th smallest p-value receives a two-sided interval at level 1 − 0.05/(8 − i + 1).

## 4. Backbones and checkpoints

| Name in this repository | Model | Weights |
|---|---|---|
| `clip_vitb32_control` (headline) | CLIP ViT-B/32 | OpenCLIP `ViT-B-32-quickgelu`, `pretrained="openai"` |
| `remoteclip_vitb32` (diagnostic) | RemoteCLIP ViT-B/32 | Hugging Face `chendelong/RemoteCLIP`, file `RemoteCLIP-ViT-B-32.pt` |
| `clip_rn50_control` | CLIP RN50 | OpenCLIP `RN50-quickgelu`, `pretrained="openai"` |
| `remoteclip_rn50` | RemoteCLIP RN50 | `chendelong/RemoteCLIP`, `RemoteCLIP-RN50.pt` |
| `remoteclip_vitl14` | RemoteCLIP ViT-L/14 | `chendelong/RemoteCLIP`, `RemoteCLIP-ViT-L-14.pt` |
| `imagenet_tresnet_m` | TResNet-M, ImageNet-1K | timm `tresnet_m` |
| `imagenet21k_tresnet_m` | TResNet-M, ImageNet-21K | timm `tresnet_m.miil_in21k` |
| `satmae_vitlarge` | SatMAE ViT-L (fMoW pretraining) | `rshf`, `MVRL/satmae-vitlarge-fmow-pretrain-800` |

Images are resized to 224 × 224 (no crop) and normalized with CLIP statistics for the
CLIP-family models and ImageNet statistics otherwise. RemoteCLIP is used only as a
diagnostic: UC Merced imagery is part of its pretraining corpus (via UCM-Captions), so the
headline claims are made on stock CLIP.

## 5. Re-extracting the features (optional, GPU)

```bash
pip install -r requirements-extract.txt
python extraction/extract_features.py --data-root /path/to/datasets
```

Expected dataset layout (the multi-label annotation files are distributed by the dataset
authors):

```
AID/multilabel.csv              AID/images_tr/...  AID/images_test/...
ucmerced/LandUse_Multilabeled.csv   ucmerced/uc_merced/<class>/<image>.tif
DFC15/multilabel.csv            DFC15/images_tr/...  DFC15/images_test/...
```

The cached features were extracted on one NVIDIA RTX 3060 (12 GB). Re-extracted features can differ from the cached ones in the last
decimals because of GPU non-determinism and library versions; the rows are in label-file
order and carry their image ids, so a re-extraction is checked against `splits/`.

## 6. Repository layout

```
clab/            the method (core.py), protocol constants (config.py), data loading, statistics
experiments/     e01-e10: one script per table group, CPU only
extraction/      feature extraction, prompt ablation, backbone study (need torch / open_clip)
features/        cached frozen features and anchors (CLIP ViT-B/32, RemoteCLIP ViT-B/32)
splits/          per-image split membership for every seed
results/         per-seed results behind every number in the paper
supplementary/   values reproduced from the literature (Table 9) and the end-to-end
                 retraining variability quoted in the paper
tools/           table builder, split exporter, anchor figure
r1/              analyses added in the first revision, re-implemented CLIP encoders, outputs in r1/out/
```

## 7. Notes

* **Table 9** contains values reproduced from other papers, not rerun here; the source of
  every row is listed in `supplementary/published_results.csv`.
* **GPU-only tables.** `results/backbone_study.csv`, `prompt_ablation.csv` and
  `vocabulary_variant.csv` are the original outputs; they need the encoders (and, for the
  backbone study, the images) to regenerate.
* **The 1.16–4.66 mAP band** quoted in the paper for the seed-to-seed spread of a complete
  training pipeline comes from our preliminary end-to-end fine-tuning of TResNet-M on the
  AID ↔ UC Merced transfers (3–5 seeds per configuration). Those runs are not part of this
  method; their means and standard deviations are listed in
  `supplementary/retraining_seed_sd.csv`.
* **Licenses of the data.** The cached features are derived from AID, UC Merced and the
  2015 IEEE GRSS Data Fusion Contest imagery with the multi-label annotations released by
  their authors; please cite the original datasets (see the paper) when using them.

## Citation

```
@article{gunes2026clab,
  title   = {Multi-Label Remote Sensing Scene Classification Without Target Data via
             Centered Vision--Language Anchors},
  author  = {G{\"u}ne{\c{s}}, Ali and Ismael, Sarmad F. and {\"U}st{\"u}ner, Mustafa},
  note    = {Under review},
  year    = {2026}
}
```

## License

MIT (see `LICENSE`). Contact: Ali Güneş (corresponding author), Sarmad F. Ismael
(sarmad.ismael@uoninevah.edu.iq).
