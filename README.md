# CLAB: source-only multi-label remote sensing scene classification with centered vision–language anchors

Code, cached features, split definitions and per-seed results for the paper

> A. Güneş, S. F. Ismael, M. Üstüner. *Multi-Label Remote Sensing Scene Classification Without Target Data via Centered Vision–Language Anchors.* Under review, 2026.

CLAB (CLIP Label-Anchor Blending) labels remote sensing tiles from a new acquisition without
seeing any of it during training. It combines two branches on a frozen CLIP image embedding:

1. a **semantic branch** that scores each class by cosine similarity with its class-name
   anchor, after the anchors are **mean-centered and renormalized** (this removes a dominant
   shared direction that holds about 90% of the anchor variance);
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

| Paper item | Script | Output in `results/` |
|---|---|---|
| Table 1 (datasets) | `experiments/e10_dataset_stats.py --data-root DIR` (needs the label files) | `datasets.csv` |
| Table 2 (main results), Table 3 (paired, Holm), Table 5 (ablation), zero-shot centering gains (Section 6.2) | `experiments/e01_main_grid.py` | `main_grid.csv` |
| Table 4 (oracle-tuned CLIP-Adapter / Tip-Adapter) | `experiments/e03_adapters.py` | `adapters_sweep.csv`, `adapters20.csv` |
| Table 6 (leave-one-transfer-out), Table 7 (alpha and budget sensitivity) | `experiments/e02_loto.py` | `loto_raw.csv`, `loto_selected.csv` |
| Table 8 (out-of-vocabulary imagery restored) and the alpha trade-off | `experiments/e04_robustness.py` | `robust_mismatch.csv`, `robust_standard.csv` |
| Table 9 (published results) | literature values, see `supplementary/published_results.csv` | – |
| Table 10 (eight frozen backbones) | `extraction/backbone_study.py --data-root DIR` (GPU, images) | `backbone_study.csv` |
| Table 11, Figure 3 (anchor geometry), choice of centering set | `experiments/e09_geometry.py`, `tools/make_fig_anchors.py` | `geometry_*.csv`, `fig_anchors.pdf` |
| Table 12 (prompt families), DFC15 vocabulary variant | `extraction/prompt_ablation.py` (text encoder) | `prompt_ablation.csv`, `vocabulary_variant.csv` |
| Table 13 (fusion rule) | `experiments/e05_fusion.py` | `fusion.csv` |
| Table 14 (per-class AP), Proposition 1 measurements | `experiments/e06_perclass_invariance.py` | `perclass.csv`, `invariance.csv` |
| Source-head budget sweep (Section 6.6) | `experiments/e07_source_budget.py` | `source_budget.csv` |
| Oracle alpha and Table 15 (label-free estimators) | `experiments/e08_alpha_estimators.py` | `alpha_oracle.csv`, `alpha_estimators.csv` |
| Table 16 (contamination audit) | computed from `main_grid.csv` and `backbone_study.csv` | – |

`python tools/make_tables.py` prints every table above from the CSVs in `results/`.

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
| `clip_vitb32_control` (headline) | CLIP ViT-B/32 | OpenCLIP `ViT-B-32`, `pretrained="openai"` |
| `remoteclip_vitb32` (diagnostic) | RemoteCLIP ViT-B/32 | Hugging Face `chendelong/RemoteCLIP`, file `RemoteCLIP-ViT-B-32.pt` |
| `clip_rn50_control` | CLIP RN50 | OpenCLIP `RN50`, `pretrained="openai"` |
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
tools/           table builder, split exporter, Figure 3
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
