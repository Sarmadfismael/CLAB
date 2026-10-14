## Table 1. Datasets

| dataset | images | cat. | labels/img | shared-6 images | removed |
|---|---|---|---|---|---|
| aid | 3000 | 17 | 5.15 | 2740 | 8.7% (260) |
| ucm | 2100 | 17 | 3.33 | 1726 | 17.8% (374) |
| dfc15 | 3342 | 8 | 2.80 | 2190 | 34.5% (1152) |

## Table 2. Main results, CLIP ViT-B/32 (mAP, mean +- SD over 20 seeds)

| method | AID->UCM | UCM->AID | AID->DFC15 | UCM->DFC15 | mean |
|---|---|---|---|---|---|
| zeroshot_raw | 40.54 +- 1.21 | 48.92 +- 1.75 | 61.49 +- 1.70 | 61.49 +- 1.70 | 53.11 |
| zeroshot | 57.08 +- 1.21 | 63.75 +- 1.91 | 71.08 +- 1.96 | 71.08 +- 1.96 | 65.75 |
| probe | 61.30 +- 1.69 | 70.53 +- 2.77 | 71.60 +- 1.78 | 78.33 +- 1.54 | 70.44 |
| clab_raw | 62.05 +- 1.47 | 72.11 +- 2.69 | 73.41 +- 1.59 | 79.23 +- 1.40 | 71.70 |
| clab_long | 65.22 +- 1.34 | 72.21 +- 2.56 | 73.97 +- 2.17 | 76.98 +- 1.69 | 72.10 |
| clab_a02 | 66.59 +- 1.39 | 72.68 +- 2.64 | 74.55 +- 1.80 | 79.40 +- 1.54 | 73.31 |
| clab_a04 | 66.61 +- 1.17 | 72.14 +- 2.53 | 75.59 +- 1.80 | 79.05 +- 1.65 | 73.35 |
| clab | 66.80 +- 1.27 | 72.66 +- 2.58 | 75.23 +- 1.78 | 79.51 +- 1.61 | 73.55 |

## Table 3. CLAB minus probe, paired over 20 seeds, Holm over 8 contrasts

| backbone | transfer | delta | 95% CI | Holm-adjusted CI | resolved |
|---|---|---|---|---|---|
| clip_vitb32_control | AID->UCM | +5.50 | [+5.05, +5.95] | [+4.87, +6.13] | yes |
| clip_vitb32_control | UCM->AID | +2.13 | [+1.80, +2.46] | [+1.75, +2.51] | yes |
| clip_vitb32_control | AID->DFC15 | +3.63 | [+3.15, +4.12] | [+3.00, +4.27] | yes |
| clip_vitb32_control | UCM->DFC15 | +1.17 | [+0.87, +1.48] | [+0.87, +1.48] | yes |
| remoteclip_vitb32 | AID->UCM | +11.44 | [+11.00, +11.87] | [+10.80, +12.08] | yes |
| remoteclip_vitb32 | UCM->AID | +1.83 | [+1.65, +2.01] | [+1.59, +2.08] | yes |
| remoteclip_vitb32 | AID->DFC15 | +9.29 | [+8.79, +9.79] | [+8.57, +10.01] | yes |
| remoteclip_vitb32 | UCM->DFC15 | +1.73 | [+1.48, +1.98] | [+1.42, +2.04] | yes |

## Table 5. Ablation (mean delta mAP vs full CLAB over 4 transfers x 2 backbones)

- raw anchors (no centering): -3.64
- no source head: -6.51
- no semantic branch: -4.59
- 800-epoch head: -1.08

Centering gain in zero-shot (targets UCM, AID, DFC15): clip UCM +16.54, clip AID +14.82, clip DFC15 +9.60, remoteclip UCM +12.23, remoteclip AID +7.31, remoteclip DFC15 +21.23

## Table 4. Oracle-tuned CLIP adaptation baselines (CLIP ViT-B/32)

| transfer | best baseline (config) | baseline | CLAB | delta [95% CI] |
|---|---|---|---|---|
| AID->UCM | clip_adapter|centred|0.2|0.001|200 | 61.30 | 66.80 | +5.51 [+4.73, +6.28] |
| UCM->AID | clip_adapter|centred|0.2|0.001|200 | 64.76 | 72.66 | +7.91 [+7.29, +8.52] |
| AID->DFC15 | tip_adapter|centred|0.4|1.0|z|None | 72.21 | 75.23 | +3.03 [+2.53, +3.53] |
| UCM->DFC15 | clip_adapter|raw|0.5|0.001|200 | 77.12 | 79.51 | +2.39 [+1.90, +2.89] |

Centering gain inside Tip-Adapter: 10.50 to 15.22; inside CLIP-Adapter: -8.63 to 6.24

## Table 6. Leave-one-transfer-out selection

| held out | alpha | epochs | CLAB | probe | delta [95% CI] |
|---|---|---|---|---|---|
| AID->UCM | 0.30 | 50 | 66.80 | 61.30 | +5.50 [+5.05, +5.95] |
| UCM->AID | 0.30 | 50 | 72.66 | 70.53 | +2.13 [+1.80, +2.46] |
| AID->DFC15 | 0.30 | 50 | 75.23 | 71.60 | +3.63 [+3.15, +4.12] |
| UCM->DFC15 | 0.30 | 100 | 79.07 | 77.90 | +1.17 [+0.88, +1.45] |

## Table 7. Sensitivity to alpha (50 epochs) and to the budget (alpha = 0.30)

| alpha | AID->UCM | UCM->AID | AID->DFC15 | UCM->DFC15 | mean |
|---|---|---|---|---|---|
| 0.05 | 63.81 | 71.43 | 72.53 | 78.70 | 71.62 |
| 0.10 | 65.41 | 72.07 | 73.35 | 78.98 | 72.45 |
| 0.15 | 66.17 | 72.46 | 74.01 | 79.23 | 72.97 |
| 0.20 | 66.59 | 72.68 | 74.55 | 79.40 | 73.31 |
| 0.25 | 66.73 | 72.73 | 74.92 | 79.50 | 73.47 |
| 0.30 | 66.80 | 72.66 | 75.23 | 79.51 | 73.55 |
| 0.35 | 66.76 | 72.48 | 75.42 | 79.35 | 73.50 |
| 0.40 | 66.61 | 72.14 | 75.59 | 79.05 | 73.35 |
| 0.45 | 66.38 | 71.72 | 75.67 | 78.67 | 73.11 |
| 0.50 | 66.03 | 71.20 | 75.56 | 78.24 | 72.76 |
- budget 20: mean 73.37
- budget 50: mean 73.55
- budget 100: mean 73.46
- budget 200: mean 73.22

## Table 8. Out-of-vocabulary imagery restored in the target (alpha = 0.30)

| transfer | OOV % | probe | CLAB | delta [95% CI] |
|---|---|---|---|---|
| AID->UCM | 17.2 | 71.20 | 74.06 | +2.85 [+2.45, +3.26] |
| AID->DFC15 | 34.4 | 69.19 | 72.15 | +2.96 [+2.57, +3.35] |
| UCM->AID | 8.7 | 82.80 | 83.29 | +0.48 [+0.19, +0.78] |
| UCM->DFC15 | 34.4 | 75.15 | 75.50 | +0.35 [+0.08, +0.61] |
| DFC15->AID | 8.7 | 76.20 | 80.32 | +4.12 [+3.56, +4.68] |
| DFC15->UCM | 17.2 | 70.75 | 72.56 | +1.81 [+1.46, +2.16] |

Mean over six: +2.10; worst direction at alpha 0.20 / 0.30 / 0.40: +0.66 / +0.35 / -0.63
Standard transfers: mean gain over the probe varies by 0.24 for alpha in [0.20, 0.40]

## Table 9. Published results (reproduced from the literature) and margins

| method | backbone | AID->UCM | UCM->AID | AID->DFC15 | UCM->DFC15 | source |
|---|---|---|---|---|---|---|
| ResNet-101 (source only) | ResNet-101 | 57.5 | 51.7 | 56.9 | 66.4 | Singh et al., DDA-MLIC (WACV 2024), Tables 1-2 (their re-implementation) |
| ML-GCN | ResNet-101 | 53.7 | 51.3 | 58.9 | 64.6 | Singh et al., DDA-MLIC (WACV 2024), Tables 1-2 |
| ML-AGCN | TResNet-M | 55.2 | 52.1 | 51.6 | 70.3 | Singh et al., DDA-MLIC (WACV 2024), Tables 1-2 |
| ASL (source only) | TResNet-M | 55.4 | 54.1 | 56.1 | 68.9 | Singh et al., DDA-MLIC (WACV 2024), Tables 1-2 (their re-implementation) |
| DANN | TResNet-M | 52.5 | 51.6 | 43.0 | 64.1 | Singh et al., DDA-MLIC (WACV 2024), Tables 1-2 (their re-implementation) |
| DALN | TResNet-M | 53.1 | 53.2 | 44.7 | 65.6 | Singh et al., DDA-MLIC (WACV 2024), Tables 1-2 (their re-implementation) |
| DA-MAIC | TResNet-M | 54.4 | 50.5 | 55.4 | 65.8 | Singh et al., DDA-MLIC (WACV 2024), Tables 1-2 (their re-implementation, with ASL) |
| DDA-MLIC | TResNet-M | 63.2 | 54.9 | 62.1 | 70.6 | Singh et al., DDA-MLIC (WACV 2024), Tables 1-2 |
| DA-AGCN | TResNet-M | 59.0 | 57.2 | 65.7 | 76.5 | Singh et al., ML-AGCN (CVIU 2024), Tables 6-7 |
| DDA-MLIC + DeepEM | TResNet-M | 60.5 | 56.2 | 63.2 | 73.1 | Singh et al., arXiv:2505.14333 (2025), Tables 1-2 |
| CLAB (ours) | CLIP ViT-B/32 | 66.8 | 72.7 | 75.2 | 79.5 | |

Margin over best published: AID->UCM +3.6, UCM->AID +15.5, AID->DFC15 +9.5, UCM->DFC15 +3.0
Spread among published methods per transfer: AID->UCM 10.7, UCM->AID 6.7, AID->DFC15 22.7, UCM->DFC15 12.4

## Table 10. Frozen-feature backbone study (5 head seeds, split seed 0)

| backbone | AID->UCM | UCM->AID | AID->DFC15 | UCM->DFC15 |
|---|---|---|---|---|
| imagenet_tresnet_m | 31.97 +- 0.36 | 39.74 +- 0.18 | 38.61 +- 0.19 | 39.18 +- 0.46 |
| imagenet21k_tresnet_m | 28.49 +- 0.16 | 40.02 +- 0.15 | 43.33 +- 0.43 | 42.93 +- 0.39 |
| remoteclip_rn50 | 68.26 +- 0.32 | 68.47 +- 0.15 | 59.69 +- 0.21 | 71.66 +- 0.30 |
| remoteclip_vitb32 | 79.17 +- 0.56 | 75.52 +- 0.18 | 59.79 +- 0.59 | 76.68 +- 0.49 |
| remoteclip_vitl14 | 76.34 +- 0.66 | 75.66 +- 0.09 | 74.74 +- 0.86 | 76.03 +- 0.68 |
| clip_vitb32_control | 59.17 +- 0.31 | 70.41 +- 0.18 | 69.50 +- 0.34 | 77.68 +- 0.26 |
| clip_rn50_control | 62.33 +- 0.57 | 65.01 +- 0.22 | 60.14 +- 0.23 | 74.18 +- 0.30 |
| satmae_vitlarge | 48.84 +- 0.46 | 59.45 +- 0.09 | 49.25 +- 0.22 | 59.52 +- 0.43 |
- clip_vitb32_control minus ImageNet-1K TResNet-M: +27.20, +30.67, +30.89, +38.50
- clip_rn50_control minus ImageNet-1K TResNet-M: +30.37, +25.27, +21.54, +35.00

## Table 16. Contamination audit (RemoteCLIP ViT-B/32 minus CLIP ViT-B/32)

- UC Merced: zero-shot +27.27, frozen probe +20.00
- AID: zero-shot +5.76, frozen probe +5.12
- DFC15: zero-shot +0.42, frozen probe -9.71 / -1.01

## Table 11. Anchor geometry (CLIP ViT-B/32, 17 labels)

| anchors   |   PC1 |   PC2 |   PC3 |   PC1_3 |
|:----------|------:|------:|------:|--------:|
| raw       |  90.2 |   1.3 |   1.1 |    92.7 |
| centered  |  13   |  11.2 |   8   |    32.2 |

 anchors  n_pairs   mean  median    min   max    sd
     raw      136  0.896   0.900  0.804 0.961 0.027
centered      136 -0.061  -0.075 -0.282 0.381 0.110

                     anchor_set  dfc15_zero_shot_mAP_all_images  mean_pairwise_cos
six evaluated categories (used)                           70.95              -0.20
    all seventeen, six retained                           71.56              -0.08
             none (raw anchors)                           61.78               0.90

## Table 12. Prompt families (mean over the three target datasets)

                                     cos_raw_mean  mAP_raw  mAP_centered    gain
prompt_set                                                                      
a photo of {}                               0.838   54.677        62.244   7.568
bare category name                          0.841   54.819        61.562   6.743
three aerial templates + a photo of         0.882   52.039        62.923  10.883
a satellite image of {}                     0.883   45.193        61.041  15.848
three aerial templates (used)               0.896   48.944        62.830  13.886
an aerial photograph containing {}          0.897   52.504        63.485  10.981
a remote sensing image showing {}           0.905   44.588        62.750  18.161

Range of raw mAP 10.23; of centered mAP 2.44

## Table 13. Fusion rule (alpha = 0.30, CLIP ViT-B/32, 20 seeds)

setting  AID->UCM  UCM->AID  AID->DFC15  UCM->DFC15   mean
rule                                                      
zscore      66.80     72.66       75.23       79.51  73.55
minmax      65.76     72.31       75.11       79.60  73.19
rank        65.71     72.49       73.66       78.73  72.65
sigmoid     61.90     70.82       71.65       78.33  70.68
raw         61.64     70.70       71.78       78.40  70.63

Cost of no standardization: 2.92

## Table 14. Per-class AP on the DFC15 transfers

   setting     label  prevalence  zeroshot  probe  clab     d
UCM->DFC15     water       45.57     91.22  93.02 93.69  0.67
UCM->DFC15     grass       49.59     78.28  90.62 90.86  0.24
UCM->DFC15 buildings       45.71     86.34  90.59 91.66  1.08
UCM->DFC15     trees       11.78     49.94  42.13 49.25  7.12
UCM->DFC15      ship       12.33     45.75  74.71 70.01 -4.69
UCM->DFC15      cars       32.19     74.96  78.94 81.57  2.62
AID->DFC15     water       45.57     91.22  90.27 91.89  1.63
AID->DFC15     grass       49.59     78.28  81.93 88.42  6.48
AID->DFC15 buildings       45.71     86.34  85.10 87.93  2.83
AID->DFC15     trees       11.78     49.94  27.19 36.72  9.53
AID->DFC15      ship       12.33     45.75  69.77 62.63 -7.14
AID->DFC15      cars       32.19     74.96  75.33 83.81  8.47

## Proposition 1 measurements

- rho on source: 45.6 to 49.7; on target: 31.6 to 39.0
- target constants instead of source: mean -0.11, worst -0.43
- zero locations: largest |change| 0.000000

## Source-head budget (Section 6.6, 5 seeds)

                                       src_train_mAP  tgt_linear_mAP
setting    backbone            epochs                               
AID->DFC15 clip_vitb32_control 20              94.60           71.39
                               50              95.48           71.81
                               100             96.27           72.09
                               200             97.02           71.94
                               400             97.73           70.82
                               800             98.37           68.71
           remoteclip_vitb32   20              95.76           62.33
                               50              96.49           62.31
                               100             97.17           62.45
                               200             97.83           62.65
                               400             98.41           62.59
                               800             98.90           61.72
AID->UCM   clip_vitb32_control 20              84.69           61.12
                               50              89.49           61.57
                               100             93.38           61.41
                               200             94.79           60.76
                               400             96.21           59.69
                               800             97.54           58.30
           remoteclip_vitb32   20              87.11           76.81
                               50              93.39           78.56
                               100             94.94           79.17
                               200             96.11           79.28
                               400             97.25           78.87
                               800             98.28           77.41
UCM->AID   clip_vitb32_control 20              96.94           69.29
                               50              97.74           69.54
                               100             98.31           69.54
                               200             98.87           69.47
                               400             99.30           69.32
                               800             99.57           69.05
           remoteclip_vitb32   20              97.89           75.07
                               50              98.43           74.89
                               100             98.84           74.67
                               200             99.19           74.47
                               400             99.47           74.16
                               800             99.67           73.66
UCM->DFC15 clip_vitb32_control 20              97.17           78.47
                               50              97.95           78.20
                               100             98.51           77.86
                               200             98.99           77.28
                               400             99.37           76.59
                               800             99.64           75.95
           remoteclip_vitb32   20              98.29           78.94
                               50              98.74           78.69
                               100             99.03           78.44
                               200             99.32           78.03
                               400             99.56           77.49
                               800             99.75           76.63

Oracle gains over alpha = 0.30: global 0.22, per-class 2.53

## Table 15. Label-free per-class estimators vs fixed alpha = 0.30

| estimator | mean delta | arms improved (of 8) | worst arm |
|---|---|---|---|
| rank_agreement | -0.10 | 2 | -0.73 |
| source_validation | -2.34 | 0 | -5.28 |
| shrunk_source_validation | -0.54 | 1 | -1.78 |
| self_training | +0.00 | 1 | -0.00 |
