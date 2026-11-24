## Table 1. Datasets

| dataset | images | cat. | labels/img | shared-6 images | removed |
|---|---|---|---|---|---|
| aid | 3000 | 17 | 5.15 | 2740 | 8.7% (260) |
| ucm | 2100 | 17 | 3.33 | 1726 | 17.8% (374) |
| dfc15 | 3342 | 8 | 2.80 | 2190 | 34.5% (1152) |

## Table 2. Main results, CLIP ViT-B/32 (mAP, mean +- SD over 20 seeds)

| method | AID->UCM | UCM->AID | AID->DFC15 | UCM->DFC15 | mean |
|---|---|---|---|---|---|
| zeroshot_raw | 43.18 +- 1.31 | 50.62 +- 2.20 | 64.19 +- 1.82 | 64.19 +- 1.82 | 55.55 |
| zeroshot | 59.74 +- 1.28 | 63.64 +- 2.14 | 70.73 +- 1.93 | 70.73 +- 1.93 | 66.21 |
| probe | 65.65 +- 1.36 | 72.82 +- 2.67 | 70.57 +- 1.63 | 77.57 +- 1.50 | 71.65 |
| clab_raw | 67.69 +- 1.20 | 73.07 +- 2.41 | 73.93 +- 1.78 | 79.47 +- 1.53 | 73.54 |
| clab_long | 68.44 +- 1.34 | 73.65 +- 2.38 | 72.25 +- 2.28 | 76.13 +- 1.95 | 72.62 |
| clab_a02 | 70.86 +- 1.49 | 74.11 +- 2.50 | 73.46 +- 1.79 | 78.89 +- 1.64 | 74.33 |
| clab_a04 | 70.70 +- 1.30 | 73.20 +- 2.33 | 74.57 +- 1.84 | 78.86 +- 1.79 | 74.33 |
| clab | 71.02 +- 1.45 | 73.85 +- 2.38 | 74.18 +- 1.78 | 79.15 +- 1.69 | 74.55 |

## Table 3. CLAB minus probe, paired over 20 seeds, Holm over 8 contrasts

| backbone | transfer | delta | 95% CI | Holm-adjusted CI | resolved |
|---|---|---|---|---|---|
| clip_vitb32_control | AID->UCM | +5.37 | [+4.97, +5.77] | [+4.81, +5.93] | yes |
| clip_vitb32_control | UCM->AID | +1.03 | [+0.72, +1.33] | [+0.72, +1.33] | yes |
| clip_vitb32_control | AID->DFC15 | +3.61 | [+3.19, +4.02] | [+3.06, +4.16] | yes |
| clip_vitb32_control | UCM->DFC15 | +1.59 | [+1.32, +1.85] | [+1.27, +1.90] | yes |
| remoteclip_vitb32 | AID->UCM | +11.44 | [+11.00, +11.87] | [+10.80, +12.08] | yes |
| remoteclip_vitb32 | UCM->AID | +1.83 | [+1.65, +2.01] | [+1.59, +2.08] | yes |
| remoteclip_vitb32 | AID->DFC15 | +9.29 | [+8.79, +9.79] | [+8.57, +10.01] | yes |
| remoteclip_vitb32 | UCM->DFC15 | +1.73 | [+1.48, +1.98] | [+1.42, +2.04] | yes |

## Table 5. Ablation (mean delta mAP vs full CLAB over 4 transfers x 2 backbones)

- raw anchors (no centering): -3.22
- no source head: -6.77
- no semantic branch: -4.49
- 800-epoch head: -1.32

Centering gain in zero-shot (targets UCM, AID, DFC15): clip UCM +16.56, clip AID +13.02, clip DFC15 +6.53, remoteclip UCM +12.23, remoteclip AID +7.31, remoteclip DFC15 +21.23

## Table 4. Oracle-tuned CLIP adaptation baselines (CLIP ViT-B/32)

| transfer | best baseline (config) | baseline | CLAB | delta [95% CI] |
|---|---|---|---|---|
| AID->UCM | clip_adapter|centred|0.2|0.001|200 | 65.26 | 71.02 | +5.76 [+4.88, +6.64] |
| UCM->AID | clip_adapter|centred|0.2|0.001|200 | 67.27 | 73.85 | +6.57 [+6.09, +7.06] |
| AID->DFC15 | tip_adapter|centred|0.4|1.0|z|None | 72.58 | 74.18 | +1.59 [+1.13, +2.06] |
| UCM->DFC15 | clip_adapter|raw|0.2|0.003|200 | 75.85 | 79.15 | +3.31 [+2.71, +3.90] |

Centering gain inside Tip-Adapter: 8.42 to 13.55; inside CLIP-Adapter: -9.36 to 6.63

## Table 6. Leave-one-transfer-out selection

| held out | alpha | epochs | CLAB | probe | delta [95% CI] |
|---|---|---|---|---|---|
| AID->UCM | 0.30 | 20 | 70.33 | 64.93 | +5.40 [+5.06, +5.74] |
| UCM->AID | 0.35 | 50 | 73.56 | 72.82 | +0.74 [+0.41, +1.07] |
| AID->DFC15 | 0.25 | 50 | 73.86 | 70.57 | +3.29 [+2.92, +3.66] |
| UCM->DFC15 | 0.30 | 50 | 79.15 | 77.57 | +1.59 [+1.32, +1.85] |

## Table 7. Sensitivity to alpha (50 epochs) and to the budget (alpha = 0.30)

| alpha | AID->UCM | UCM->AID | AID->DFC15 | UCM->DFC15 | mean |
|---|---|---|---|---|---|
| 0.05 | 67.93 | 73.43 | 71.50 | 77.96 | 72.70 |
| 0.10 | 69.60 | 73.82 | 72.30 | 78.32 | 73.51 |
| 0.15 | 70.46 | 74.04 | 73.02 | 78.63 | 74.04 |
| 0.20 | 70.86 | 74.11 | 73.46 | 78.89 | 74.33 |
| 0.25 | 71.00 | 74.04 | 73.86 | 79.12 | 74.50 |
| 0.30 | 71.02 | 73.85 | 74.18 | 79.15 | 74.55 |
| 0.35 | 70.93 | 73.56 | 74.44 | 79.06 | 74.50 |
| 0.40 | 70.70 | 73.20 | 74.57 | 78.86 | 74.33 |
| 0.45 | 70.34 | 72.76 | 74.62 | 78.46 | 74.05 |
| 0.50 | 69.86 | 72.22 | 74.56 | 78.04 | 73.67 |
- budget 20: mean 74.38
- budget 50: mean 74.55
- budget 100: mean 74.41
- budget 200: mean 74.01

## Table 8. Out-of-vocabulary imagery restored in the target (alpha = 0.30)

| transfer | OOV % | probe | CLAB | delta [95% CI] |
|---|---|---|---|---|
| AID->UCM | 17.2 | 74.75 | 77.53 | +2.78 [+2.40, +3.17] |
| AID->DFC15 | 34.4 | 68.07 | 70.97 | +2.91 [+2.47, +3.35] |
| UCM->AID | 8.7 | 83.70 | 83.93 | +0.23 [-0.04, +0.49] |
| UCM->DFC15 | 34.4 | 74.16 | 75.28 | +1.12 [+0.79, +1.45] |
| DFC15->AID | 8.7 | 77.03 | 80.26 | +3.23 [+2.80, +3.66] |
| DFC15->UCM | 17.2 | 73.26 | 74.32 | +1.06 [+0.64, +1.48] |

Mean over six: +1.89; worst direction at alpha 0.20 / 0.30 / 0.40: +0.45 / +0.23 / -0.33
Standard transfers: mean gain over the probe varies by 0.22 for alpha in [0.20, 0.40]

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
| CPLG | ResNet-50 | 71.8 | 75.6 | 56.8 | 57.5 | Huang, Ma and Huang, IEEE JSTARS 17 (2024) 20344-20354, Tables II-V (mean of 10 runs) |
| CPLG + CDAN | ResNet-50 | 73.9 | 76.2 | 58.4 | 61.6 | Huang, Ma and Huang, IEEE JSTARS 17 (2024) 20344-20354, Tables II-V (mean of 10 runs) |
| CLAB (ours) | CLIP ViT-B/32 | 71.0 | 73.8 | 74.2 | 79.2 | |

Margin over best published: AID->UCM -2.8, UCM->AID -2.3, AID->DFC15 +8.5, UCM->DFC15 +2.7
Spread among published methods per transfer: AID->UCM 21.4, UCM->AID 25.7, AID->DFC15 22.7, UCM->DFC15 19.0

## Table 10. Frozen-feature backbone study (5 head seeds, split seed 0)

| backbone | AID->UCM | UCM->AID | AID->DFC15 | UCM->DFC15 |
|---|---|---|---|---|
| imagenet_tresnet_m | 31.97 +- 0.36 | 39.74 +- 0.18 | 38.61 +- 0.19 | 39.18 +- 0.46 |
| imagenet21k_tresnet_m | 28.49 +- 0.16 | 40.02 +- 0.15 | 43.33 +- 0.43 | 42.93 +- 0.39 |
| remoteclip_rn50 | 68.26 +- 0.32 | 68.47 +- 0.15 | 59.69 +- 0.21 | 71.66 +- 0.30 |
| remoteclip_vitb32 | 79.17 +- 0.56 | 75.52 +- 0.18 | 59.79 +- 0.59 | 76.68 +- 0.49 |
| remoteclip_vitl14 | 76.34 +- 0.66 | 75.66 +- 0.09 | 74.74 +- 0.86 | 76.03 +- 0.68 |
| clip_rn50_control | 62.33 +- 0.57 | 65.01 +- 0.22 | 60.14 +- 0.23 | 74.18 +- 0.30 |
| satmae_vitlarge | 48.84 +- 0.46 | 59.45 +- 0.09 | 49.25 +- 0.22 | 59.52 +- 0.43 |
| clip_vitb32_control | 64.29 +- 0.22 | 72.57 +- 0.18 | 70.61 +- 0.31 | 76.28 +- 0.32 |
- clip_vitb32_control minus ImageNet-1K TResNet-M: +32.32, +32.83, +32.00, +37.10
- clip_rn50_control minus ImageNet-1K TResNet-M: +30.37, +25.27, +21.54, +35.00

## Table 16. Contamination audit (RemoteCLIP ViT-B/32 minus CLIP ViT-B/32)

- UC Merced: zero-shot +24.60, frozen probe +14.88
- AID: zero-shot +5.87, frozen probe +2.95
- DFC15: zero-shot +0.78, frozen probe -10.82 / +0.40

## Table 11. Anchor geometry (CLIP ViT-B/32, 17 labels)

| anchors   |   PC1 |   PC2 |   PC3 |   PC1_3 |
|:----------|------:|------:|------:|--------:|
| raw       |  87   |   1.8 |   1.5 |    90.3 |
| centered  |  13.2 |  12.4 |   8.3 |    33.9 |

 anchors  n_pairs   mean  median    min   max    sd
     raw      136  0.861   0.862  0.742 0.952 0.035
centered      136 -0.061  -0.083 -0.308 0.435 0.122

                     anchor_set  dfc15_zero_shot_mAP_all_images  mean_pairwise_cos
six evaluated categories (used)                           70.75              -0.20
    all seventeen, six retained                           70.52              -0.09
             none (raw anchors)                           64.42               0.87

## Table 12. Prompt families (mean over the three target datasets)

                                     cos_raw_mean  mAP_raw  mAP_centered    gain
prompt_set                                                                      
a photo of {}                               0.806   53.728        62.887   9.159
bare category name                          0.819   53.363        63.054   9.690
a satellite image of {}                     0.849   45.746        60.697  14.951
three aerial templates + a photo of         0.852   52.713        63.697  10.984
three aerial templates (used)               0.863   51.187        63.583  12.395
an aerial photograph containing {}          0.864   53.324        63.913  10.589
a remote sensing image showing {}           0.869   49.291        64.279  14.988

Range of raw mAP 7.98; of centered mAP 3.58

## Table 13. Fusion rule (alpha = 0.30, CLIP ViT-B/32, 20 seeds)

setting  AID->UCM  UCM->AID  AID->DFC15  UCM->DFC15   mean
rule                                                      
zscore      71.02     73.85       74.18       79.15  74.55
minmax      69.71     73.54       74.10       78.96  74.07
rank        69.94     73.63       72.55       77.94  73.51
sigmoid     66.22     73.03       70.52       77.55  71.83
raw         65.94     72.95       70.75       77.64  71.82

Cost of no standardization: 2.73

## Table 14. Per-class AP on the DFC15 transfers

   setting     label  prevalence  zeroshot  probe  clab     d
UCM->DFC15     water       45.57     91.58  94.29 94.35  0.06
UCM->DFC15     grass       49.59     76.63  89.49 90.04  0.55
UCM->DFC15 buildings       45.71     86.60  89.44 90.65  1.21
UCM->DFC15     trees       11.78     53.56  38.46 47.69  9.23
UCM->DFC15      ship       12.33     42.58  74.84 71.35 -3.49
UCM->DFC15      cars       32.19     73.41  78.89 80.85  1.96
AID->DFC15     water       45.57     91.58  91.56 92.48  0.92
AID->DFC15     grass       49.59     76.63  80.08 85.94  5.85
AID->DFC15 buildings       45.71     86.60  83.10 86.37  3.26
AID->DFC15     trees       11.78     53.56  23.00 34.63 11.63
AID->DFC15      ship       12.33     42.58  71.40 62.37 -9.03
AID->DFC15      cars       32.19     73.41  74.27 83.28  9.01

## Proposition 1 measurements

- rho on source: 44.0 to 48.2; on target: 27.7 to 37.2
- target constants instead of source: mean -0.14, worst -0.62
- zero locations: largest |change| 0.000000

## Source-head budget (Section 6.6, 5 seeds)

                                       src_train_mAP  tgt_linear_mAP
setting    backbone            epochs                               
AID->DFC15 clip_vitb32_control 20              94.80           70.21
                               50              95.61           70.98
                               100             96.33           71.60
                               200             97.03           71.55
                               400             97.69           70.61
                               800             98.31           68.66
           remoteclip_vitb32   20              95.76           62.33
                               50              96.49           62.31
                               100             97.17           62.45
                               200             97.83           62.65
                               400             98.41           62.59
                               800             98.90           61.72
AID->UCM   clip_vitb32_control 20              86.24           64.91
                               50              90.34           65.68
                               100             93.52           65.43
                               200             94.82           64.41
                               400             96.17           62.94
                               800             97.45           61.34
           remoteclip_vitb32   20              87.11           76.81
                               50              93.39           78.56
                               100             94.94           79.17
                               200             96.11           79.28
                               400             97.25           78.87
                               800             98.28           77.41
UCM->AID   clip_vitb32_control 20              97.08           72.01
                               50              97.79           71.93
                               100             98.34           71.67
                               200             98.85           71.41
                               400             99.26           71.13
                               800             99.53           70.77
           remoteclip_vitb32   20              97.89           75.07
                               50              98.43           74.89
                               100             98.84           74.67
                               200             99.19           74.47
                               400             99.47           74.16
                               800             99.67           73.66
UCM->DFC15 clip_vitb32_control 20              97.37           77.91
                               50              98.08           77.52
                               100             98.60           76.90
                               200             99.05           76.22
                               400             99.40           75.27
                               800             99.65           74.38
           remoteclip_vitb32   20              98.29           78.94
                               50              98.74           78.69
                               100             99.03           78.44
                               200             99.32           78.03
                               400             99.56           77.49
                               800             99.75           76.63

Oracle gains over alpha = 0.30: global 0.26, per-class 2.88

## Table 15. Label-free per-class estimators vs fixed alpha = 0.30

| estimator | mean delta | arms improved (of 8) | worst arm |
|---|---|---|---|
| rank_agreement | -0.03 | 2 | -1.08 |
| source_validation | -2.20 | 0 | -5.28 |
| shrunk_source_validation | -0.50 | 2 | -1.78 |
| self_training | -0.00 | 2 | -0.00 |
