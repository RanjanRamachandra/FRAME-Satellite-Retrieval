# FRAME: Statistical Validation Report

## 1. Objective

This analysis examines retrieval behaviour across five methods:

- FRAME
- Teacher
- Autoencoder
- Pixel
- SSIM

The objective is to quantify temporal separation, ranking agreement, and Top-5 overlap using the independent final-test dataset.

This analysis does not establish semantic relevance or retrieval accuracy because independently verified relevance labels are unavailable.

## 2. Evaluation Dataset

| Property | Value |
|---|---:|
| Training observations | 112 |
| Validation observations | 56 |
| Final-test observations | 56 |
| Retrieval methods | 5 |
| Query-method results | 280 |
| Pairwise method combinations | 10 |
| Pairwise comparison records | 560 |
| Test period | January 22–28, 2022 |

The 56 test queries span seven consecutive days.

## 3. Statistical Methodology

### 3.1 Bootstrap confidence intervals

A day-cluster bootstrap with 10,000 resamples was used to estimate 95% confidence intervals.

The seven daily averages were resampled as clusters rather than treating all 56 observations as independent.

Metrics:

- Top-1 temporal separation.
- Pairwise Top-5 overlap.

### 3.2 Paired Wilcoxon signed-rank tests

Paired comparisons were performed using seven daily average temporal-separation values.

All ten method combinations were tested.

Holm correction was applied to control the family-wise error rate.

### 3.3 Wilson confidence intervals

Wilson 95% confidence intervals were calculated for pairwise Top-1 agreement proportions.

These intervals use individual queries and do not account for possible within-day dependence.

## 4. Temporal Separation Results

| Method | Mean (hours) | 95% CI lower | 95% CI upper |
|---|---:|---:|---:|
| Teacher | 387.43 | 354.37 | 419.46 |
| FRAME | 429.53 | 388.82 | 476.04 |
| Autoencoder | 352.44 | 320.46 | 384.21 |
| Pixel | 351.53 | 328.61 | 373.92 |
| SSIM | 412.28 | 362.12 | 464.78 |

FRAME produced the highest observed mean temporal separation.

This is a temporal retrieval characteristic, not evidence of improved semantic relevance.

## 5. Paired Statistical Tests

All ten pairwise temporal comparisons were evaluated.

The smallest Holm-adjusted p-value was 0.1562.

No comparison achieved statistical significance at the 0.05 level after multiple-comparison correction.

Consequently, the observed temporal differences should be treated as descriptive rather than statistically established differences.

## 6. Pairwise Top-1 Agreement

| Method pair | Agreements | Agreement |
|---|---:|---:|
| Autoencoder–Pixel | 1/56 | 1.79% |
| Autoencoder–SSIM | 1/56 | 1.79% |
| FRAME–Autoencoder | 5/56 | 8.93% |
| FRAME–Pixel | 2/56 | 3.57% |
| FRAME–SSIM | 1/56 | 1.79% |
| Pixel–SSIM | 11/56 | 19.64% |
| Teacher–Autoencoder | 3/56 | 5.36% |
| Teacher–FRAME | 8/56 | 14.29% |
| Teacher–Pixel | 1/56 | 1.79% |
| Teacher–SSIM | 3/56 | 5.36% |

The results demonstrate substantial differences in Top-1 retrieval selections across methods.

## 7. Pairwise Top-5 Overlap

| Method pair | Mean overlap |
|---|---:|
| Autoencoder–Pixel | 0.571 |
| Autoencoder–SSIM | 0.607 |
| FRAME–Autoencoder | 0.768 |
| FRAME–Pixel | 0.482 |
| FRAME–SSIM | 0.589 |
| Pixel–SSIM | 1.857 |
| Teacher–Autoencoder | 1.179 |
| Teacher–FRAME | 1.393 |
| Teacher–Pixel | 0.661 |
| Teacher–SSIM | 0.857 |

Pixel and SSIM exhibited the highest Top-5 overlap.

FRAME and Teacher shared an average of 1.393 observations among their Top-5 results.

## 8. Interpretation

The analysis supports the following observations:

1. FRAME produced the highest observed mean temporal separation.
2. No paired temporal comparison was statistically significant after Holm correction.
3. Pixel and SSIM showed the highest Top-1 agreement.
4. Pixel and SSIM also showed the highest average Top-5 overlap.
5. Retrieval methods frequently selected different database observations.

These findings do not demonstrate that any method is more semantically relevant than another.

## 9. Limitations

- Only 56 independent test queries were evaluated.
- The test period covers seven consecutive days.
- Seven daily clusters provide limited statistical power.
- Temporal and ranking agreement are indirect retrieval characteristics.
- Independently verified relevance labels are unavailable.
- Wilson intervals do not account for within-day query dependence.
- The human visual review involved one annotator, and all 250 ratings were identical.
- Results are limited to one geographic region, one month, and three Himawari-8 spectral bands.

## 10. Future Research

Recommended improvements:

1. Expand evaluation across multiple months and seasons.
2. Include additional geographic regions.
3. Recruit multiple independent human annotators.
4. Establish a clear relevance-rating rubric.
5. Measure inter-annotator agreement.
6. Evaluate Recall@K, Precision@K, NDCG@K and mean Average Precision when relevance labels become available.
7. Repeat statistical testing using a larger number of independent temporal clusters.
8. Investigate the relationship between temporal separation and visual relevance.

## 11. Reproducibility

Statistical analysis script:

`scripts/statistical_validation_v2.py`

Input files:

- `results/expanded_2022/final_test_per_query.csv`
- `results/expanded_2022/final_test_pairwise_comparison.csv`

Output directory:

`results/expanded_2022/statistical_validation/`

Generated reports:

- `temporal_bootstrap_ci.csv`
- `temporal_paired_tests.csv`
- `top1_agreement_wilson_ci.csv`
- `top5_overlap_bootstrap_ci.csv`

Random seed: 42

Bootstrap repetitions: 10,000

---

**Conclusion:** The statistical analysis provides exploratory evidence about differences in retrieval behaviour. Larger datasets and independently established relevance labels are necessary before making claims about retrieval quality or FRAME superiority.
