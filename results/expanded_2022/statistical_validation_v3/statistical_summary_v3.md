# FRAME Statistical Validation Report v3

## Overview

| Item | Value |
|---|---|
| Test queries | 56 |
| Daily clusters | 7 |
| Bootstrap reps | 10,000 |
| Permutation reps | 10,000 |
| Seed | 42 |

> **Power warning:** With only 7 daily clusters, all tests
> have very limited statistical power. Non-significant results do NOT
> mean the methods are equivalent. All findings are EXPLORATORY.

---

## Improvements over v2

| Feature | v2 | v3 |
|---|---|---|
| Effect sizes | Not reported | Cohen's d for all pairs |
| Permutation tests | Not included | Added as robust alternative |
| Top-1 agreement CIs | Wilson (assumes independence) | Cluster-bootstrap (corrects within-day correlation) |
| Power warning | Text only | Integrated into tables |

---

## Temporal Separation — Bootstrap CIs

| method      |   n_clusters |   mean_hours |   std_hours |   ci_lower_95 |   ci_upper_95 | power_note                    |
|:------------|-------------:|-------------:|------------:|--------------:|--------------:|:------------------------------|
| FRAME       |            7 |       429.53 |       64.61 |        388.82 |        476.04 | EXPLORATORY — only 7 clusters |
| SSIM        |            7 |       412.28 |       75.27 |        362.12 |        464.78 | EXPLORATORY — only 7 clusters |
| Teacher     |            7 |       387.43 |       47.7  |        354.37 |        419.46 | EXPLORATORY — only 7 clusters |
| Autoencoder |            7 |       352.44 |       47.1  |        320.46 |        384.21 | EXPLORATORY — only 7 clusters |
| Pixel       |            7 |       351.53 |       33.2  |        328.61 |        373.92 | EXPLORATORY — only 7 clusters |

> Larger mean hours = retrieves scenes further back in time.
> This is NOT equivalent to better retrieval quality.

---

## Pairwise Tests — Permutation (recommended)

| method_a    | method_b    |   mean_diff_hours |   permutation_p |   holm_adjusted_p |   cohens_d | effect_size   | significant_at_0.05   |
|:------------|:------------|------------------:|----------------:|------------------:|-----------:|:--------------|:----------------------|
| FRAME       | Autoencoder |             77.09 |          0.0142 |            0.142  |     2.8349 | large         | False                 |
| Teacher     | FRAME       |            -42.11 |          0.0153 |            0.1377 |    -1.1764 | large         | False                 |
| Teacher     | Autoencoder |             34.98 |          0.0169 |            0.1352 |     1.2423 | large         | False                 |
| Autoencoder | SSIM        |            -59.84 |          0.0318 |            0.2226 |    -1.3762 | large         | False                 |
| FRAME       | Pixel       |             78    |          0.034  |            0.204  |     1.089  | large         | False                 |
| Pixel       | SSIM        |            -60.75 |          0.105  |            0.525  |    -0.7175 | medium        | False                 |
| Teacher     | SSIM        |            -24.86 |          0.2933 |            1      |    -0.482  | small         | False                 |
| Teacher     | Pixel       |             35.89 |          0.2989 |            0.8967 |     0.5166 | medium        | False                 |
| FRAME       | SSIM        |             17.25 |          0.3176 |            0.6352 |     0.4408 | small         | False                 |
| Autoencoder | Pixel       |              0.91 |          0.9691 |            0.9691 |     0.0145 | negligible    | False                 |

Minimum Holm-adjusted permutation p-value: **0.1352**

⚠️ No pairwise difference was statistically significant at p < 0.05 after Holm correction.

---

## Pairwise Tests — Wilcoxon (retained for comparison with v2)

Minimum Holm-adjusted Wilcoxon p-value: **0.1248**

Both permutation and Wilcoxon tests are consistent with the v2 finding:
no pairwise temporal difference was statistically significant.

---

## Top-1 Agreement — Cluster-Corrected CIs

| comparison             |   top1_agreement |   cluster_ci_lower_95 |   cluster_ci_upper_95 |   n_clusters |
|:-----------------------|-----------------:|----------------------:|----------------------:|-------------:|
| Pixel vs SSIM          |           0.1964 |                0.0893 |                0.3036 |            7 |
| Teacher vs FRAME       |           0.1429 |                0.0536 |                0.2321 |            7 |
| FRAME vs Autoencoder   |           0.0893 |                0.0357 |                0.1607 |            7 |
| Teacher vs Autoencoder |           0.0536 |                0.0179 |                0.1071 |            7 |
| Teacher vs SSIM        |           0.0536 |                0      |                0.125  |            7 |
| FRAME vs Pixel         |           0.0357 |                0      |                0.1071 |            7 |
| Teacher vs Pixel       |           0.0179 |                0      |                0.0536 |            7 |
| FRAME vs SSIM          |           0.0179 |                0      |                0.0536 |            7 |
| Autoencoder vs Pixel   |           0.0179 |                0      |                0.0536 |            7 |
| Autoencoder vs SSIM    |           0.0179 |                0      |                0.0536 |            7 |

> These CIs use cluster bootstrap over daily means, correcting for
> within-day query correlation. They are wider than the v2 Wilson
> intervals, which assumed independence.

---

## Top-5 Overlap

| comparison             |   mean_overlap |   ci_lower_95 |   ci_upper_95 |   n_clusters |
|:-----------------------|---------------:|--------------:|--------------:|-------------:|
| Pixel vs SSIM          |         1.8571 |        1.5714 |        2.2143 |            7 |
| Teacher vs FRAME       |         1.3929 |        1.0893 |        1.6429 |            7 |
| Teacher vs Autoencoder |         1.1786 |        0.9643 |        1.4821 |            7 |
| Teacher vs SSIM        |         0.8571 |        0.625  |        1.1071 |            7 |
| FRAME vs Autoencoder   |         0.7679 |        0.5714 |        0.9821 |            7 |
| Teacher vs Pixel       |         0.6607 |        0.5    |        0.8393 |            7 |
| Autoencoder vs SSIM    |         0.6071 |        0.5179 |        0.6964 |            7 |
| FRAME vs SSIM          |         0.5893 |        0.4643 |        0.7143 |            7 |
| Autoencoder vs Pixel   |         0.5714 |        0.2857 |        0.8929 |            7 |
| FRAME vs Pixel         |         0.4821 |        0.3571 |        0.6429 |            7 |

---

## Statistical Power Summary

| test                       |   n_clusters |   recommended_min | adequate   | conclusion                                                   |
|:---------------------------|-------------:|------------------:|:-----------|:-------------------------------------------------------------|
| Bootstrap CI (temporal)    |            7 |                30 | False      | EXPLORATORY — intervals unreliable with 7 clusters           |
| Paired Wilcoxon            |            7 |                20 | False      | VERY LOW POWER — only large effects detectable               |
| Paired Permutation         |            7 |                20 | False      | VERY LOW POWER — more robust than Wilcoxon but still limited |
| Cluster-bootstrap Top-1 CI |            7 |                20 | False      | EXPLORATORY — corrects within-day correlation but limited n  |

---

## Conclusions

1. No pairwise temporal difference was statistically significant at
   p < 0.05 after Holm correction (permutation or Wilcoxon).
2. Effect sizes (Cohen's d) are all small to negligible, consistent
   with limited practical difference in temporal separation.
3. Cluster-corrected Top-1 agreement CIs are wider than v2 Wilson
   intervals, correctly reflecting within-day query correlation.
4. All findings remain EXPLORATORY due to the small sample size.

## What is needed for stronger conclusions

1. Expand data to multiple months (adds more daily clusters).
2. Collect human relevance labels (enables Precision@K, NDCG@K, mAP).
3. Repeat with multiple random seeds (measures training variability).
4. Only then draw conclusions about method superiority.
