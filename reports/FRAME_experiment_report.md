# FRAME Satellite Image Retrieval — Experiment Report

## 1. Research objective

Evaluate FRAME (Focus Region Aware Masking Embedding) for retrieving historical Himawari-8 satellite images from partially masked query images.

This is a modified research reproduction using Himawari-8 data, not an exact reproduction of the original COMS experiment.

## 2. Dataset

- Source: NOAA Himawari-8 HSD
- Period: 15–21 January 2022
- Bands: B07, B08, B13
- Processed image size: 1000 × 1000
- Model input size: 200 × 200
- Total observations: 56

Chronological split:
- Training: 45 observations
- Development: 11 observations
- A 3-hour chronological gap was used.

The 11 development observations were also used during validation and checkpoint selection. They are not an independent final test set.

## 3. Model architecture

### Original encoder
- SE residual encoder
- Input: three satellite channels
- Embedding dimension: 512
- Global average pooling
- Trained using image reconstruction loss.

### Masked encoder
- Independently initialized SE residual autoencoder
- Random square masks with side lengths 40–100 pixels
- Reconstruction objective.

### FRAME
- Original encoder and masked encoder are frozen.
- Mask encoder processes binary mask geometry.
- Original image embedding is multiplied element-wise by mask embedding.
- Projection head predicts the masked encoder embedding.
- Training objective: cosine embedding similarity.

Important architectural limitation: the mask encoder receives mask geometry, not the visible satellite pixels directly.

## 4. Retrieval experiment

- Database: 45 training observations.
- Queries: 11 development observations.
- Four methods: FRAME, Teacher, Pixel, SSIM.
- Each method returns Top-5 candidates.
- A 12-hour temporal exclusion was applied.
- Candidates at or within 12 hours of query time were excluded.

Total ranking entries: 220.

## 5. Pairwise Top-5 overlap

| Comparison | Mean overlap |
|---|---:|
| Pixel vs SSIM | 3.18 / 5 |
| Teacher vs FRAME | 2.73 / 5 |
| SSIM vs FRAME | 2.55 / 5 |
| Pixel vs FRAME | 2.18 / 5 |
| SSIM vs Teacher | 2.09 / 5 |
| Pixel vs Teacher | 1.91 / 5 |

Pixel and SSIM show the highest ranking overlap. FRAME and Teacher also show substantial overlap, but their rankings are not identical.

## 6. Human relevance annotation

- Queries: 11
- Unique query-candidate pairs rated: 118
- Rating scale:
  - 0: Not relevant
  - 1: Partially similar
  - 2: Strong match
- Observed distribution:
  - Rating 0: 0
  - Rating 1: 118
  - Rating 2: 0

All 220 ranking entries were covered by the 118 unique pair labels.

| Method | Top-5 entries | Judged entries | Unique candidates | Mean rating |
|---|---:|---:|---:|---:|
| FRAME | 55 | 55 | 28 | 1.0 |
| Pixel | 55 | 55 | 23 | 1.0 |
| SSIM | 55 | 55 | 28 | 1.0 |
| Teacher | 55 | 55 | 29 | 1.0 |

Because every candidate received the middle rating, these annotations cannot distinguish the relevance performance of the four methods.

This is a pilot annotation result, not evidence of 100% retrieval relevance.

## 7. Main limitations

1. Small dataset: 56 observations.
2. Only 11 development queries.
3. Development data was used for validation and model selection.
4. Human relevance labels have no variation.
5. One annotator was used.
6. Teacher imitation does not establish independent semantic retrieval quality.
7. Temporal similarity is not equivalent to meteorological relevance.
8. The current FRAME mask encoder uses mask geometry rather than visible-region image content.

## 8. Conclusions

The implementation demonstrates a working experimental pipeline for masked satellite image retrieval.

FRAME produces rankings that partially overlap with Teacher rankings and the visual baselines. However, current human annotations cannot establish which method retrieves more relevant cloud patterns.

No claim of FRAME superiority is justified by this experiment.

## 9. Recommended next experiments

1. Calibrate human relevance labels using clear examples of ratings 0, 1 and 2.
2. Conduct a new blinded annotation round with multiple annotators if possible.
3. Use a larger, independent chronological test set.
4. Report Precision@5, NDCG@5 and Recall@5 only when relevance labels have sufficient variation and the candidate pool is properly defined.
5. Compare geometry-only mask conditioning with visible-image-aware mask conditioning.
6. Evaluate robustness across different dates and weather conditions.

## 10. Reproducibility artifacts

- `results/temporal_exclusion_12h_rankings.csv`
- `results/temporal_exclusion_12h_top1.csv`
- `results/human_relevance_labels.csv`
- `reports/human_relevance_method_summary.csv`

