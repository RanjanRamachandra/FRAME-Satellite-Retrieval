# FRAME Mask Ablation — Comparison Report

## Overview

- Configurations evaluated: 4
- Test queries: 56
- Database size: 112

> **Interpretation note:** These metrics compare retrieval behaviour,
> not retrieval quality. Relevance requires human annotations.
> Use compute_retrieval_metrics.py after collecting labels.

## Retrieval Diversity and Temporal Separation

| configuration   | shape   | size   |   unique_top1 |   mean_temporal_hours |   median_temporal_hours |
|:----------------|:--------|:-------|--------------:|----------------------:|------------------------:|
| square_small    | square  | small  |            30 |                369.21 |                   360   |
| circle_small    | circle  | small  |            22 |                373.34 |                   358.5 |
| square_medium   | square  | medium |            24 |                377.62 |                   351   |
| square_large    | square  | large  |            26 |                426.21 |                   420   |

## Agreement with Baseline FRAME (square, medium)

| configuration   |   top1_agreement_vs_baseline |   mean_top5_overlap_vs_baseline |   n_queries |
|:----------------|-----------------------------:|--------------------------------:|------------:|
| square_medium   |                       0.2321 |                          1.375  |          56 |
| circle_small    |                       0.1607 |                          1.2857 |          56 |
| square_small    |                       0.125  |                          1.3571 |          56 |
| square_large    |                       0.0357 |                          1.1429 |          56 |

## Interpretation

- **Unique Top-1 matches**: higher = more diverse retrieval.
  Does NOT mean better relevance.
- **Temporal separation**: larger = retrieves scenes further back in time.
  Does NOT mean better relevance.
- **Top-1 agreement vs baseline**: how often this config agrees with
  the baseline square-medium FRAME. Agreement ≠ correctness.

## Next Steps

1. Collect human relevance labels using create_improved_visual_review.py
2. Verify inter-annotator agreement with inter_annotator_agreement.py
3. Compute Precision@K, Recall@K, NDCG@K with compute_retrieval_metrics.py
4. Only then compare configurations on retrieval quality.