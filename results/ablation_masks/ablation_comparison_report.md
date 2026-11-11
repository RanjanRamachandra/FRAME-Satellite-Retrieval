# FRAME Mask Ablation — Comparison Report

## Overview

- Configurations evaluated: 9
- Test queries: 56
- Database size: 112

> **Interpretation note:** These metrics compare retrieval behaviour,
> not retrieval quality. Relevance requires human annotations.
> Use compute_retrieval_metrics.py after collecting labels.

## Retrieval Diversity and Temporal Separation

| configuration    | shape     | size   |   unique_top1 |   mean_temporal_hours |   median_temporal_hours |
|:-----------------|:----------|:-------|--------------:|----------------------:|------------------------:|
| square_small     | square    | small  |            30 |                369.21 |                  360    |
| rectangle_small  | rectangle | small  |            26 |                372.37 |                  355.5  |
| square_medium    | square    | medium |            24 |                376.82 |                  351    |
| circle_small     | circle    | small  |            29 |                379.76 |                  361.5  |
| circle_large     | circle    | large  |            23 |                389.46 |                  361.5  |
| circle_medium    | circle    | medium |            28 |                389.78 |                  358.5  |
| rectangle_medium | rectangle | medium |            26 |                402.48 |                  400.5  |
| square_large     | square    | large  |            26 |                426.21 |                  420    |
| rectangle_large  | rectangle | large  |            18 |                447.48 |                  487.42 |

## Agreement with Baseline FRAME (square, medium)

| configuration    |   top1_agreement_vs_baseline |   mean_top5_overlap_vs_baseline |   n_queries |
|:-----------------|-----------------------------:|--------------------------------:|------------:|
| circle_large     |                       0.25   |                          1.6071 |          56 |
| square_medium    |                       0.2321 |                          1.375  |          56 |
| rectangle_medium |                       0.1964 |                          1.5893 |          56 |
| circle_medium    |                       0.1786 |                          1.5714 |          56 |
| circle_small     |                       0.1607 |                          1.2857 |          56 |
| square_small     |                       0.125  |                          1.3571 |          56 |
| rectangle_small  |                       0.0893 |                          1.1429 |          56 |
| square_large     |                       0.0357 |                          1.1429 |          56 |
| rectangle_large  |                       0      |                          0.2679 |          56 |

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