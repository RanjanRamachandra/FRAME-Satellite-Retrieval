# CHANGELOG

All notable changes to this project are documented here.

---

## [v1.2.0] — 2024 — Improvements: Evaluation, Ablations, Statistics

### Added

#### Phase 1 — Evaluation Foundation
- `scripts/compute_retrieval_metrics.py`
  - Computes Precision@K, Recall@K, NDCG@K and mAP once human labels exist.
  - Supports K = 1, 3, 5, 10.
  - Reads from `results/expanded_2022/annotations/annotated_relevance.csv`.
  - Outputs per-query and method-level summaries.

- `scripts/inter_annotator_agreement.py`
  - Computes Cohen's Kappa (pairwise) and Krippendorff's Alpha (all annotators).
  - Identifies high-disagreement cases for rubric refinement.
  - Merges annotator ratings into `annotated_relevance.csv` for metrics.

- `scripts/create_improved_visual_review.py`
  - Generates blinded review sheets with a clear 4-level rubric (0–3).
  - Produces separate rating templates per annotator.
  - Randomises candidate order independently per annotator.
  - Keeps method keys hidden until after rating is complete.
  - Includes `merge_ratings.py` helper for combining completed templates.

#### Phase 3 — Model Ablations
- `scripts/train_ablation_masks.py`
  - Trains FRAME variants across 3 mask shapes × 3 mask sizes = 9 configs.
  - Shapes: square (baseline), circle, rectangle.
  - Sizes: small (20–40 px), medium (40–80 px), large (80–120 px).
  - Saves best checkpoint per configuration.

- `scripts/evaluate_ablation_results.py`
  - Evaluates all ablation checkpoints on the same test set.
  - Compares Top-1 diversity, temporal separation and agreement with baseline.
  - Outputs comparison report and CSV summaries.

#### Phase 4 — Statistical Improvements
- `scripts/statistical_validation_v3.py`
  - Adds Cohen's d effect sizes for all pairwise comparisons.
  - Adds paired permutation tests as a robust alternative to Wilcoxon.
  - Replaces Wilson intervals with cluster-bootstrap CIs for Top-1 agreement,
    correcting the within-day correlation assumption violation in v2.
  - Integrates power warnings directly into output tables.
  - Outputs `statistical_summary_v3.md` with full findings.

#### Documentation
- `CHANGELOG.md` — This file.
- `CONTRIBUTING.md` — Contribution guidelines.

### Changed
- `README.md`
  - Removed duplicate Section 6 (Human visual evaluation).
  - Fixed section numbering (sections now 1–13, no gaps).
  - Added research integrity disclaimer at top.
  - Added explicit warning: larger temporal separation ≠ better retrieval.
  - Added band selection scientific rationale table.
  - Added plain-language FRAME intuition paragraph.
  - Added training results table with loss-type annotations.
  - Strengthened visual pilot section: clearly labelled non-discriminative.

- `scripts/statistical_validation_v2.py`
  - Added top-of-file docstring with n=7 power warnings.
  - Added inline comment on Wilson interval assumption.

- `scripts/evaluate_expanded_final_test.py`
  - Added docstring with evaluation design and metric interpretation notes.

- `reports/expanded_2022_final_test_report.md`
  - Strengthened visual pilot conclusion: "produced no discriminative data".
  - Added explicit temporal separation warning.

- `reports/statistical_validation_report.md`
  - Added power warning callouts after each test section.
  - Added power summary table (n=7 vs recommended minimums).
  - Added temporal separation warning.

### Fixed
- `.gitignore` — Added `results/expanded_2022/final_test_rankings.csv`
  (1.8 MB file now excluded; regenerate by running evaluate_expanded_final_test.py).

---

## [v1.1.0] — 2024 — Expanded 28-day Dataset and Final Test

### Added
- Expanded from 7-day to 28-day dataset (January 1–28, 2022, 224 observations).
- `scripts/download_expanded_28days.sh`
- `scripts/decompress_expanded_2022.sh`
- `scripts/process_expanded_2022.py`
- `scripts/prepare_expanded_stats.py`
- `scripts/expanded_satellite_dataset.py`
- `scripts/train_expanded_autoencoder.py`
- `scripts/train_expanded_masked_encoder.py`
- `scripts/train_expanded_frame.py`
- `scripts/evaluate_expanded_final_test.py`
- `scripts/create_blind_visual_review.py`
- `scripts/statistical_validation_v2.py`
- Blinded visual review pilot (10 queries, 250 ratings — all rated 2).
- Statistical validation: bootstrap CIs, Wilcoxon, Holm correction.

### Changed
- Split: train 112 / val 56 / test 56 (chronological).
- Normalization computed from training observations only.

---

## [v1.0.0] — 2024 — Initial 7-day Baseline

### Added
- Initial Himawari-8 HSD download, processing and validation pipeline.
- Full-image autoencoder (SE residual, 512-dim, 23.9M parameters).
- Independently initialized masked-image teacher encoder.
- FRAME model (frozen encoders + trainable projection head).
- Retrieval evaluation: Teacher, FRAME, Autoencoder, Pixel, SSIM.
- Initial experiment report.
