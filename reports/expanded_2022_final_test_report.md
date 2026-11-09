# Expanded Himawari-8 Satellite Retrieval Experiment

## 1. Research objective

Evaluate FRAME (Focus Region Aware Masking Embedding) for region-specific satellite image retrieval using Himawari-8 observations.

Compare FRAME against:
- Masked Teacher
- Full-image Autoencoder
- Visible Pixel similarity
- Structural Similarity (SSIM)

The objective is to evaluate retrieval behaviour and investigate whether FRAME produces visually relevant satellite-image matches.

This is a modified implementation inspired by the original research, using Himawari-8 rather than COMS imagery.

## 2. Dataset

Source: NOAA public Himawari-8 AHI Level-1b archive.

Period: January 1–28, 2022.

Bands:
- B07: Shortwave infrared
- B08: Water vapour
- B13: Clean infrared window

Geographic coverage:
- Longitude: 90°E–160°E
- Latitude: 10°S–60°N

Processed image dimensions: 1000 × 1000 pixels.

Model input dimensions: 200 × 200 pixels.

Total observations: 224.

### Chronological split

| Split | Period | Observations |
|---|---|---:|
| Training | January 1–14 | 112 |
| Validation | January 15–21 | 56 |
| Final test | January 22–28 | 56 |
| Total | January 1–28 | 224 |

January 22, 2022, 21:00 was unavailable in the NOAA archive. The actual 20:50 observation was used as a replacement and retained under its original timestamp.

All 224 NPZ files passed validation:
- Required bands present.
- Expected dimensions and data types.
- Finite pixel values.
- No duplicate observations.

Normalization statistics were calculated using training observations only.

## 3. Model training

All expanded experiments used:
- Random seed: 42
- Embedding dimension: 512
- Device: Apple MPS
- Training observations: 112
- Validation observations: 56

The final test dataset was not loaded during training or model selection.

### 3.1 Expanded Autoencoder

Architecture: SE residual encoder-decoder.

Loss: Mean Squared Error.

Optimizer: Adam.

Initial learning rate: 0.001.

Best checkpoint:
- Epoch: 12
- Validation loss: 0.52638845

### 3.2 Expanded Masked Teacher

The masked teacher was independently initialized.

Training objective: reconstruct the complete image from a randomly masked input.

Loss: Mean Squared Error.

Best checkpoint:
- Epoch: 11
- Validation loss: 0.53053618

### 3.3 Expanded FRAME

The expanded Autoencoder encoder and Masked Teacher encoder were frozen.

FRAME was trained to predict the masked Teacher embedding using:
- Original full-image embedding.
- Binary mask embedding.
- Projection head.

Loss: 1 - cosine similarity.

Best checkpoint:
- Epoch: 12
- Validation loss: 0.00350319

These losses measure reconstruction or teacher-embedding imitation. They do not independently establish retrieval relevance.

## 4. Final test methodology

The independent final test used:
- 56 test queries.
- 112 training database observations.
- Five retrieval methods.
- Deterministic masks using seed 1000 plus query index.

Methods:
1. Teacher: masked query embedding against full-image Teacher database embeddings.
2. FRAME: FRAME query embedding against full-image Teacher database embeddings.
3. Autoencoder: full-image Autoencoder query embedding against Autoencoder database embeddings.
4. Pixel: visible-region mean squared error.
5. SSIM: visible-region structural similarity.

Cosine embeddings were L2-normalized for retrieval scoring.

Validation observations were not used for final scoring.

No test query appeared in the training database.

## 5. Final test results

### Top-1 retrieval diversity

| Method | Queries | Unique top-1 matches |
|---|---:|---:|
| Teacher | 56 | 34 |
| FRAME | 56 | 23 |
| Autoencoder | 56 | 31 |
| Pixel | 56 | 25 |
| SSIM | 56 | 38 |

Unique top-1 matches measure retrieval diversity, not relevance or accuracy.

### Pairwise comparison

| Comparison | Top-1 agreement | Average Top-5 overlap |
|---|---:|---:|
| Autoencoder vs Pixel | 1.79% | 0.57 |
| Autoencoder vs SSIM | 1.79% | 0.61 |
| FRAME vs Autoencoder | 8.93% | 0.77 |
| FRAME vs Pixel | 3.57% | 0.48 |
| FRAME vs SSIM | 1.79% | 0.59 |
| Pixel vs SSIM | 19.64% | 1.86 |
| Teacher vs Autoencoder | 5.36% | 1.18 |
| Teacher vs FRAME | 14.29% | 1.39 |
| Teacher vs Pixel | 1.79% | 0.66 |
| Teacher vs SSIM | 5.36% | 0.86 |

These metrics measure agreement between ranking methods. They do not establish which ranking is correct.

## 6. Temporal retrieval analysis

Absolute time difference between each test query and its top-1 retrieved training observation:

| Method | Mean hours | Median hours |
|---|---:|---:|
| Autoencoder | 352.44 | 336.0 |
| FRAME | 429.53 | 442.5 |
| Pixel | 351.53 | 336.0 |
| SSIM | 412.28 | 352.5 |
| Teacher | 387.43 | 372.0 |

FRAME had the largest mean and median temporal separation.

Temporal separation is not equivalent to meteorological relevance. These results indicate different retrieval behaviour, not superiority.

## 7. Human visual relevance evaluation

Ten final-test queries were selected for blinded visual assessment.

For each query:
- One query image was displayed.
- Five anonymous method groups were presented.
- Each group contained five retrieved images.
- Method labels were hidden during rating.

Total ratings: 250.

Rating scale:
- 0: No meaningful visual similarity.
- 1: Weak similarity.
- 2: Moderate similarity.
- 3: Strong similarity.

### Observed results

| Method | Images rated | Mean rating |
|---|---:|---:|
| Teacher | 50 | 2.0 |
| FRAME | 50 | 2.0 |
| Autoencoder | 50 | 2.0 |
| Pixel | 50 | 2.0 |
| SSIM | 50 | 2.0 |

All 250 images received a score of 2.

The ratings were manually entered and passed structural validation.

This evaluation did not discriminate between methods. It should not be interpreted as proof that all methods have equal retrieval quality.

The single-annotator assessment and lack of rating variation are important limitations.

## 8. Reproducibility and integrity audit

Final test audit:
- Ranking rows: 31,360.
- Per-query rows: 280.
- Method summary rows: 5.
- Pairwise comparison rows: 560.
- Complete query-method combinations: 280.
- Incorrect ranking counts: 0.
- Duplicate query/database pairs: 0.
- Non-finite scores: 0.
- Test queries appearing in training database: 0.

All structural assertions passed.

## 9. Research limitations

1. Only 224 observations from one month were used.
2. The final test contains 56 observations.
3. Only three Himawari-8 bands were evaluated.
4. Teacher embeddings are learned targets, not ground-truth semantic labels.
5. Ranking agreement is not retrieval relevance.
6. The human evaluation used one annotator.
7. Human ratings had no variation.
8. Temporal distance does not measure meteorological similarity.
9. Results do not establish generalization across seasons or other satellite platforms.
10. The experiment is a modified reproduction, not an exact reproduction of the original COMS study.

## 10. Conclusions

A complete expanded satellite retrieval pipeline was implemented and evaluated.

The experiment demonstrates:
- Expanded chronological data preparation.
- Independent Autoencoder, Masked Teacher and FRAME training.
- Five-method retrieval evaluation.
- Independent final-test scoring.
- Temporal retrieval analysis.
- Blinded human visual assessment.
- Reproducible result-file validation.

FRAME produces retrieval rankings that differ from the other methods. It also shows greater temporal separation in its top-1 results.

However, the available human relevance ratings do not establish that FRAME retrieves more relevant images.

No claim of FRAME superiority is justified by the current evidence.

## 11. Future work

1. Develop a more discriminative human relevance rubric.
2. Obtain independent ratings from multiple annotators.
3. Measure inter-annotator agreement.
4. Expand the dataset across additional months and weather conditions.
5. Investigate visible-image-aware mask conditioning.
6. Evaluate retrieval using Precision@5 and NDCG@5 after obtaining meaningful relevance judgments.
7. Test generalization on a separate seasonal or geographic dataset.

## 12. Key artifacts

Dataset:
- `data/expanded_2022/metadata/expanded_dataset_manifest.csv`
- `data/expanded_2022/metadata/normalization_stats.json`

Checkpoints:
- `checkpoints/expanded_2022/best_autoencoder.pth`
- `checkpoints/expanded_2022/best_masked_encoder.pth`
- `checkpoints/expanded_2022/best_frame_model.pth`

Final results:
- `results/expanded_2022/final_test_rankings.csv`
- `results/expanded_2022/final_test_per_query.csv`
- `results/expanded_2022/final_test_method_summary.csv`
- `results/expanded_2022/final_test_pairwise_comparison.csv`

Human evaluation:
- `results/expanded_2022/blind_review/blind_scoring_template.csv`
- `results/expanded_2022/blind_review/method_key.csv`
- `results/expanded_2022/blind_review/selected_queries.csv`
