"""
Retrieval metrics for FRAME satellite image retrieval experiment.

Computes Precision@K, Recall@K, NDCG@K and Mean Average Precision (mAP)
for each retrieval method once human relevance labels are available.

USAGE
-----
1. Collect relevance labels using create_improved_visual_review.py
2. Merge labels into the annotated_relevance.csv format (see below)
3. Run:  python scripts/compute_retrieval_metrics.py

INPUT FILES
-----------
results/expanded_2022/final_test_rankings.csv
    Columns: query, method, rank, retrieved_image, score
    Contains complete ranked lists (112 candidates) for all 56 queries x 5 methods.

results/expanded_2022/annotations/annotated_relevance.csv
    Columns: query, retrieved_image, relevance
    relevance: 0 (not relevant), 1 (weakly relevant), 2 (relevant), 3 (highly relevant)
    One row per (query, retrieved_image) pair that was annotated.
    Queries and images not in this file are treated as unannotated (excluded).

OUTPUT FILES
------------
results/expanded_2022/retrieval_metrics/
    precision_at_k.csv       — Precision@1,3,5,10 per method
    recall_at_k.csv          — Recall@1,3,5,10 per method
    ndcg_at_k.csv            — NDCG@1,3,5,10 per method
    mean_average_precision.csv — mAP per method
    per_query_metrics.csv    — Full per-query breakdown for all methods
    metrics_summary.csv      — Single summary table for reporting

RELEVANCE THRESHOLD
-------------------
By default, relevance >= 2 is treated as "relevant" for binary metrics
(Precision@K, Recall@K, mAP). NDCG@K uses the raw graded scores.
Change RELEVANCE_THRESHOLD below to adjust.

IMPORTANT INTERPRETATION NOTE
------------------------------
These metrics are only meaningful when relevance labels are:
- Independently assigned by multiple annotators
- Based on a clear, well-anchored rubric
- Measured with acceptable inter-annotator agreement (Cohen's Kappa >= 0.6)

Do not interpret metrics computed from a single annotator or from
the non-discriminative pilot (where all ratings were 2) as evidence
of retrieval quality.
"""

from pathlib import Path
import numpy as np
import pandas as pd

# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE = Path("results/expanded_2022")
RANKINGS_FILE = BASE / "final_test_rankings.csv"
RELEVANCE_FILE = BASE / "annotations" / "annotated_relevance.csv"
OUT = BASE / "retrieval_metrics"
OUT.mkdir(parents=True, exist_ok=True)

METHODS = ["Teacher", "FRAME", "Autoencoder", "Pixel", "SSIM"]
K_VALUES = [1, 3, 5, 10]

# Relevance score >= this threshold counts as "relevant" for binary metrics
RELEVANCE_THRESHOLD = 2

# --------------------------------------------------
# Load data
# --------------------------------------------------

print("FRAME RETRIEVAL METRICS")
print("=" * 50)

if not RANKINGS_FILE.exists():
    raise FileNotFoundError(
        f"Rankings file not found: {RANKINGS_FILE}\n"
        "Run evaluate_expanded_final_test.py first."
    )

if not RELEVANCE_FILE.exists():
    raise FileNotFoundError(
        f"Relevance labels not found: {RELEVANCE_FILE}\n"
        "Collect annotations using create_improved_visual_review.py\n"
        "then save them to:\n"
        f"  {RELEVANCE_FILE}\n"
        "with columns: query, retrieved_image, relevance"
    )

rankings = pd.read_csv(RANKINGS_FILE)
relevance = pd.read_csv(RELEVANCE_FILE)

print(f"Rankings loaded   : {len(rankings)} rows")
print(f"Relevance labels  : {len(relevance)} rows")
print(f"Annotated queries : {relevance['query'].nunique()}")
print(f"Relevance threshold (binary): >= {RELEVANCE_THRESHOLD}")
print()

# Validate relevance file structure
required_cols = {"query", "retrieved_image", "relevance"}
missing = required_cols - set(relevance.columns)
if missing:
    raise ValueError(
        f"annotated_relevance.csv is missing columns: {missing}\n"
        f"Expected columns: query, retrieved_image, relevance"
    )

valid_scores = {0, 1, 2, 3}
invalid = set(relevance["relevance"].unique()) - valid_scores
if invalid:
    raise ValueError(
        f"Invalid relevance scores found: {invalid}\n"
        f"Valid scores are: 0, 1, 2, 3"
    )

# Build relevance lookup: (query, retrieved_image) -> relevance score
relevance_lookup = relevance.set_index(
    ["query", "retrieved_image"]
)["relevance"].to_dict()

annotated_queries = relevance["query"].unique()
print(f"Queries with annotations: {len(annotated_queries)}")

# --------------------------------------------------
# Core metric functions
# --------------------------------------------------

def get_relevance_scores(query, ranked_list):
    """
    Return graded relevance scores for a ranked list of retrieved images.
    Images not in the relevance lookup get score 0.
    """
    return np.array([
        relevance_lookup.get((query, img), 0)
        for img in ranked_list
    ], dtype=float)


def precision_at_k(relevance_scores, k, threshold=RELEVANCE_THRESHOLD):
    """
    Precision@K: fraction of top-K results that are relevant.
    """
    top_k = relevance_scores[:k]
    relevant = (top_k >= threshold).sum()
    return relevant / k


def recall_at_k(relevance_scores, k, threshold=RELEVANCE_THRESHOLD):
    """
    Recall@K: fraction of all relevant items retrieved in top-K.
    """
    total_relevant = (relevance_scores >= threshold).sum()
    if total_relevant == 0:
        return 0.0
    top_k = relevance_scores[:k]
    retrieved_relevant = (top_k >= threshold).sum()
    return retrieved_relevant / total_relevant


def dcg_at_k(relevance_scores, k):
    """
    Discounted Cumulative Gain@K using graded relevance.
    Uses log base 2: DCG = sum(rel_i / log2(i+2)) for i in 0..k-1
    """
    top_k = relevance_scores[:k]
    gains = top_k / np.log2(np.arange(2, len(top_k) + 2))
    return gains.sum()


def ndcg_at_k(relevance_scores, k):
    """
    Normalized DCG@K: DCG@K divided by ideal DCG@K.
    Ideal DCG assumes results are sorted by relevance descending.
    """
    dcg = dcg_at_k(relevance_scores, k)
    ideal_scores = np.sort(relevance_scores)[::-1]
    ideal_dcg = dcg_at_k(ideal_scores, k)
    if ideal_dcg == 0:
        return 0.0
    return dcg / ideal_dcg


def average_precision(relevance_scores, threshold=RELEVANCE_THRESHOLD):
    """
    Average Precision: area under the precision-recall curve.
    AP = sum(P@k * rel@k) / total_relevant
    """
    total_relevant = (relevance_scores >= threshold).sum()
    if total_relevant == 0:
        return 0.0

    precisions = []
    num_relevant = 0

    for i, score in enumerate(relevance_scores):
        if score >= threshold:
            num_relevant += 1
            precisions.append(num_relevant / (i + 1))

    return np.mean(precisions) if precisions else 0.0


# --------------------------------------------------
# Compute metrics per query per method
# --------------------------------------------------

per_query_rows = []

for method in METHODS:
    method_rankings = rankings[rankings["method"] == method]

    for query in annotated_queries:
        query_rankings = method_rankings[
            method_rankings["query"] == query
        ].sort_values("rank")

        if len(query_rankings) == 0:
            print(f"  WARNING: No rankings found for {query} / {method}")
            continue

        ranked_list = query_rankings["retrieved_image"].tolist()
        rel_scores = get_relevance_scores(query, ranked_list)

        row = {
            "query": query,
            "method": method,
            "ap": average_precision(rel_scores),
            "total_relevant": int((rel_scores >= RELEVANCE_THRESHOLD).sum()),
        }

        for k in K_VALUES:
            row[f"precision@{k}"] = precision_at_k(rel_scores, k)
            row[f"recall@{k}"] = recall_at_k(rel_scores, k)
            row[f"ndcg@{k}"] = ndcg_at_k(rel_scores, k)

        per_query_rows.append(row)

per_query = pd.DataFrame(per_query_rows)

# --------------------------------------------------
# Aggregate to method-level summaries
# --------------------------------------------------

def method_summary(metric_col):
    return (
        per_query.groupby("method")[metric_col]
        .agg(["mean", "std", "min", "max"])
        .round(4)
        .rename(columns={
            "mean": f"mean_{metric_col}",
            "std": f"std_{metric_col}",
            "min": f"min_{metric_col}",
            "max": f"max_{metric_col}"
        })
    )


# Precision@K
precision_rows = []
for k in K_VALUES:
    col = f"precision@{k}"
    agg = per_query.groupby("method")[col].mean().reset_index()
    agg["k"] = k
    agg.rename(columns={col: "mean_precision"}, inplace=True)
    precision_rows.append(agg)
precision_df = pd.concat(precision_rows).pivot(
    index="method", columns="k", values="mean_precision"
).round(4)
precision_df.columns = [f"P@{k}" for k in precision_df.columns]

# Recall@K
recall_rows = []
for k in K_VALUES:
    col = f"recall@{k}"
    agg = per_query.groupby("method")[col].mean().reset_index()
    agg["k"] = k
    agg.rename(columns={col: "mean_recall"}, inplace=True)
    recall_rows.append(agg)
recall_df = pd.concat(recall_rows).pivot(
    index="method", columns="k", values="mean_recall"
).round(4)
recall_df.columns = [f"R@{k}" for k in recall_df.columns]

# NDCG@K
ndcg_rows = []
for k in K_VALUES:
    col = f"ndcg@{k}"
    agg = per_query.groupby("method")[col].mean().reset_index()
    agg["k"] = k
    agg.rename(columns={col: "mean_ndcg"}, inplace=True)
    ndcg_rows.append(agg)
ndcg_df = pd.concat(ndcg_rows).pivot(
    index="method", columns="k", values="mean_ndcg"
).round(4)
ndcg_df.columns = [f"NDCG@{k}" for k in ndcg_df.columns]

# mAP
map_df = per_query.groupby("method")["ap"].mean().round(4).reset_index()
map_df.rename(columns={"ap": "mAP"}, inplace=True)
map_df = map_df.set_index("method")

# Combined summary
summary = pd.concat(
    [precision_df, recall_df, ndcg_df, map_df], axis=1
).reset_index()

# Reorder rows to match METHODS order
summary["method"] = pd.Categorical(
    summary["method"], categories=METHODS, ordered=True
)
summary = summary.sort_values("method").reset_index(drop=True)

# --------------------------------------------------
# Save outputs
# --------------------------------------------------

per_query.to_csv(OUT / "per_query_metrics.csv", index=False)
precision_df.reset_index().to_csv(OUT / "precision_at_k.csv", index=False)
recall_df.reset_index().to_csv(OUT / "recall_at_k.csv", index=False)
ndcg_df.reset_index().to_csv(OUT / "ndcg_at_k.csv", index=False)
map_df.reset_index().to_csv(OUT / "mean_average_precision.csv", index=False)
summary.to_csv(OUT / "metrics_summary.csv", index=False)

# --------------------------------------------------
# Print results
# --------------------------------------------------

print("\nRETRIEVAL METRICS SUMMARY")
print("=" * 50)
print(f"Queries evaluated : {per_query['query'].nunique()}")
print(f"Methods evaluated : {len(METHODS)}")
print(f"Relevance threshold: >= {RELEVANCE_THRESHOLD} (binary metrics)")
print()
print(summary.to_string(index=False))

print("\nPER-METHOD mAP")
print(map_df.to_string())

print("\nPRECISION@K")
print(precision_df.to_string())

print("\nRECALL@K")
print(recall_df.to_string())

print("\nNDCG@K")
print(ndcg_df.to_string())

print("\nOUTPUT FILES")
for path in sorted(OUT.glob("*.csv")):
    print(f"  {path}")

print("\nINTERPRETATION REMINDER")
print("-" * 50)
print("These metrics are only meaningful when relevance labels are:")
print("  - Assigned by multiple independent annotators")
print("  - Based on a clear, well-anchored rubric")
print("  - Validated with inter-annotator agreement (Kappa >= 0.6)")
print("Run inter_annotator_agreement.py to check agreement first.")
