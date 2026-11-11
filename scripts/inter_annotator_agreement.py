"""
Inter-annotator agreement for FRAME visual relevance evaluation.

Computes Cohen's Kappa, Krippendorff's Alpha and percentage agreement
across multiple human annotators to validate that relevance labels
are reliable before computing retrieval metrics.

USAGE
-----
1. Collect ratings from multiple annotators using create_improved_visual_review.py
2. Save each annotator's ratings as a separate CSV (see format below)
3. Run: python scripts/inter_annotator_agreement.py

INPUT FORMAT
------------
results/expanded_2022/annotations/annotator_<name>.csv
    Columns: query, retrieved_image, relevance
    relevance: 0, 1, 2, or 3
    One file per annotator, same (query, retrieved_image) pairs in each.

Example files:
    annotator_ranjan.csv
    annotator_alice.csv
    annotator_bob.csv

OUTPUT FILES
------------
results/expanded_2022/annotations/
    agreement_summary.csv        — Kappa and Alpha per annotator pair
    pairwise_agreement_matrix.csv — Full pairwise percentage agreement
    disagreement_cases.csv       — Cases where annotators disagreed most
    annotated_relevance.csv      — Final merged relevance labels (mean/majority)

INTERPRETATION
--------------
Cohen's Kappa:
    < 0.20  : Poor agreement — do NOT use labels for metrics
    0.20–0.39: Fair agreement — use with caution
    0.40–0.59: Moderate agreement — acceptable for exploratory work
    0.60–0.79: Substantial agreement — good for research claims
    0.80–1.00: Almost perfect — excellent

Krippendorff's Alpha:
    >= 0.667 : Acceptable for research publication
    >= 0.800 : Good
    < 0.667  : Unreliable — improve rubric and re-annotate

If agreement is below acceptable thresholds:
    1. Review disagreement_cases.csv to identify confusing items
    2. Improve rubric anchors in create_improved_visual_review.py
    3. Run a calibration session with annotators
    4. Re-annotate and re-check agreement
"""

from pathlib import Path
import numpy as np
import pandas as pd
from itertools import combinations

# --------------------------------------------------
# Configuration
# --------------------------------------------------

ANNOTATIONS_DIR = Path("results/expanded_2022/annotations")
ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)

KAPPA_THRESHOLDS = {
    0.20: "Poor — do NOT use for metrics",
    0.40: "Fair — use with caution",
    0.60: "Moderate — acceptable for exploratory work",
    0.80: "Substantial — good for research claims",
    1.01: "Almost perfect — excellent",
}

# --------------------------------------------------
# Load annotator files
# --------------------------------------------------

annotator_files = sorted(ANNOTATIONS_DIR.glob("annotator_*.csv"))

if len(annotator_files) < 2:
    print("ERROR: Need at least 2 annotator files to compute agreement.")
    print(f"Expected files in: {ANNOTATIONS_DIR}")
    print("Expected filename pattern: annotator_<name>.csv")
    print("Expected columns: query, retrieved_image, relevance")
    print()
    print("Example file structure:")
    print("  query,retrieved_image,relevance")
    print("  20220122_1200,20220105_0000,2")
    print("  20220122_1200,20220109_0900,1")
    print("  ...")
    raise SystemExit(1)

print("INTER-ANNOTATOR AGREEMENT")
print("=" * 50)
print(f"Annotator files found: {len(annotator_files)}")

annotators = {}
for f in annotator_files:
    name = f.stem.replace("annotator_", "")
    df = pd.read_csv(f)
    df["item"] = df["query"] + "|" + df["retrieved_image"]
    annotators[name] = df.set_index("item")["relevance"]
    print(f"  {name}: {len(df)} ratings")

# Find common items rated by all annotators
all_items = set.intersection(*[set(s.index) for s in annotators.values()])
print(f"\nItems rated by all annotators: {len(all_items)}")

if len(all_items) == 0:
    raise ValueError(
        "No items in common across all annotator files. "
        "Check that all annotators rated the same (query, retrieved_image) pairs."
    )

# Align all annotators to common items
aligned = pd.DataFrame({
    name: series.reindex(all_items)
    for name, series in annotators.items()
}).dropna()

print(f"Items used for agreement: {len(aligned)}")

# --------------------------------------------------
# Cohen's Kappa (pairwise)
# --------------------------------------------------

def cohen_kappa(rater_a, rater_b):
    """
    Compute Cohen's Kappa for two raters with ordinal categories 0-3.
    """
    categories = [0, 1, 2, 3]
    n = len(rater_a)

    # Observed agreement
    p_o = (rater_a == rater_b).mean()

    # Expected agreement
    p_e = sum(
        (rater_a == c).mean() * (rater_b == c).mean()
        for c in categories
    )

    if p_e == 1.0:
        return 1.0

    return (p_o - p_e) / (1 - p_e)


def kappa_interpretation(kappa):
    for threshold, label in sorted(KAPPA_THRESHOLDS.items()):
        if kappa < threshold:
            return label
    return "Perfect"


# --------------------------------------------------
# Krippendorff's Alpha (all annotators simultaneously)
# --------------------------------------------------

def krippendorff_alpha(data_matrix):
    """
    Compute Krippendorff's Alpha for ordinal data.
    data_matrix: numpy array of shape (n_annotators, n_items)
    Missing values should be np.nan.

    Uses ordinal metric: d^2 = (rank_i - rank_j)^2
    """
    n_annotators, n_items = data_matrix.shape
    categories = np.array([0, 1, 2, 3])

    # Observed disagreement (Do)
    do_sum = 0.0
    do_count = 0

    for item in range(n_items):
        values = data_matrix[:, item]
        valid = values[~np.isnan(values)]
        m = len(valid)
        if m < 2:
            continue
        for i in range(m):
            for j in range(i + 1, m):
                do_sum += (valid[i] - valid[j]) ** 2
                do_count += 1

    if do_count == 0:
        return 1.0

    do = do_sum / do_count

    # Expected disagreement (De) — based on overall value distribution
    all_values = data_matrix.flatten()
    all_values = all_values[~np.isnan(all_values)]
    n_total = len(all_values)

    de_sum = 0.0
    for i in range(n_total):
        for j in range(n_total):
            if i != j:
                de_sum += (all_values[i] - all_values[j]) ** 2

    de = de_sum / (n_total * (n_total - 1))

    if de == 0:
        return 1.0

    return 1.0 - (do / de)


# --------------------------------------------------
# Compute all pairwise Kappa values
# --------------------------------------------------

annotator_names = list(annotators.keys())
pairwise_rows = []

print("\nPAIRWISE COHEN'S KAPPA")
print("-" * 50)

for name_a, name_b in combinations(annotator_names, 2):
    rater_a = aligned[name_a].values
    rater_b = aligned[name_b].values

    kappa = cohen_kappa(rater_a, rater_b)
    pct_agree = (rater_a == rater_b).mean()

    interpretation = kappa_interpretation(kappa)

    print(f"  {name_a} vs {name_b}:")
    print(f"    Kappa            : {kappa:.4f}")
    print(f"    % agreement      : {pct_agree:.1%}")
    print(f"    Interpretation   : {interpretation}")
    print()

    pairwise_rows.append({
        "annotator_a": name_a,
        "annotator_b": name_b,
        "cohen_kappa": round(kappa, 4),
        "pct_agreement": round(pct_agree, 4),
        "n_items": len(aligned),
        "interpretation": interpretation
    })

pairwise_df = pd.DataFrame(pairwise_rows)

# --------------------------------------------------
# Krippendorff's Alpha
# --------------------------------------------------

data_matrix = aligned[annotator_names].values.T.astype(float)
alpha = krippendorff_alpha(data_matrix)

print(f"KRIPPENDORFF'S ALPHA (all annotators)")
print(f"  Alpha: {alpha:.4f}")
if alpha >= 0.800:
    alpha_label = "Good — suitable for research publication"
elif alpha >= 0.667:
    alpha_label = "Acceptable — suitable for exploratory research"
else:
    alpha_label = "Unreliable — improve rubric and re-annotate"
print(f"  Interpretation: {alpha_label}")
print()

# --------------------------------------------------
# Pairwise agreement matrix
# --------------------------------------------------

matrix = pd.DataFrame(
    index=annotator_names,
    columns=annotator_names,
    dtype=float
)

for name_a, name_b in combinations(annotator_names, 2):
    kappa = pairwise_df[
        (pairwise_df["annotator_a"] == name_a) &
        (pairwise_df["annotator_b"] == name_b)
    ]["cohen_kappa"].values[0]
    matrix.loc[name_a, name_b] = kappa
    matrix.loc[name_b, name_a] = kappa

for name in annotator_names:
    matrix.loc[name, name] = 1.0

# --------------------------------------------------
# Disagreement cases
# --------------------------------------------------

aligned["range"] = aligned[annotator_names].max(axis=1) - aligned[annotator_names].min(axis=1)
aligned["mean_rating"] = aligned[annotator_names].mean(axis=1)
aligned["std_rating"] = aligned[annotator_names].std(axis=1)

disagreements = aligned[aligned["range"] >= 2].copy()
disagreements = disagreements.reset_index().rename(columns={"index": "item"})
disagreements[["query", "retrieved_image"]] = disagreements["item"].str.split("|", expand=True)
disagreements = disagreements.sort_values("range", ascending=False)

print(f"HIGH-DISAGREEMENT CASES (range >= 2): {len(disagreements)}")
if len(disagreements) > 0:
    print(disagreements[["query", "retrieved_image", "range", "mean_rating"] + annotator_names].head(10).to_string(index=False))
print()

# --------------------------------------------------
# Merge labels — use mean rating rounded to nearest integer
# --------------------------------------------------

aligned["final_relevance"] = aligned[annotator_names].mean(axis=1).round().astype(int)

merged = aligned.reset_index().rename(columns={"index": "item"})
merged[["query", "retrieved_image"]] = merged["item"].str.split("|", expand=True)
merged = merged[["query", "retrieved_image", "final_relevance"] + annotator_names]
merged = merged.rename(columns={"final_relevance": "relevance"})

# --------------------------------------------------
# Save outputs
# --------------------------------------------------

pairwise_df.to_csv(ANNOTATIONS_DIR / "agreement_summary.csv", index=False)
matrix.to_csv(ANNOTATIONS_DIR / "pairwise_agreement_matrix.csv")
disagreements[
    ["query", "retrieved_image", "range", "mean_rating", "std_rating"] + annotator_names
].to_csv(ANNOTATIONS_DIR / "disagreement_cases.csv", index=False)
merged[["query", "retrieved_image", "relevance"]].to_csv(
    ANNOTATIONS_DIR / "annotated_relevance.csv", index=False
)

print("OUTPUT FILES")
for path in sorted(ANNOTATIONS_DIR.glob("*.csv")):
    print(f"  {path}")

print()
print("NEXT STEPS")
print("-" * 50)
if alpha >= 0.667:
    print("  Agreement is acceptable.")
    print("  annotated_relevance.csv is ready for compute_retrieval_metrics.py")
    print("  Run: python scripts/compute_retrieval_metrics.py")
else:
    print("  Agreement is too low to use labels reliably.")
    print("  1. Review disagreement_cases.csv")
    print("  2. Improve rubric anchors in create_improved_visual_review.py")
    print("  3. Run a calibration session with annotators")
    print("  4. Re-annotate and re-run this script")
