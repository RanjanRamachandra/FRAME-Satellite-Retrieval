"""
Statistical validation for FRAME satellite retrieval experiment.

IMPORTANT — STATISTICAL POWER AND INTERPRETATION NOTICE
--------------------------------------------------------
This analysis uses 7 daily clusters (one per test day, January 22-28).

Bootstrap confidence intervals:
  - 7 clusters is well below the ~30 recommended for stable bootstrap CIs.
  - Intervals should be treated as EXPLORATORY and ILLUSTRATIVE only.

Paired Wilcoxon signed-rank tests:
  - n=7 gives very limited power. Only very large effects are detectable.
  - Non-significant results do NOT mean methods are equivalent.
  - The smallest Holm-adjusted p-value in this experiment was 0.1562.
  - No pairwise temporal difference was statistically significant at p < 0.05.

Wilson confidence intervals (Top-1 agreement):
  - Computed treating each of the 56 query pairs as independent.
  - Queries within the same day may be correlated (same weather system).
  - These intervals may be OPTIMISTIC (too narrow).
  - They should be treated as approximate/illustrative only.

All findings should be labelled as EXPLORATORY until a larger,
multi-month, multi-region dataset with ground-truth relevance labels
is available.
"""

from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from math import sqrt

BASE = Path("results/expanded_2022")
OUT = BASE / "statistical_validation"
OUT.mkdir(parents=True, exist_ok=True)

METHODS = ["Teacher", "FRAME", "Autoencoder", "Pixel", "SSIM"]
BOOTSTRAP_REPS = 10000
SEED = 42
Z = 1.959963984540054

rng = np.random.default_rng(SEED)

queries = pd.read_csv(BASE / "final_test_per_query.csv")
pairs = pd.read_csv(BASE / "final_test_pairwise_comparison.csv")

# Validate data structure.
assert len(queries) == 280
assert queries["query"].nunique() == 56
assert len(pairs) == 560
assert pairs["query"].nunique() == 56
assert set(queries["method"]) == set(METHODS)

# Temporal separation.
queries["query_time"] = pd.to_datetime(
    queries["query"], format="%Y%m%d_%H%M"
)
queries["top1_time"] = pd.to_datetime(
    queries["top1"], format="%Y%m%d_%H%M"
)
queries["temporal_distance_hours"] = (
    queries["query_time"] - queries["top1_time"]
).abs().dt.total_seconds() / 3600

queries["day"] = queries["query_time"].dt.strftime("%Y%m%d")

# Seven daily averages per method.
daily = (
    queries.groupby(["day", "method"])
    ["temporal_distance_hours"]
    .mean()
    .reset_index()
)

assert daily["day"].nunique() == 7
assert daily.groupby("method").size().eq(7).all()

# --------------------------------------------------
# 1. Day-cluster bootstrap confidence intervals
# --------------------------------------------------

def bootstrap_summary(frame, column):
    results = []

    for method in METHODS:
        values = (
            frame[frame["method"] == method]
            .groupby("day")[column]
            .mean()
            .to_numpy()
        )

        samples = rng.choice(
            values,
            size=(BOOTSTRAP_REPS, len(values)),
            replace=True
        ).mean(axis=1)

        results.append({
            "method": method,
            "observed_daily_mean": values.mean(),
            "ci_lower_95": np.percentile(samples, 2.5),
            "ci_upper_95": np.percentile(samples, 97.5),
            "days": len(values)
        })

    return pd.DataFrame(results)


temporal_ci = bootstrap_summary(
    queries, "temporal_distance_hours"
)

# --------------------------------------------------
# 2. Paired Wilcoxon tests on daily averages
# --------------------------------------------------

wide = daily.pivot(
    index="day",
    columns="method",
    values="temporal_distance_hours"
)[METHODS]

test_results = []

for method_a, method_b in combinations(METHODS, 2):
    a = wide[method_a]
    b = wide[method_b]
    differences = a - b

    if np.allclose(differences, 0):
        statistic, p_value = 0.0, 1.0
    else:
        result = wilcoxon(
            a, b,
            alternative="two-sided",
            method="exact"
        )
        statistic = result.statistic
        p_value = result.pvalue

    test_results.append({
        "method_a": method_a,
        "method_b": method_b,
        "median_daily_difference_hours": differences.median(),
        "wilcoxon_statistic": statistic,
        "p_value": p_value
    })

tests = pd.DataFrame(test_results)

# Holm multiple-comparison correction.
order = np.argsort(tests["p_value"].to_numpy())
sorted_p = tests["p_value"].to_numpy()[order]
m = len(sorted_p)

adjusted = np.maximum.accumulate(
    [(m - i) * p for i, p in enumerate(sorted_p)]
)
adjusted = np.minimum(adjusted, 1.0)

holm = np.empty(m)
holm[order] = adjusted
tests["p_value_holm"] = holm

# --------------------------------------------------
# 3. Wilson confidence intervals for Top-1 agreement
# NOTE: These intervals assume individual query independence.
# Queries within the same day may be correlated (same weather system),
# so these intervals may be optimistic (too narrow).
# Treat as approximate/illustrative only.
# --------------------------------------------------

agreement_results = []

for (method_a, method_b), group in pairs.groupby(
    ["method_a", "method_b"]
):
    successes = int(group["top1_agreement"].sum())
    total = len(group)
    proportion = successes / total

    denominator = 1 + Z**2 / total
    center = (
        proportion + Z**2 / (2 * total)
    ) / denominator

    margin = (
        Z * sqrt(
            proportion * (1 - proportion) / total
            + Z**2 / (4 * total**2)
        ) / denominator
    )

    agreement_results.append({
        "method_a": method_a,
        "method_b": method_b,
        "agreements": successes,
        "queries": total,
        "agreement_rate": proportion,
        "ci_lower_95": center - margin,
        "ci_upper_95": center + margin
    })

agreement = pd.DataFrame(agreement_results)

# --------------------------------------------------
# 4. Top-5 overlap summary and bootstrap intervals
# --------------------------------------------------

top5 = pairs.copy()
top5["day"] = pd.to_datetime(
    top5["query"], format="%Y%m%d_%H%M"
).dt.strftime("%Y%m%d")

overlap_results = []

for (method_a, method_b), group in top5.groupby(
    ["method_a", "method_b"]
):
    daily_values = (
        group.groupby("day")["top5_overlap"].mean().to_numpy()
    )

    samples = rng.choice(
        daily_values,
        size=(BOOTSTRAP_REPS, len(daily_values)),
        replace=True
    ).mean(axis=1)

    overlap_results.append({
        "method_a": method_a,
        "method_b": method_b,
        "observed_daily_mean": daily_values.mean(),
        "ci_lower_95": np.percentile(samples, 2.5),
        "ci_upper_95": np.percentile(samples, 97.5),
        "days": len(daily_values)
    })

overlap_ci = pd.DataFrame(overlap_results)

# --------------------------------------------------
# 5. Save results
# --------------------------------------------------

temporal_ci.to_csv(
    OUT / "temporal_bootstrap_ci.csv", index=False
)

tests.to_csv(
    OUT / "temporal_paired_tests.csv", index=False
)

agreement.to_csv(
    OUT / "top1_agreement_wilson_ci.csv", index=False
)

overlap_ci.to_csv(
    OUT / "top5_overlap_bootstrap_ci.csv", index=False
)

print("\nSTATISTICAL VALIDATION COMPLETED")

print("\nTEMPORAL DAILY-CLUSTER BOOTSTRAP")
print(temporal_ci.round(3).to_string(index=False))

print("\nPAIRED WILCOXON TESTS WITH HOLM CORRECTION")
print(tests.round(4).to_string(index=False))

print("\nTOP-1 AGREEMENT WITH WILSON INTERVALS")
print(agreement.round(4).to_string(index=False))

print("\nTOP-5 OVERLAP BOOTSTRAP")
print(overlap_ci.round(3).to_string(index=False))

print("\nOUTPUT FILES")
for path in sorted(OUT.glob("*.csv")):
    print(path)
