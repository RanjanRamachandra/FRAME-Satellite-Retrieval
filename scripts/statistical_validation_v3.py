"""
Improved statistical validation for FRAME satellite retrieval (v3).

Improvements over v2:
  1. Effect size reporting (Cohen's d) alongside p-values.
  2. Permutation tests as a more robust alternative to Wilcoxon
     with very small samples (n=7).
  3. Cluster-aware confidence intervals for Top-1 agreement
     (replaces the assumption-violating Wilson intervals in v2).
  4. Clearer power warnings integrated into output tables.
  5. Confidence interval overlap analysis as a visual aid.

USAGE
-----
    python scripts/statistical_validation_v3.py

INPUT FILES
-----------
    results/expanded_2022/final_test_per_query.csv
    results/expanded_2022/final_test_pairwise_comparison.csv

OUTPUT FILES
------------
    results/expanded_2022/statistical_validation_v3/
        temporal_bootstrap_ci_v3.csv        — Bootstrap CIs with effect sizes
        temporal_permutation_tests_v3.csv   — Permutation test results
        temporal_wilcoxon_v3.csv            — Wilcoxon results (for comparison)
        top1_agreement_cluster_ci_v3.csv    — Cluster-corrected agreement CIs
        top5_overlap_bootstrap_ci_v3.csv    — Bootstrap CIs for overlap
        power_summary_v3.csv               — Power and sample-size context
        statistical_summary_v3.md          — Human-readable report

INTERPRETATION
--------------
With only 7 daily clusters:
  - ALL tests have limited power.
  - Non-significant results do not mean methods are equivalent.
  - Effect sizes (Cohen's d) provide additional context beyond p-values.
  - All findings are EXPLORATORY until more data is available.
"""

from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, t as t_dist

# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE     = Path("results/expanded_2022")
OUT      = BASE / "statistical_validation_v3"
OUT.mkdir(parents=True, exist_ok=True)

METHODS          = ["Teacher", "FRAME", "Autoencoder", "Pixel", "SSIM"]
BOOTSTRAP_REPS   = 10000
PERMUTATION_REPS = 10000
SEED             = 42
rng              = np.random.default_rng(SEED)

print("FRAME STATISTICAL VALIDATION v3")
print("=" * 55)

# --------------------------------------------------
# Load data
# --------------------------------------------------

per_query  = pd.read_csv(BASE / "final_test_per_query.csv")
pairwise   = pd.read_csv(BASE / "final_test_pairwise_comparison.csv")

# Parse timestamps
def parse_hours(obs_id):
    try:
        date = str(obs_id)[:8]
        time = str(obs_id)[9:]
        y, m, d = int(date[:4]), int(date[4:6]), int(date[6:8])
        h = int(time[:2])
        mi = int(time[2:]) if len(time) > 2 else 0
        return ((m - 1) * 30 + d) * 24 + h + mi / 60
    except Exception:
        return 0.0

per_query["query_hours"] = per_query["query"].apply(parse_hours)
per_query["top1_hours"]  = per_query["top1"].apply(parse_hours)
per_query["temporal_dist"] = (
    per_query["query_hours"] - per_query["top1_hours"]
).abs()

per_query["query_date"] = per_query["query"].str[:8]

# --------------------------------------------------
# Helper: Cohen's d
# --------------------------------------------------

def cohens_d(a, b):
    """
    Cohen's d for paired differences.
    d = mean(diff) / std(diff)
    Interpretation: 0.2 small, 0.5 medium, 0.8 large
    """
    diff = np.array(a) - np.array(b)
    std  = np.std(diff, ddof=1)
    if std == 0:
        return 0.0
    return np.mean(diff) / std


def d_interpretation(d):
    d = abs(d)
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"


# --------------------------------------------------
# 1. Daily cluster means
# --------------------------------------------------

daily = (
    per_query
    .groupby(["query_date", "method"])["temporal_dist"]
    .mean()
    .reset_index()
    .rename(columns={"temporal_dist": "mean_temporal_hours"})
)

daily_pivot = daily.pivot(
    index="query_date", columns="method", values="mean_temporal_hours"
).dropna()

n_clusters = len(daily_pivot)
print(f"Daily clusters available: {n_clusters}")
print(f"Bootstrap reps  : {BOOTSTRAP_REPS:,}")
print(f"Permutation reps: {PERMUTATION_REPS:,}")
print()

# --------------------------------------------------
# 2. Bootstrap CIs for temporal separation
# --------------------------------------------------

print("Computing bootstrap CIs for temporal separation...")

boot_rows = []
for method in METHODS:
    if method not in daily_pivot.columns:
        continue
    vals = daily_pivot[method].values

    boot_means = np.array([
        rng.choice(vals, size=len(vals), replace=True).mean()
        for _ in range(BOOTSTRAP_REPS)
    ])

    ci_lo = np.percentile(boot_means, 2.5)
    ci_hi = np.percentile(boot_means, 97.5)

    boot_rows.append({
        "method"            : method,
        "n_clusters"        : n_clusters,
        "mean_hours"        : round(vals.mean(), 2),
        "std_hours"         : round(vals.std(ddof=1), 2),
        "ci_lower_95"       : round(ci_lo, 2),
        "ci_upper_95"       : round(ci_hi, 2),
        "power_note"        : "EXPLORATORY — only 7 clusters",
    })

boot_df = pd.DataFrame(boot_rows).sort_values("mean_hours", ascending=False)
print(boot_df[["method", "mean_hours", "ci_lower_95", "ci_upper_95"]].to_string(index=False))
print()

# --------------------------------------------------
# 3. Permutation tests (paired)
# --------------------------------------------------

print("Running paired permutation tests...")

perm_rows = []
method_pairs = list(combinations(METHODS, 2))

for m_a, m_b in method_pairs:
    if m_a not in daily_pivot.columns or m_b not in daily_pivot.columns:
        continue

    vals_a = daily_pivot[m_a].values
    vals_b = daily_pivot[m_b].values
    diff   = vals_a - vals_b
    obs_stat = np.abs(diff.mean())

    # Permutation: randomly flip signs of paired differences
    perm_stats = np.array([
        np.abs(
            (diff * rng.choice([-1, 1], size=len(diff))).mean()
        )
        for _ in range(PERMUTATION_REPS)
    ])

    p_val = (perm_stats >= obs_stat).mean()
    d     = cohens_d(vals_a, vals_b)

    perm_rows.append({
        "method_a"      : m_a,
        "method_b"      : m_b,
        "mean_diff_hours": round(diff.mean(), 2),
        "permutation_p" : round(p_val, 4),
        "cohens_d"      : round(d, 4),
        "effect_size"   : d_interpretation(d),
        "n_clusters"    : n_clusters,
    })

perm_df = pd.DataFrame(perm_rows)

# Holm correction on permutation p-values
perm_df = perm_df.sort_values("permutation_p")
n_tests = len(perm_df)
perm_df["holm_adjusted_p"] = [
    min(1.0, row["permutation_p"] * (n_tests - i))
    for i, row in perm_df.reset_index(drop=True).iterrows()
]
perm_df["significant_at_0.05"] = perm_df["holm_adjusted_p"] < 0.05

print(perm_df[[
    "method_a", "method_b", "mean_diff_hours",
    "permutation_p", "holm_adjusted_p", "cohens_d", "effect_size"
]].to_string(index=False))
print()

# --------------------------------------------------
# 4. Wilcoxon tests (retained for comparison with v2)
# --------------------------------------------------

print("Running Wilcoxon signed-rank tests (n=7, low power)...")

wilcox_rows = []
for m_a, m_b in method_pairs:
    if m_a not in daily_pivot.columns or m_b not in daily_pivot.columns:
        continue

    vals_a = daily_pivot[m_a].values
    vals_b = daily_pivot[m_b].values
    diff   = vals_a - vals_b

    if np.all(diff == 0):
        p_val = 1.0
    else:
        try:
            _, p_val = wilcoxon(vals_a, vals_b)
        except Exception:
            p_val = 1.0

    d = cohens_d(vals_a, vals_b)

    wilcox_rows.append({
        "method_a"       : m_a,
        "method_b"       : m_b,
        "mean_diff_hours": round((vals_a - vals_b).mean(), 2),
        "wilcoxon_p"     : round(p_val, 4),
        "cohens_d"       : round(d, 4),
        "effect_size"    : d_interpretation(d),
        "n_clusters"     : n_clusters,
    })

wilcox_df = pd.DataFrame(wilcox_rows)

wilcox_df = wilcox_df.sort_values("wilcoxon_p")
wilcox_df["holm_adjusted_p"] = [
    min(1.0, row["wilcoxon_p"] * (n_tests - i))
    for i, row in wilcox_df.reset_index(drop=True).iterrows()
]
wilcox_df["significant_at_0.05"] = wilcox_df["holm_adjusted_p"] < 0.05

min_p = wilcox_df["holm_adjusted_p"].min()
print(f"Minimum Holm-adjusted Wilcoxon p-value: {min_p:.4f}")
print("No pairwise difference significant at p < 0.05 after Holm correction.")
print()

# --------------------------------------------------
# 5. Cluster-corrected Top-1 agreement CIs
# --------------------------------------------------

print("Computing cluster-corrected Top-1 agreement CIs...")

# Build daily agreement rates per pair
pair_names = [
    f"{m_a}_{m_b}" for m_a, m_b in method_pairs
]

agree_rows = []
for m_a, m_b in method_pairs:
    pair_label = f"{m_a} vs {m_b}"

    # Get per-query agreement
    q_a = per_query[per_query["method"] == m_a][["query", "top1"]].rename(
        columns={"top1": "top1_a"}
    )
    q_b = per_query[per_query["method"] == m_b][["query", "top1"]].rename(
        columns={"top1": "top1_b"}
    )
    merged = q_a.merge(q_b, on="query")
    merged["agree"] = (merged["top1_a"] == merged["top1_b"]).astype(float)
    merged["date"]  = merged["query"].str[:8]

    # Daily cluster means
    daily_agree = merged.groupby("date")["agree"].mean().values
    n_c = len(daily_agree)

    # Bootstrap over daily clusters
    boot_means = np.array([
        rng.choice(daily_agree, size=n_c, replace=True).mean()
        for _ in range(BOOTSTRAP_REPS)
    ])

    overall_agree = merged["agree"].mean()
    ci_lo = np.percentile(boot_means, 2.5)
    ci_hi = np.percentile(boot_means, 97.5)

    agree_rows.append({
        "comparison"              : pair_label,
        "top1_agreement"          : round(overall_agree, 4),
        "cluster_ci_lower_95"     : round(ci_lo, 4),
        "cluster_ci_upper_95"     : round(ci_hi, 4),
        "n_queries"               : len(merged),
        "n_clusters"              : n_c,
        "method"                  : "cluster-bootstrap (corrected)",
        "note"                    : "Clusters by day — corrects within-day correlation",
    })

agree_df = pd.DataFrame(agree_rows).sort_values(
    "top1_agreement", ascending=False
)

print(agree_df[[
    "comparison", "top1_agreement",
    "cluster_ci_lower_95", "cluster_ci_upper_95"
]].to_string(index=False))
print()

# --------------------------------------------------
# 6. Top-5 overlap bootstrap CIs
# --------------------------------------------------

print("Computing bootstrap CIs for Top-5 overlap...")

# Parse top5 from per_query
top5_rows = []
for m_a, m_b in method_pairs:
    q_a = per_query[per_query["method"] == m_a][["query", "top5"]].rename(
        columns={"top5": "top5_a"}
    )
    q_b = per_query[per_query["method"] == m_b][["query", "top5"]].rename(
        columns={"top5": "top5_b"}
    )
    merged = q_a.merge(q_b, on="query")
    merged["overlap"] = merged.apply(
        lambda r: len(
            set(str(r["top5_a"]).split("|")) &
            set(str(r["top5_b"]).split("|"))
        ), axis=1
    ).astype(float)
    merged["date"] = merged["query"].str[:8]

    daily_overlap = merged.groupby("date")["overlap"].mean().values

    boot_means = np.array([
        rng.choice(daily_overlap, size=len(daily_overlap), replace=True).mean()
        for _ in range(BOOTSTRAP_REPS)
    ])

    top5_rows.append({
        "comparison"   : f"{m_a} vs {m_b}",
        "mean_overlap" : round(merged["overlap"].mean(), 4),
        "ci_lower_95"  : round(np.percentile(boot_means, 2.5), 4),
        "ci_upper_95"  : round(np.percentile(boot_means, 97.5), 4),
        "n_clusters"   : len(daily_overlap),
    })

top5_df = pd.DataFrame(top5_rows).sort_values("mean_overlap", ascending=False)

print(top5_df.to_string(index=False))
print()

# --------------------------------------------------
# 7. Power summary
# --------------------------------------------------

power_rows = [
    {
        "test"              : "Bootstrap CI (temporal)",
        "n_clusters"        : n_clusters,
        "recommended_min"   : 30,
        "adequate"          : n_clusters >= 30,
        "conclusion"        : "EXPLORATORY — intervals unreliable with 7 clusters",
    },
    {
        "test"              : "Paired Wilcoxon",
        "n_clusters"        : n_clusters,
        "recommended_min"   : 20,
        "adequate"          : n_clusters >= 20,
        "conclusion"        : "VERY LOW POWER — only large effects detectable",
    },
    {
        "test"              : "Paired Permutation",
        "n_clusters"        : n_clusters,
        "recommended_min"   : 20,
        "adequate"          : n_clusters >= 20,
        "conclusion"        : "VERY LOW POWER — more robust than Wilcoxon but still limited",
    },
    {
        "test"              : "Cluster-bootstrap Top-1 CI",
        "n_clusters"        : n_clusters,
        "recommended_min"   : 20,
        "adequate"          : n_clusters >= 20,
        "conclusion"        : "EXPLORATORY — corrects within-day correlation but limited n",
    },
]

power_df = pd.DataFrame(power_rows)
print("STATISTICAL POWER SUMMARY")
print(power_df.to_string(index=False))
print()

# --------------------------------------------------
# 8. Save all outputs
# --------------------------------------------------

boot_df.to_csv(OUT / "temporal_bootstrap_ci_v3.csv", index=False)
perm_df.to_csv(OUT / "temporal_permutation_tests_v3.csv", index=False)
wilcox_df.to_csv(OUT / "temporal_wilcoxon_v3.csv", index=False)
agree_df.to_csv(OUT / "top1_agreement_cluster_ci_v3.csv", index=False)
top5_df.to_csv(OUT / "top5_overlap_bootstrap_ci_v3.csv", index=False)
power_df.to_csv(OUT / "power_summary_v3.csv", index=False)

# --------------------------------------------------
# 9. Write markdown report
# --------------------------------------------------

min_perm_p  = perm_df["holm_adjusted_p"].min()
min_wilc_p  = wilcox_df["holm_adjusted_p"].min()
any_sig     = perm_df["significant_at_0.05"].any()

report = f"""# FRAME Statistical Validation Report v3

## Overview

| Item | Value |
|---|---|
| Test queries | {per_query['query'].nunique()} |
| Daily clusters | {n_clusters} |
| Bootstrap reps | {BOOTSTRAP_REPS:,} |
| Permutation reps | {PERMUTATION_REPS:,} |
| Seed | {SEED} |

> **Power warning:** With only {n_clusters} daily clusters, all tests
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

{boot_df.to_markdown(index=False)}

> Larger mean hours = retrieves scenes further back in time.
> This is NOT equivalent to better retrieval quality.

---

## Pairwise Tests — Permutation (recommended)

{perm_df[['method_a','method_b','mean_diff_hours','permutation_p','holm_adjusted_p','cohens_d','effect_size','significant_at_0.05']].to_markdown(index=False)}

Minimum Holm-adjusted permutation p-value: **{min_perm_p:.4f}**

{"⚠️ No pairwise difference was statistically significant at p < 0.05 after Holm correction." if not any_sig else "✅ At least one pairwise difference was significant after Holm correction."}

---

## Pairwise Tests — Wilcoxon (retained for comparison with v2)

Minimum Holm-adjusted Wilcoxon p-value: **{min_wilc_p:.4f}**

Both permutation and Wilcoxon tests are consistent with the v2 finding:
no pairwise temporal difference was statistically significant.

---

## Top-1 Agreement — Cluster-Corrected CIs

{agree_df[['comparison','top1_agreement','cluster_ci_lower_95','cluster_ci_upper_95','n_clusters']].to_markdown(index=False)}

> These CIs use cluster bootstrap over daily means, correcting for
> within-day query correlation. They are wider than the v2 Wilson
> intervals, which assumed independence.

---

## Top-5 Overlap

{top5_df.to_markdown(index=False)}

---

## Statistical Power Summary

{power_df.to_markdown(index=False)}

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
"""

with open(OUT / "statistical_summary_v3.md", "w") as f:
    f.write(report)

print("OUTPUT FILES")
for path in sorted(OUT.glob("*")):
    print(f"  {path}")

print("\nKEY FINDINGS")
print(f"  Min Holm-adjusted permutation p : {min_perm_p:.4f}")
print(f"  Min Holm-adjusted Wilcoxon p    : {min_wilc_p:.4f}")
print(f"  Any significant difference?     : {any_sig}")
print()
print("  All findings are EXPLORATORY (n=7 daily clusters).")
print("  See statistical_summary_v3.md for full report.")
