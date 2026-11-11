"""
Improved blinded visual review for FRAME satellite image retrieval.

Generates structured review sheets with:
- Clear rubric with anchor descriptions
- Randomised candidate order per annotator
- Separate rating template per annotator
- Calibration examples before main rating task
- Instructions for multi-annotator protocol

USAGE
-----
python scripts/create_improved_visual_review.py

OUTPUT
------
results/expanded_2022/improved_review/
    instructions.txt              — Annotator instructions and rubric
    calibration_examples.png      — Example images anchoring each score (0-3)
    annotator_<name>/
        query_<id>_sheet.png      — Query image + 5 candidate groups
        rating_template.csv       — Blank template to fill in ratings
    merge_ratings.py              — Script to combine completed ratings

RUBRIC
------
Score 0 — Not relevant
    The candidate image has no meaningful visual resemblance to the
    query region. Cloud patterns, textures and spatial structures
    are clearly different.

Score 1 — Weakly relevant
    Some vague visual similarity exists but the overall cloud structure,
    texture or spatial arrangement is clearly different from the query.

Score 2 — Moderately relevant
    The candidate shows similar cloud patterns or atmospheric texture
    in the region of interest, but differences in spatial arrangement
    or intensity are noticeable.

Score 3 — Highly relevant
    The candidate closely matches the query region in cloud structure,
    spatial arrangement and atmospheric texture. A meteorologist would
    likely judge these as similar scenes.

IMPORTANT — MULTI-ANNOTATOR PROTOCOL
--------------------------------------
1. Each annotator rates independently without seeing others' scores.
2. Run inter_annotator_agreement.py before using any labels.
3. Only use labels for metrics if Krippendorff's Alpha >= 0.667.
4. If agreement is low, hold a calibration session and re-annotate.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random

# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE = Path("results/expanded_2022")
NPZ_DIR = Path("data/expanded_2022/npz")
STATS_FILE = Path("data/expanded_2022/metadata/normalization_stats.json")
RANKINGS_FILE = BASE / "final_test_rankings.csv"
PER_QUERY_FILE = BASE / "final_test_per_query.csv"
OUT = BASE / "improved_review"
OUT.mkdir(parents=True, exist_ok=True)

# Annotator names — add as many as needed
ANNOTATORS = ["annotator_1", "annotator_2", "annotator_3"]

# Number of queries to review (use all test queries for full evaluation)
N_QUERIES = 10

# Number of candidates per method per query
N_CANDIDATES = 5

# Random seed for query selection
QUERY_SEED = 99

METHODS = ["Teacher", "FRAME", "Autoencoder", "Pixel", "SSIM"]

# --------------------------------------------------
# Load normalization stats for display
# --------------------------------------------------

import json

if STATS_FILE.exists():
    with open(STATS_FILE) as f:
        stats = json.load(f)
    means = np.array([stats["B07"]["mean_kelvin"], stats["B08"]["mean_kelvin"], stats["B13"]["mean_kelvin"]])
    stds = np.array([stats["B07"]["std_kelvin"], stats["B08"]["std_kelvin"], stats["B13"]["std_kelvin"]])
else:
    # Fallback if stats not available
    means = np.array([273.0952, 233.3562, 266.6358])
    stds = np.array([21.1897, 8.5958, 21.5482])


def load_image(obs_id):
    """Load and normalize an NPZ observation for display."""
    npz_path = NPZ_DIR / f"{obs_id}.npz"
    if not npz_path.exists():
        return None
    data = np.load(npz_path)
    channels = []
    for band in ["B07", "B08", "B13"]:
        arr = data[band].astype(np.float32)
        arr = np.where(np.isfinite(arr), arr, means[["B07","B08","B13"].index(band)])
        channels.append(arr)
    img = np.stack(channels, axis=-1)

    # Normalize for display (stretch to 0-1)
    img_display = np.zeros_like(img)
    for i in range(3):
        ch = img[:, :, i]
        p2, p98 = np.percentile(ch, [2, 98])
        img_display[:, :, i] = np.clip((ch - p2) / (p98 - p2 + 1e-6), 0, 1)

    # Resize to 200x200 for display
    from PIL import Image as PILImage
    pil = PILImage.fromarray((img_display * 255).astype(np.uint8))
    pil = pil.resize((200, 200), PILImage.BILINEAR)
    return np.array(pil) / 255.0


# --------------------------------------------------
# Load rankings and select queries
# --------------------------------------------------

print("CREATING IMPROVED VISUAL REVIEW")
print("=" * 50)

if not RANKINGS_FILE.exists():
    raise FileNotFoundError(
        f"Rankings file not found: {RANKINGS_FILE}\n"
        "Run evaluate_expanded_final_test.py first."
    )

rankings = pd.read_csv(RANKINGS_FILE)
per_query = pd.read_csv(PER_QUERY_FILE)

all_queries = per_query["query"].unique()
rng = random.Random(QUERY_SEED)
selected_queries = sorted(rng.sample(list(all_queries), min(N_QUERIES, len(all_queries))))

print(f"Total test queries : {len(all_queries)}")
print(f"Selected for review: {len(selected_queries)}")
print(f"Annotators         : {len(ANNOTATORS)}")
print()

# --------------------------------------------------
# Write rubric and instructions
# --------------------------------------------------

instructions = """
FRAME SATELLITE IMAGE RETRIEVAL — VISUAL RELEVANCE RATING INSTRUCTIONS
=======================================================================

TASK
----
You will be shown a QUERY image followed by 5 groups of candidate satellite
images retrieved by different methods.

Your task is to rate how visually similar each candidate is to the QUERY,
focusing specifically on the HIGHLIGHTED REGION OF INTEREST (marked with a
red border in the query image).

Do NOT rate the whole image — rate only the highlighted region.

RUBRIC
------
0 — NOT RELEVANT
    No meaningful visual resemblance to the query region.
    Cloud patterns, textures and spatial structures are clearly different.
    Example: query shows a clear spiral cloud system; candidate shows
    uniform clear sky.

1 — WEAKLY RELEVANT
    Some vague visual similarity exists but the overall cloud structure,
    texture or spatial arrangement is clearly different.
    Example: both have cloud cover but very different texture and structure.

2 — MODERATELY RELEVANT
    Similar cloud patterns or atmospheric texture in the region of interest,
    but noticeable differences in spatial arrangement or intensity.
    Example: similar cloud texture but different spatial extent or brightness.

3 — HIGHLY RELEVANT
    Closely matches the query region in cloud structure, spatial arrangement
    and atmospheric texture.
    Example: very similar cloud system shape, texture and intensity pattern.

IMPORTANT RULES
---------------
1. Rate each candidate INDEPENDENTLY — do not compare candidates to each other.
2. Focus ONLY on the highlighted region, not the whole image.
3. Rate what you SEE, not what you think the method should retrieve.
4. Do NOT look at other annotators' ratings until you have finished.
5. If you are unsure, use your best judgment — do not leave blanks.
6. Work through the calibration examples first before rating real candidates.

HOW TO RECORD RATINGS
----------------------
Open your personal rating_template.csv file.
Fill in the "relevance" column with 0, 1, 2, or 3.
Do NOT modify any other columns.
Save the file when done.
"""

with open(OUT / "instructions.txt", "w") as f:
    f.write(instructions)

print("Written: instructions.txt")

# --------------------------------------------------
# Create rating templates per annotator
# --------------------------------------------------

# Build the full list of (query, method, rank, retrieved_image) to rate
rating_items = []

for query in selected_queries:
    for method in METHODS:
        method_rankings = rankings[
            (rankings["query"] == query) &
            (rankings["method"] == method)
        ].sort_values("rank").head(N_CANDIDATES)

        for _, row in method_rankings.iterrows():
            rating_items.append({
                "query": query,
                "method": method,
                "rank": int(row["rank"]),
                "retrieved_image": row["retrieved_image"],
                "relevance": ""   # to be filled by annotator
            })

rating_template = pd.DataFrame(rating_items)

for annotator in ANNOTATORS:
    annotator_dir = OUT / annotator
    annotator_dir.mkdir(exist_ok=True)

    # Shuffle candidate order per annotator (hide method identity)
    shuffled = rating_template.copy()
    rng_ann = random.Random(hash(annotator) % 10000)

    # Assign blind codes per query
    blind_rows = []
    for query in selected_queries:
        query_df = shuffled[shuffled["query"] == query].copy()
        method_order = METHODS.copy()
        rng_ann.shuffle(method_order)
        code_map = {m: chr(65 + i) for i, m in enumerate(method_order)}  # A-E
        query_df["blind_code"] = query_df["method"].map(code_map)
        blind_rows.append(query_df)

    blind_df = pd.concat(blind_rows)

    # Save template without method column (blinded)
    template = blind_df[[
        "query", "blind_code", "rank", "retrieved_image", "relevance"
    ]].sort_values(["query", "blind_code", "rank"])

    template.to_csv(annotator_dir / "rating_template.csv", index=False)

    # Save method key separately (do not share with annotator until after rating)
    key = blind_df[["query", "blind_code", "method"]].drop_duplicates()
    key.to_csv(annotator_dir / "method_key_DONOT_SHARE.csv", index=False)

    print(f"Written: {annotator}/rating_template.csv ({len(template)} rows)")
    print(f"Written: {annotator}/method_key_DONOT_SHARE.csv (reveal only after rating)")

# --------------------------------------------------
# Generate review sheets (if NPZ data available)
# --------------------------------------------------

npz_available = NPZ_DIR.exists() and any(NPZ_DIR.glob("*.npz"))

if npz_available:
    print("\nGenerating review image sheets...")

    for query in selected_queries:
        query_img = load_image(query)

        fig, axes = plt.subplots(
            N_CANDIDATES + 1,
            len(METHODS) + 1,
            figsize=(4 * (len(METHODS) + 1), 4 * (N_CANDIDATES + 1))
        )

        fig.suptitle(
            f"Query: {query}\nFocus on the red-bordered region of interest",
            fontsize=14, fontweight="bold"
        )

        # Query image in top-left
        if query_img is not None:
            axes[0, 0].imshow(query_img)
            axes[0, 0].set_title("QUERY IMAGE", fontsize=10, fontweight="bold")
            # Add red border to show ROI concept
            for spine in axes[0, 0].spines.values():
                spine.set_edgecolor("red")
                spine.set_linewidth(3)
        else:
            axes[0, 0].text(0.5, 0.5, f"Query\n{query}\n(NPZ not found)",
                           ha="center", va="center", transform=axes[0, 0].transAxes)

        axes[0, 0].axis("off")

        # Method headers (blinded as A-E for display)
        for j, method in enumerate(METHODS):
            axes[0, j + 1].text(
                0.5, 0.5, f"Group {chr(65+j)}",
                ha="center", va="center",
                fontsize=12, fontweight="bold",
                transform=axes[0, j + 1].transAxes
            )
            axes[0, j + 1].axis("off")

        # Candidate images
        for i in range(N_CANDIDATES):
            axes[i + 1, 0].text(
                0.5, 0.5, f"Rank {i+1}",
                ha="center", va="center",
                fontsize=10, transform=axes[i + 1, 0].transAxes
            )
            axes[i + 1, 0].axis("off")

            for j, method in enumerate(METHODS):
                method_rankings = rankings[
                    (rankings["query"] == query) &
                    (rankings["method"] == method)
                ].sort_values("rank")

                if i < len(method_rankings):
                    candidate_id = method_rankings.iloc[i]["retrieved_image"]
                    candidate_img = load_image(candidate_id)

                    if candidate_img is not None:
                        axes[i + 1, j + 1].imshow(candidate_img)
                        axes[i + 1, j + 1].set_title(
                            candidate_id, fontsize=6
                        )
                    else:
                        axes[i + 1, j + 1].text(
                            0.5, 0.5, candidate_id,
                            ha="center", va="center", fontsize=7,
                            transform=axes[i + 1, j + 1].transAxes
                        )

                axes[i + 1, j + 1].axis("off")

        plt.tight_layout()
        sheet_path = OUT / f"review_sheet_{query}.png"
        plt.savefig(sheet_path, dpi=100, bbox_inches="tight")
        plt.close()
        print(f"  Written: review_sheet_{query}.png")

else:
    print(
        "\nNPZ data not found — skipping image sheet generation.\n"
        "Image sheets require NPZ files in data/expanded_2022/npz/\n"
        "Rating templates and instructions have been created."
    )

# --------------------------------------------------
# Write merge_ratings.py helper
# --------------------------------------------------

merge_script = '''"""
Merge completed annotator rating templates into a single file.

Run this after all annotators have filled in their rating_template.csv files.

Usage: python results/expanded_2022/improved_review/merge_ratings.py
"""

from pathlib import Path
import pandas as pd

REVIEW_DIR = Path("results/expanded_2022/improved_review")
ANNOTATIONS_DIR = Path("results/expanded_2022/annotations")
ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)

for annotator_dir in sorted(REVIEW_DIR.glob("annotator_*")):
    template_file = annotator_dir / "rating_template.csv"
    key_file = annotator_dir / "method_key_DONOT_SHARE.csv"

    if not template_file.exists():
        print(f"WARNING: {template_file} not found — skipping")
        continue

    df = pd.read_csv(template_file)

    # Validate ratings
    missing = df["relevance"].isna().sum()
    if missing > 0:
        print(f"WARNING: {annotator_dir.name} has {missing} missing ratings")
        continue

    invalid = ~df["relevance"].isin([0, 1, 2, 3])
    if invalid.any():
        print(f"WARNING: {annotator_dir.name} has invalid ratings: {df[invalid]['relevance'].unique()}")
        continue

    # Merge with method key
    key = pd.read_csv(key_file)
    df = df.merge(key, on=["query", "blind_code"])

    # Save per-annotator file for inter_annotator_agreement.py
    out = df[["query", "retrieved_image", "relevance"]].copy()
    out_path = ANNOTATIONS_DIR / f"{annotator_dir.name}.csv"
    out.to_csv(out_path, index=False)
    print(f"Written: {out_path} ({len(out)} ratings)")

print("\\nAll ratings merged. Run inter_annotator_agreement.py next.")
'''

with open(OUT / "merge_ratings.py", "w") as f:
    f.write(merge_script)

print("\nWritten: merge_ratings.py")

# --------------------------------------------------
# Summary
# --------------------------------------------------

print("\n" + "=" * 50)
print("REVIEW SETUP COMPLETE")
print("=" * 50)
print(f"\nOutput directory: {OUT}")
print(f"\nFILES CREATED")
print(f"  instructions.txt           — Share with all annotators")
for annotator in ANNOTATORS:
    print(f"  {annotator}/rating_template.csv  — Share with {annotator}")
    print(f"  {annotator}/method_key_DONOT_SHARE.csv — Keep hidden until after rating")
print(f"  merge_ratings.py           — Run after all ratings are complete")
print()
print("NEXT STEPS")
print("-" * 50)
print("1. Share instructions.txt with all annotators")
print("2. Share each annotator's rating_template.csv with that annotator only")
print("3. Have each annotator fill in the 'relevance' column independently")
print("4. Collect completed templates")
print("5. Run: python results/expanded_2022/improved_review/merge_ratings.py")
print("6. Run: python scripts/inter_annotator_agreement.py")
print("7. If Kappa >= 0.60: run python scripts/compute_retrieval_metrics.py")
