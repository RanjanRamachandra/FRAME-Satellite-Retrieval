"""
Evaluate retrieval behaviour for all mask ablation configurations.

Loads each ablation checkpoint and runs the same retrieval evaluation
as the baseline experiment, then compares Top-1 diversity, pairwise
agreement, Top-5 overlap and temporal separation across configurations.

USAGE
-----
Run AFTER train_ablation_masks.py has completed:
    python scripts/evaluate_ablation_results.py

INPUT
-----
checkpoints/ablation_masks/frame_<shape>_<size>.pth
results/expanded_2022/final_test_per_query.csv   (baseline reference)
data/expanded_2022/metadata/expanded_dataset_manifest.csv

OUTPUT
------
results/ablation_masks/
    ablation_retrieval_rankings.csv   — Full ranked lists per config
    ablation_per_query.csv            — Top-1/Top-5 per query per config
    ablation_method_summary.csv       — Diversity + temporal summary
    ablation_vs_baseline.csv          — Agreement with baseline FRAME
    ablation_comparison_report.md     — Human-readable summary

INTERPRETATION NOTE
-------------------
These metrics compare retrieval BEHAVIOUR across configurations.
They do not establish which configuration retrieves more relevant scenes.
Relevance requires human annotations — use compute_retrieval_metrics.py
after collecting labels.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from expanded_satellite_dataset import ExpandedHimawariDataset
from models.se_res_autoencoder import SEResEncoder
from models.frame_model import FRAMEModel

# --------------------------------------------------
# Configuration
# --------------------------------------------------

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

AUTO_CHECKPOINT  = "checkpoints/expanded_2022/best_autoencoder.pth"
TEACHER_CHECKPOINT = "checkpoints/expanded_2022/best_masked_encoder.pth"
ABLATION_DIR     = Path("checkpoints/ablation_masks")
BASELINE_FILE    = Path("results/expanded_2022/final_test_per_query.csv")
OUT              = Path("results/ablation_masks")
OUT.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE    = 200
EMBEDDING_DIM = 512

print("FRAME ABLATION EVALUATION")
print(f"Device: {DEVICE}")
print()

# --------------------------------------------------
# Find ablation checkpoints
# --------------------------------------------------

checkpoints = sorted(ABLATION_DIR.glob("frame_*.pth"))
if not checkpoints:
    raise FileNotFoundError(
        f"No ablation checkpoints found in {ABLATION_DIR}\n"
        "Run train_ablation_masks.py first."
    )

print(f"Ablation checkpoints found: {len(checkpoints)}")
for c in checkpoints:
    print(f"  {c.name}")
print()

# --------------------------------------------------
# Load frozen encoders
# --------------------------------------------------

auto_ckpt    = torch.load(AUTO_CHECKPOINT,    map_location="cpu", weights_only=False)
teacher_ckpt = torch.load(TEACHER_CHECKPOINT, map_location="cpu", weights_only=False)

auto_encoder = SEResEncoder(EMBEDDING_DIM)
auto_state = {
    k.replace("encoder.", "", 1): v
    for k, v in auto_ckpt["model_state_dict"].items()
    if k.startswith("encoder.")
}
auto_encoder.load_state_dict(auto_state)
auto_encoder = auto_encoder.to(DEVICE).eval()
for p in auto_encoder.parameters():
    p.requires_grad = False

teacher_encoder = SEResEncoder(EMBEDDING_DIM)
teacher_encoder.load_state_dict(teacher_ckpt["encoder_state_dict"])
teacher_encoder = teacher_encoder.to(DEVICE).eval()
for p in teacher_encoder.parameters():
    p.requires_grad = False

print("Frozen encoders loaded.")

# --------------------------------------------------
# Load datasets
# --------------------------------------------------

train_dataset = ExpandedHimawariDataset(split="train", image_size=IMAGE_SIZE)
test_dataset  = ExpandedHimawariDataset(split="test",  image_size=IMAGE_SIZE)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=1, shuffle=False
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=1, shuffle=False
)

print(f"Database (train): {len(train_dataset)}")
print(f"Queries  (test) : {len(test_dataset)}")
print()

# --------------------------------------------------
# Build teacher database embeddings (shared across configs)
# --------------------------------------------------

print("Building teacher database embeddings...")
db_embeddings = {}
db_timestamps = []

with torch.no_grad():
    for images, filenames in train_loader:
        images = images.to(DEVICE)
        emb = teacher_encoder(images)
        emb = F.normalize(emb, p=2, dim=1)
        obs_id = Path(filenames[0]).stem
        db_embeddings[obs_id] = emb.squeeze(0).cpu()
        db_timestamps.append(obs_id)

db_ids   = list(db_embeddings.keys())
db_matrix = torch.stack([db_embeddings[k] for k in db_ids])  # [112, 512]

# --------------------------------------------------
# Deterministic mask per query (same as baseline)
# --------------------------------------------------

def make_deterministic_mask(query_id, image_size=200, min_side=40, max_side=100):
    """Deterministic square mask using query_id as seed."""
    seed = sum(ord(c) for c in query_id)
    rng = torch.Generator()
    rng.manual_seed(seed)
    side = torch.randint(min_side, max_side + 1, (1,), generator=rng).item()
    top  = torch.randint(0, image_size - side + 1, (1,), generator=rng).item()
    left = torch.randint(0, image_size - side + 1, (1,), generator=rng).item()
    mask = torch.ones(1, 1, image_size, image_size)
    mask[:, :, top:top + side, left:left + side] = 0
    return mask, side


def make_ablation_mask(query_id, shape, min_side, max_side, image_size=200):
    """Deterministic mask for ablation configuration."""
    seed = sum(ord(c) for c in query_id) + hash(shape) % 1000
    torch.manual_seed(seed)

    mask = torch.ones(1, 1, image_size, image_size)

    if shape == "square":
        side = min_side + (seed % (max_side - min_side + 1))
        side = min(side, max_side)
        top  = seed % (image_size - side)
        left = (seed * 7) % (image_size - side)
        mask[:, :, top:top + side, left:left + side] = 0

    elif shape == "circle":
        radius = (min_side + (seed % (max_side - min_side + 1))) // 2
        radius = max(1, min(radius, image_size // 2 - 1))
        cy = radius + (seed % (image_size - 2 * radius))
        cx = radius + ((seed * 7) % (image_size - 2 * radius))
        y_grid = torch.arange(image_size).unsqueeze(1).float()
        x_grid = torch.arange(image_size).unsqueeze(0).float()
        dist = ((y_grid - cy) ** 2 + (x_grid - cx) ** 2).sqrt()
        mask[0, 0][dist <= radius] = 0

    elif shape == "rectangle":
        short = min_side + (seed % (max_side - min_side + 1))
        short = min(short, max_side)
        long  = min(short * 2, image_size)
        if seed % 2 == 0:
            h, w = long, short
        else:
            h, w = short, long
        top  = seed % max(1, (image_size - h))
        left = (seed * 7) % max(1, (image_size - w))
        mask[:, :, top:top + h, left:left + w] = 0

    return mask


def parse_timestamp(obs_id):
    """Parse observation ID to hours since epoch for temporal comparison."""
    try:
        date_str = obs_id[:8]
        time_str = obs_id[9:]
        year  = int(date_str[:4])
        month = int(date_str[4:6])
        day   = int(date_str[6:8])
        hour  = int(time_str[:2])
        minute = int(time_str[2:]) if len(time_str) > 2 else 0
        # Simple hours since 2022-01-01
        day_of_year = (month - 1) * 30 + day
        return (year - 2022) * 8760 + day_of_year * 24 + hour + minute / 60
    except Exception:
        return 0.0


# --------------------------------------------------
# Evaluate each ablation configuration
# --------------------------------------------------

all_rankings = []
all_per_query = []

for ckpt_path in checkpoints:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config  = ckpt["configuration"]
    shape   = ckpt["shape"]
    size_nm = ckpt["size_name"]
    min_s   = ckpt["min_side"]
    max_s   = ckpt["max_side"]

    print(f"Evaluating: {config}")

    model = FRAMEModel(EMBEDDING_DIM).to(DEVICE)
    model.load_state_dict(ckpt["frame_model_state_dict"])
    model.eval()

    with torch.no_grad():
        for images, filenames in test_loader:
            images   = images.to(DEVICE)
            query_id = Path(filenames[0]).stem

            # Build query embedding
            z_full = auto_encoder(images)
            mask   = make_ablation_mask(
                query_id, shape, min_s, max_s
            ).to(DEVICE)
            z_pred = F.normalize(model(z_full, mask), p=2, dim=1)

            # Score against database
            scores = (db_matrix.to(DEVICE) @ z_pred.T).squeeze()
            ranked_indices = scores.argsort(descending=True).cpu().tolist()

            top1_id = db_ids[ranked_indices[0]]
            top5_ids = [db_ids[i] for i in ranked_indices[:5]]

            q_hours  = parse_timestamp(query_id)
            t1_hours = parse_timestamp(top1_id)
            temporal_dist = abs(q_hours - t1_hours)

            all_per_query.append({
                "configuration": config,
                "shape": shape,
                "size": size_nm,
                "query": query_id,
                "top1": top1_id,
                "top5": "|".join(top5_ids),
                "temporal_dist_hours": round(temporal_dist, 2),
            })

            for rank, idx in enumerate(ranked_indices, 1):
                all_rankings.append({
                    "configuration": config,
                    "query": query_id,
                    "rank": rank,
                    "retrieved_image": db_ids[idx],
                    "score": round(scores[idx].item(), 8),
                })

    print(f"  Done: {len(test_dataset)} queries evaluated.")

# --------------------------------------------------
# Build DataFrames
# --------------------------------------------------

rankings_df  = pd.DataFrame(all_rankings)
per_query_df = pd.DataFrame(all_per_query)

# --------------------------------------------------
# Method summary — diversity and temporal
# --------------------------------------------------

summary_rows = []
configs = per_query_df["configuration"].unique()

for config in configs:
    cfg_df = per_query_df[per_query_df["configuration"] == config]
    summary_rows.append({
        "configuration": config,
        "shape": cfg_df["shape"].iloc[0],
        "size": cfg_df["size"].iloc[0],
        "unique_top1": cfg_df["top1"].nunique(),
        "mean_temporal_hours": round(cfg_df["temporal_dist_hours"].mean(), 2),
        "median_temporal_hours": round(cfg_df["temporal_dist_hours"].median(), 2),
    })

summary_df = pd.DataFrame(summary_rows).sort_values("mean_temporal_hours")

# --------------------------------------------------
# Agreement with baseline FRAME
# --------------------------------------------------

baseline_df = pd.read_csv(BASELINE_FILE)
baseline_frame = baseline_df[baseline_df["method"] == "FRAME"][
    ["query", "top1", "top5"]
].rename(columns={"top1": "baseline_top1", "top5": "baseline_top5"})

agreement_rows = []
for config in configs:
    cfg_df = per_query_df[per_query_df["configuration"] == config].merge(
        baseline_frame, on="query", how="inner"
    )
    top1_agree = (cfg_df["top1"] == cfg_df["baseline_top1"]).mean()

    top5_overlaps = []
    for _, row in cfg_df.iterrows():
        ablation_set = set(row["top5"].split("|"))
        baseline_set = set(row["baseline_top5"].split("|"))
        top5_overlaps.append(len(ablation_set & baseline_set))

    agreement_rows.append({
        "configuration": config,
        "top1_agreement_vs_baseline": round(top1_agree, 4),
        "mean_top5_overlap_vs_baseline": round(np.mean(top5_overlaps), 4),
        "n_queries": len(cfg_df),
    })

agreement_df = pd.DataFrame(agreement_rows).sort_values(
    "top1_agreement_vs_baseline", ascending=False
)

# --------------------------------------------------
# Save outputs
# --------------------------------------------------

rankings_df.to_csv(OUT / "ablation_retrieval_rankings.csv", index=False)
per_query_df.to_csv(OUT / "ablation_per_query.csv", index=False)
summary_df.to_csv(OUT / "ablation_method_summary.csv", index=False)
agreement_df.to_csv(OUT / "ablation_vs_baseline.csv", index=False)

# --------------------------------------------------
# Write comparison report
# --------------------------------------------------

report_lines = [
    "# FRAME Mask Ablation — Comparison Report",
    "",
    "## Overview",
    "",
    f"- Configurations evaluated: {len(configs)}",
    f"- Test queries: {per_query_df['query'].nunique()}",
    f"- Database size: {len(db_ids)}",
    "",
    "> **Interpretation note:** These metrics compare retrieval behaviour,",
    "> not retrieval quality. Relevance requires human annotations.",
    "> Use compute_retrieval_metrics.py after collecting labels.",
    "",
    "## Retrieval Diversity and Temporal Separation",
    "",
    summary_df.to_markdown(index=False),
    "",
    "## Agreement with Baseline FRAME (square, medium)",
    "",
    agreement_df.to_markdown(index=False),
    "",
    "## Interpretation",
    "",
    "- **Unique Top-1 matches**: higher = more diverse retrieval.",
    "  Does NOT mean better relevance.",
    "- **Temporal separation**: larger = retrieves scenes further back in time.",
    "  Does NOT mean better relevance.",
    "- **Top-1 agreement vs baseline**: how often this config agrees with",
    "  the baseline square-medium FRAME. Agreement ≠ correctness.",
    "",
    "## Next Steps",
    "",
    "1. Collect human relevance labels using create_improved_visual_review.py",
    "2. Verify inter-annotator agreement with inter_annotator_agreement.py",
    "3. Compute Precision@K, Recall@K, NDCG@K with compute_retrieval_metrics.py",
    "4. Only then compare configurations on retrieval quality.",
]

report_path = OUT / "ablation_comparison_report.md"
with open(report_path, "w") as f:
    f.write("\n".join(report_lines))

# --------------------------------------------------
# Print summary
# --------------------------------------------------

print("\nABLATION SUMMARY — Diversity and Temporal Separation")
print("=" * 60)
print(summary_df.to_string(index=False))

print("\nAGREEMENT VS BASELINE FRAME")
print("=" * 60)
print(agreement_df.to_string(index=False))

print("\nOUTPUT FILES")
for path in sorted(OUT.glob("*")):
    print(f"  {path}")

print("\nREMINDER")
print("  These metrics describe behaviour, not quality.")
print("  Run compute_retrieval_metrics.py after collecting annotations.")
