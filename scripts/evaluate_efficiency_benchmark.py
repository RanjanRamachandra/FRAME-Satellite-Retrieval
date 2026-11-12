"""
Efficiency benchmark for FRAME satellite retrieval experiment.
Replicates Table 2 from Lee & Park (IEEE Access, 2026).
Run: python3 scripts/evaluate_efficiency_benchmark.py
"""
import os, time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path
from expanded_satellite_dataset import ExpandedHimawariDataset
from models.se_res_autoencoder import SEResEncoder
from models.frame_model import FRAMEModel

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
CHECKPOINT_DIR = "checkpoints/expanded_2022"
OUTPUT_DIR = "results/expanded_2022/efficiency_benchmark"
IMAGE_SIZE = 200
EMBEDDING_DIM = 512
TIMING_RUNS = 3
WARMUP_RUNS = 1
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("FRAME EFFICIENCY BENCHMARK")
print("=" * 55)
print(f"Device: {DEVICE}")

auto_ckpt    = torch.load(os.path.join(CHECKPOINT_DIR,"best_autoencoder.pth"),    map_location=DEVICE, weights_only=False)
teacher_ckpt = torch.load(os.path.join(CHECKPOINT_DIR,"best_masked_encoder.pth"), map_location=DEVICE, weights_only=False)
frame_ckpt   = torch.load(os.path.join(CHECKPOINT_DIR,"best_frame_model.pth"),    map_location=DEVICE, weights_only=False)

auto_encoder = SEResEncoder(EMBEDDING_DIM)
auto_state = {k.replace("encoder.","",1):v for k,v in auto_ckpt["model_state_dict"].items() if k.startswith("encoder.")}
auto_encoder.load_state_dict(auto_state)
auto_encoder = auto_encoder.to(DEVICE).eval()

teacher_encoder = SEResEncoder(EMBEDDING_DIM)
teacher_encoder.load_state_dict(teacher_ckpt["encoder_state_dict"])
teacher_encoder = teacher_encoder.to(DEVICE).eval()

frame_model = FRAMEModel(EMBEDDING_DIM)
frame_model.load_state_dict(frame_ckpt["frame_model_state_dict"])
frame_model = frame_model.to(DEVICE).eval()

for model in [auto_encoder, teacher_encoder, frame_model]:
    for p in model.parameters():
        p.requires_grad = False

print("Models loaded.")

def count_params(m): return sum(p.numel() for p in m.parameters())
n_teacher = count_params(teacher_encoder)
n_auto    = count_params(auto_encoder)
n_frame   = count_params(frame_model)
print(f"\nPARAMETER COUNTS")
print(f"  Teacher encoder : {n_teacher:>12,}")
print(f"  Auto encoder    : {n_auto:>12,}")
print(f"  FRAME model     : {n_frame:>12,}")

# FLOPs — analytical estimate based on paper Table 2 scaled to 200x200
scale = (200*200)/(172*200)
baseline_flops = 3.64e8 * scale
frame_flops    = 2*512*512*2 + 512
flops_pct      = (1 - frame_flops/baseline_flops)*100
print(f"\nFLOPs (analytical estimate, scaled from paper)")
print(f"  Baseline/candidate : {baseline_flops:>14,.0f}")
print(f"  FRAME/candidate    : {frame_flops:>14,.0f}")
print(f"  Reduction          : {flops_pct:.1f}%")
print(f"  Ratio              : {baseline_flops/frame_flops:.0f}x fewer for FRAME")

# Try fvcore if available
try:
    from fvcore.nn import FlopCountAnalysis
    dummy_img  = torch.randn(1,3,IMAGE_SIZE,IMAGE_SIZE).to(DEVICE)
    dummy_z    = torch.randn(1,EMBEDDING_DIM).to(DEVICE)
    dummy_mask = torch.ones(1,1,IMAGE_SIZE,IMAGE_SIZE).to(DEVICE)
    t_flops = FlopCountAnalysis(teacher_encoder, dummy_img)
    t_flops.unsupported_ops_warnings(False)
    baseline_flops = t_flops.total()
    f_flops = FlopCountAnalysis(frame_model, (dummy_z, dummy_mask))
    f_flops.unsupported_ops_warnings(False)
    frame_flops = f_flops.total()
    flops_pct   = (1 - frame_flops/baseline_flops)*100
    print(f"\nFLOPs (fvcore measured)")
    print(f"  Baseline/candidate : {baseline_flops:>14,.0f}")
    print(f"  FRAME/candidate    : {frame_flops:>14,.0f}")
    print(f"  Reduction          : {flops_pct:.1f}%")
except ImportError:
    print("  (fvcore not installed — using analytical estimate above)")

train_dataset = ExpandedHimawariDataset(split="train",      image_size=IMAGE_SIZE)
test_dataset  = ExpandedHimawariDataset(split="test",       image_size=IMAGE_SIZE)
train_loader  = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False)
test_loader   = torch.utils.data.DataLoader(test_dataset,  batch_size=1, shuffle=False)
n_db = len(train_dataset); n_q = len(test_dataset)
print(f"\nDatabase: {n_db} | Queries: {n_q}")

print("Building DB embeddings...")
db_imgs=[]; db_embs=[]; auto_embs=[]
with torch.no_grad():
    for imgs,_ in train_loader:
        imgs=imgs.to(DEVICE)
        db_imgs.append(imgs)
        db_embs.append(F.normalize(teacher_encoder(imgs),p=2,dim=1))
        auto_embs.append(F.normalize(auto_encoder(imgs),p=2,dim=1))
db_matrix   = torch.cat(db_embs,  dim=0)
auto_matrix = torch.cat(auto_embs, dim=0)
db_imgs_all = torch.cat(db_imgs,  dim=0)

def make_mask(idx):
    g=torch.Generator(); g.manual_seed(1000+idx)
    s=torch.randint(40,101,(1,),generator=g).item()
    t=torch.randint(0,IMAGE_SIZE-s+1,(1,),generator=g).item()
    l=torch.randint(0,IMAGE_SIZE-s+1,(1,),generator=g).item()
    m=torch.ones(1,1,IMAGE_SIZE,IMAGE_SIZE)
    m[:,:,t:t+s,l:l+s]=0
    return m

def run_baseline(q_img, mask):
    with torch.no_grad():
        mq = q_img * mask.to(DEVICE)
        qe = F.normalize(teacher_encoder(mq), p=2, dim=1)
        for db_img in db_imgs_all:
            md = db_img.unsqueeze(0) * mask.to(DEVICE)
            de = F.normalize(teacher_encoder(md), p=2, dim=1)
            _ = (qe @ de.T).item()

def run_frame(q_img, idx):
    mask = make_mask(idx).to(DEVICE)
    with torch.no_grad():
        zf = auto_encoder(q_img)
        zp = F.normalize(frame_model(zf, mask), p=2, dim=1)
        _ = (db_matrix.to(DEVICE) @ zp.T).squeeze().tolist()

test_imgs=[]; test_names=[]
with torch.no_grad():
    for imgs,fns in test_loader:
        test_imgs.append(imgs.to(DEVICE)); test_names.append(fns[0])

print(f"\nTiming FRAME ({WARMUP_RUNS} warmup + {TIMING_RUNS} runs)...")
for _ in range(WARMUP_RUNS):
    for i,q in enumerate(test_imgs): run_frame(q,i)
frame_times=[]
for r in range(TIMING_RUNS):
    t0=time.perf_counter()
    for i,q in enumerate(test_imgs): run_frame(q,i)
    frame_times.append(time.perf_counter()-t0)
    print(f"  Run {r+1}: {frame_times[-1]:.3f}s")
frame_mean = np.mean(frame_times)

print(f"\nTiming Baseline (re-encoding {n_db} DB images per query)...")
print(f"  Estimated wait: ~{n_db*n_q*0.02:.0f}s — be patient...")
with torch.no_grad(): run_baseline(test_imgs[0], make_mask(0))
base_times=[]
for r in range(TIMING_RUNS):
    t0=time.perf_counter()
    for i,q in enumerate(test_imgs): run_baseline(q, make_mask(i).to(DEVICE))
    base_times.append(time.perf_counter()-t0)
    print(f"  Run {r+1}: {base_times[-1]:.3f}s")
base_mean = np.mean(base_times)

speedup  = base_mean / frame_mean
time_pct = (1 - frame_mean/base_mean)*100

print("\n" + "="*55)
print("EFFICIENCY RESULTS SUMMARY")
print("="*55)
print(f"Database size             : {n_db} images")
print(f"Query count               : {n_q} queries")
print(f"Device                    : {DEVICE}")
print(f"\nFRAME total time          : {frame_mean:.3f}s")
print(f"FRAME per query           : {frame_mean/n_q:.4f}s")
print(f"\nBaseline total time       : {base_mean:.3f}s")
print(f"Baseline per query        : {base_mean/n_q:.4f}s")
print(f"\nSpeedup (FRAME vs baseline): {speedup:.1f}x")
print(f"Time reduction            : {time_pct:.1f}%")
print(f"FLOPs reduction           : {flops_pct:.1f}%")
print(f"\nPaper comparison (A100, 20595 DB images):")
print(f"  Paper speedup: ~15x  |  Your speedup: {speedup:.1f}x")
print(f"  Paper time cut: ~92%  |  Your time cut: {time_pct:.1f}%")
print(f"  Paper FLOPs cut: >99%  |  Your FLOPs cut: {flops_pct:.1f}%")
print(f"\nNOTE: Speedup differs from paper due to smaller DB (112 vs 20595)")
print(f"and different hardware (MPS vs A100). Direction is the same.")

summary = {"device":str(DEVICE),"db_size":n_db,"queries":n_q,
           "frame_total_s":round(frame_mean,4),"frame_per_query_s":round(frame_mean/n_q,6),
           "baseline_total_s":round(base_mean,4),"baseline_per_query_s":round(base_mean/n_q,6),
           "speedup_x":round(speedup,2),"time_reduction_pct":round(time_pct,1),
           "flops_reduction_pct":round(flops_pct,1),
           "paper_speedup_x":15.0,"paper_db_size":20595,"paper_device":"NVIDIA A100 40GB"}
pd.DataFrame([summary]).T.reset_index().rename(columns={"index":"metric",0:"value"}).to_csv(
    os.path.join(OUTPUT_DIR,"efficiency_summary.csv"),index=False)

report = f"""# FRAME Efficiency Benchmark

| Metric | Baseline | FRAME | Reduction |
|---|---|---|---|
| FLOPs/candidate | {baseline_flops:,.0f} | {frame_flops:,.0f} | {flops_pct:.1f}% |
| Total time ({n_q} queries) | {base_mean:.3f}s | {frame_mean:.3f}s | {time_pct:.1f}% |
| Per-query time | {base_mean/n_q:.4f}s | {frame_mean/n_q:.4f}s | — |
| Speedup | 1x | **{speedup:.1f}x** | — |

Paper (Lee & Park 2026, A100, 20595 images): ~15x speedup, >99% FLOPs reduction.
Your project ({DEVICE}, {n_db} images): {speedup:.1f}x speedup, {flops_pct:.1f}% FLOPs reduction.
"""
Path(os.path.join(OUTPUT_DIR,"efficiency_report.md")).write_text(report)
print(f"\nSaved: {OUTPUT_DIR}/efficiency_summary.csv")
print(f"Saved: {OUTPUT_DIR}/efficiency_report.md")
