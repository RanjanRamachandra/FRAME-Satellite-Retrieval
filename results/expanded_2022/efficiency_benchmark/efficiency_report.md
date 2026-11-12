# FRAME Efficiency Benchmark

| Metric | Baseline | FRAME | Reduction |
|---|---|---|---|
| FLOPs/candidate | 423,255,814 | 1,049,088 | 99.8% |
| Total time (56 queries) | 19.706s | 0.349s | 98.2% |
| Per-query time | 0.3519s | 0.0062s | — |
| Speedup | 1x | **56.5x** | — |

Paper (Lee & Park 2026, A100, 20595 images): ~15x speedup, >99% FLOPs reduction.
Your project (mps, 112 images): 56.5x speedup, 99.8% FLOPs reduction.
