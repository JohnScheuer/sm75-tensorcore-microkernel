# Micro‑Kernel Generator for SM75 (RTX 2070)

This project explores instruction‑level Tensor Core micro‑kernel engineering on SM75.

## Highlights

- Warp-level PTX Tensor Core micro‑kernel
- ILP via dual accumulator sets
- Shared memory staging
- Manual `wmma.mma.sync` emission
- Block‑level tiling (64×64)
- Performance comparison vs cuBLAS
- Empirical efficiency vs theoretical peak

## Results (RTX 2070, SM75)

| Size | PTX Hybrid | cuBLAS |
|------|------------|--------|
| 2048 | ~8–10 TFLOPs | ~4–5 TFLOPs |
| 4096 | ~7–12 TFLOPs | ~15–19 TFLOPs |
| 8192 | ~12–13 TFLOPs | ~19–20 TFLOPs |

Peak theoretical FP16 Tensor Core: ~29 TFLOPs  
Sustained manual PTX: ~40–45% peak

## Methodology

- Warp-level emission microbench
- ILP exploration
- Branch density tuning
- Shared memory staging
- Block-level scaling
- Empirical roofline validation