# SM75 Tensor Core Micro‑Kernel Engineering (RTX 2070)

![License](https://img.shields.io/badge/license-MIT-blue.svg)

Author: **João Felipe De Souza**

A systematic exploration of instruction‑level Tensor Core programming on NVIDIA SM75 (Turing), moving from WMMA API usage to PTX‑level micro‑kernel design and hardware‑aware auto‑tuning.

This project demonstrates:

- Warp‑level Tensor Core ILP engineering
- Manual `wmma.mma.sync` PTX emission
- Shared memory staging
- Block‑level tiling (64×64)
- Empirical performance modeling
- Auto‑tuning of structural kernel parameters
- Quantitative comparison against cuBLAS
- Efficiency vs theoretical hardware peak

---

## Hardware Target

- GPU: RTX 2070
- Architecture: SM75 (Turing)
- SM count: 36
- Tensor Cores per SM: 8
- Theoretical FP16 Tensor Peak: ~29 TFLOPs
- Sustained real‑world ceiling: ~18–20 TFLOPs

All experiments executed in P0 state (validated via `nvidia-smi`).

---

## Experimental Methodology

The project evolved through controlled stages:

### 1️⃣ Warp-Level Instruction Study

Measured raw `wmma.mma.sync.m16n16k16` throughput:

- Single accumulator → strong scoreboard stalls
- Dual accumulator sets (ILP=2) → ~2× warp throughput
- Branch unrolling reduced control overhead

Result:

> ~1 TFLOP sustained per warp under dense emission.

---

### 2️⃣ Block-Level Compute Scaling

Scaled to 4 warps per block:

- Independent accumulator sets
- Large grid (4096 blocks)
- Saturation of warp schedulers

Result:

> Near Tensor Core saturation in compute‑only mode.

---

### 3️⃣ Shared Memory Staging

Introduced:

- Global → shared staging
- Shared → register reuse
- Barrier-controlled tiling

Measured global memory penalty:

> ~15–20% drop vs compute‑only kernel.

---

### 4️⃣ GEMM Correctness Phase

Replaced stress‑kernel with mathematically correct GEMM:

- 64×64 threadblock tile
- 16×16 warp sub‑tiles
- Loop over K dimension
- Full update of C matrix
- Validated against cuBLAS

---

## Performance Results

### GEMM (FP16 input → FP32 accumulate)

| Size | PTX Hybrid | cuBLAS |
|------|------------|--------|
| 2048³ | ~8–14 TFLOPs | ~4–5 TFLOPs |
| 4096³ | ~7–12 TFLOPs | ~15–19 TFLOPs |
| 8192³ | ~11–13 TFLOPs | ~19–20 TFLOPs |

### Efficiency vs Theoretical Peak (29 TFLOPs)

- Best sustained: ~40–45%
- Relative to cuBLAS: ~60–65%

All numbers measured with: GFLOPs = 2 * M * N * K / time

---

## Roofline Perspective

Theoretical roofline:

- Compute bound regime: ~29 TFLOPs (FP16 Tensor Core peak)
- Observed sustainable: ~18–20 TFLOPs
- Achieved PTX Hybrid: ~12–13 TFLOPs sustained

The kernel operates in a compute‑bound regime for large sizes (8192³), confirmed by stable throughput and minimal sensitivity to memory bandwidth after shared staging.

---

## Occupancy Model

SM75 constraints:

- 65536 registers per SM
- 64 warps per SM
- ~68 registers per thread (hybrid kernel)

Estimated active warps per SM:

~28–30 warps per SM

Observation:

> Full 100% occupancy was not required to approach peak Tensor Core throughput.  
> Instruction density and accumulator dependency dominated performance behavior.

---

## Auto‑Tuning

Structural parameters were auto‑tuned:

- WARPS_PER_BLOCK ∈ {1, 2, 4}
- STAGE_K ∈ {16, 32}
- ILP_DEPTH ∈ {1, 2}

Results show:

- STAGE_K = 32 is optimal regime
- WARPS = 4 best for SM75 occupancy balance
- ILP_DEPTH = 2 significantly improves throughput
- Excessive stage depth (64) degrades performance (register pressure)

Heatmap and plots generated via:./scripts/tune.sh
./scripts/plot.sh

---

## Key Findings

1. Accumulator dependency is the dominant warp‑level bottleneck.
2. ILP via dual accumulator sets materially increases Tensor Core issue density.
3. Branch density directly impacts instruction throughput.
4. Stage depth has a clear optimum; deeper is not always better.
5. Occupancy alone does not guarantee saturation.
6. WMMA API incurs structural overhead relative to PTX emission.
7. PTX‑level emission allows finer instruction scheduling control.
8. Memory staging cost becomes dominant once Tensor Core is saturated.

---

## Lessons Learned

- Micro‑kernel engineering requires explicit control of dependency chains.
- FLOP accounting must strictly match actual mathematical work.
- Compute‑only microbenchmarks can overestimate real GEMM throughput.
- Synchronization frequency critically affects effective throughput.
- Hardware‑aware auto‑tuning reveals non‑intuitive optimal regimes.

---

## Future Work

- Double‑buffered 2‑stage pipeline
- Larger tile exploration (128×128)
- INT8 Tensor Core variant
- Port to SM80 (Ampere) with `cp.async`
- Runtime auto‑selection backend
- Integration into Transformer runtime

---

## Project Structure
kernels/
gemm_wmma_fp16.cu # Hybrid CUDA + PTX compute core
benchmarks/
benchmark.cu # GEMM benchmark + cuBLAS comparison
scripts/
tune.sh # Structural auto‑tuner
plot.sh # Visualization

---

## Reproducibility

Build:cmake .. -G Ninja -DCMAKE_CUDA_HOST_COMPILER=gcc-13
ninja
./benchmark
Run tuner:./scripts/tune.sh
./scripts/plot.sh


---

## License

MIT License

---

## Author

2026 João Felipe De Souza  
GPU Systems & AI Infrastructure Engineering