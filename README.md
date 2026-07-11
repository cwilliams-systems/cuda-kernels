# cuda-kernels

CUDA kernel projects for Summer 2026

## Kernels

- `vector_add.cu` — vector addition, one thread per element, CUDA event timing
- `naive_gemm.cu` — naive GPU matrix multiply, 23.956ms on 1024×1024 T4, no shared memory optimization
- `tiled_gemm.cu` — tiled shared-memory matrix multiply, TILE_WIDTH=16, 18.897ms on 1024×1024 T4 (1.67x faster than naive, verified against naive output with matched seed).



# Nsight Profile Section Titled GEMM
Compute (SM) Throughput:  97.07%   ← This is showing a compute bound problem
Memory Throughput:        21.64%
Achieved Occupancy:       99.08%
Memory Throughput (GB/s): 34.93
FP64 pipeline:            97.7%    ← This is showing the primary bottleneck
Registers per thread:     42
Kernel duration:          23.64ms
```

Results: The Kernel is compute bound on the FP64 pipeline at 97.7% utilization.
The T4 has a 32:1 FP32/FP64 performance ratio switching from double to float would be the primary choice.

**Occupancy:** Near-perfect at 99.08% showing that's not a bottleneck.
