# cuda-kernels

CUDA kernel projects for Summer 2026

## Kernels

- `vector_add.cu` — vector addition, one thread per element, CUDA event timing
- `naive_gemm.cu` — naive GPU matrix multiply, 23.956ms on 1024×1024 T4, no shared memory optimization
- `tiled_gemm.cu` — tiled shared-memory matrix multiply, TILE_WIDTH=16, 18.897ms on 1024×1024 T4 (1.27x faster than naive)
