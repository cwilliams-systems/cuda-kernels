# cuda-kernels

CUDA kernel implementations written for learning GPU programming during Summer 2026.

## Kernels

- `vector_add.cu` — vector addition, one thread per element, CUDA event timing
- `native_gemm.cu` — naive GPU matrix multiply, correct on 1024×1024, no shared memory optimization
- `tiled_gemm.cu` — tiled shared-memory matrix multiply, TILE_WIDTH=16, 18.897ms on 1024×1024 T4
