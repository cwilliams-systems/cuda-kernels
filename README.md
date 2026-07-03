#   Cuda-kernels

CUDA kernel implementations written while learning GPU programming — Summer 2026.

## Kernels

- `vector_add.cu` — vector addition, one thread per element, CUDA event timing
- `native_gemm.cu` — naive GPU matrix multiply (no shared memory), correct on 1024×1024
- `tiled_gemm.cu` — tiled shared-memory matrix multiply with bank conflict avoidance (coming July 3)
