#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>

#define TILE_WIDTH  16

__global__ void tiled_gemm(const double *A, const double *B, double *C, int M, int K, int N){

  __shared__ double sharedA[TILE_WIDTH][TILE_WIDTH];
  __shared__ double sharedB[TILE_WIDTH][TILE_WIDTH];

  int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
  int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

  double sum = 0.0;

  int numTiles = (K + TILE_WIDTH - 1)/TILE_WIDTH;

  for(int t = 0; t < numTiles; t++){

    int tileRow = t * TILE_WIDTH + threadIdx.y;
    int tileCol = t * TILE_WIDTH + threadIdx.x;

    if(row < M && tileCol < K){
      sharedA[threadIdx.y][threadIdx.x] = A[row * K + tileCol];
    }else{
      sharedA[threadIdx.y][threadIdx.x] = 0.0;
    }

    if(tileRow < K && col < N){
      sharedB[threadIdx.y][threadIdx.x] = B[tileRow * N + col];
    }else{
      sharedB[threadIdx.y][threadIdx.x] = 0.0;
    }
    __syncthreads();

    for(int i = 0; i < TILE_WIDTH; i++){
      sum += sharedA[threadIdx.y][i] * sharedB[i][threadIdx.x];
    }
    __syncthreads();
  }

  if(row < M && col < N){
    C[row * N + col] = sum;
  }
}

int main(){
    const int M = 1024;
    const int K = 1024;
    const int N = 1024;

    double *A = (double*)malloc(M * K * sizeof(double));
    double *B = (double*)malloc(K * N * sizeof(double));
    double *C = (double*)malloc(M * N * sizeof(double));

    for(int i = 0; i < M * K; i++){
      A[i] = (double)(rand())/RAND_MAX;
    }

    for(int i = 0; i < K * N; i++){
      B[i] = (double)(rand())/RAND_MAX;
    }

    double *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, M * K * sizeof(double));
    cudaMalloc(&d_B, K * N * sizeof(double));
    cudaMalloc(&d_C, M * N * sizeof(double));

    cudaMemcpy(d_A, A, M * K * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, K * N * sizeof(double), cudaMemcpyHostToDevice);

    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    dim3 gridDim((N + TILE_WIDTH - 1)/TILE_WIDTH,
                 (M + TILE_WIDTH - 1)/TILE_WIDTH);

    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    tiled_gemm<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    cudaMemcpy(C, d_C, M * N * sizeof(double), cudaMemcpyDeviceToHost);

    printf("C[0] = %f\n", C[0]);
    printf("Kernel time: %.3f ms\n", ms);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(A);
    free(B);
    free(C);

    return 0;
}
