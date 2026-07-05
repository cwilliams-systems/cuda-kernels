%%writefile naive_gemm.cu
#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>

__global__ void naive_gemm(const double *A, const double *B, double *C, int M, int K, int N){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(row < M && col < N){
        double sum = 0.0;
        for(int k = 0; k < K; k++){
            sum += A[row * K + k] * B[k * N + col];
        }
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

    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1)/blockDim.x,
                 (M + blockDim.y - 1)/blockDim.y);

    // CUDA event timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    naive_gemm<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
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
