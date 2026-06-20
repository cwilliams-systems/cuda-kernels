#include <stdio.h>
#include <cuda_runtime.h>

__global__ void addVectors(int* a, int* b, int* c, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n){
        c[idx] = a[idx] + b[idx];
    }

}
int main(){
    int* h_a = new int [1000];
    int* h_b = new int [1000];
    int* h_c = new int [1000];

  for (int i = 0; i < 1000; i++){
    h_a[i] = i;
    h_b[i] = i * 2;
  }

  int* d_a;
  int* d_b;
  int* d_c;

  cudaMalloc(&d_a, sizeof(int) * 1000);
  cudaMalloc(&d_b, sizeof(int) * 1000);
  cudaMalloc(&d_c, sizeof(int) * 1000);

  cudaMemcpy(d_a, h_a, sizeof(int) * 1000, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, sizeof(int) * 1000, cudaMemcpyHostToDevice);

  int ThreadsperBlock = 256;
  int blocksperGrid = (1000 + ThreadsperBlock - 1)/ThreadsperBlock;

 addVectors<<<blocksperGrid, ThreadsperBlock>>>(d_a, d_b, d_c, 1000);

 cudaDeviceSynchronize();

  cudaMemcpy(h_c, d_c, sizeof(int) * 1000, cudaMemcpyDeviceToHost);

  printf("h_c[5] = %d\n",h_c[5]);

  cudaFree (d_a);
  cudaFree (d_b);
  cudaFree (d_c);

  delete[] h_a;
  delete[] h_b;
  delete[] h_c;

  return 0;



}
