#include <cublas_v2.h>
#include <cstdio>
#include <cuda_runtime.h>
#include <stdio.h>

#define CHECK(call)                                                         \
{                                                                           \
    const cudaError_t error = call;                                         \
    if (error != cudaSuccess)                                               \
    {                                                                       \
        printf("Error: %s:%d, ", __FILE__, __LINE__);                       \
        printf("code:%d, reason: %s\n", error, cudaGetErrorString(error));  \
        exit(-10*error);                                                    \
    }                                                                       \
}                                                                       

void initialInt(float *ip, int size) { 
    for (int i=0; i<size; i++) {
        ip[i] = (float)i; 
    }
}

int main(int argc, char *argv[]) {
    printf("%s Starting...\n", argv[0]);

    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp; 
    CHECK(cudaGetDeviceProperties(&deviceProp, dev)); 
    printf("Using Device %d: %s\n", dev, deviceProp.name); 
    CHECK(cudaSetDevice(dev));

    // set up date size of matrix 
    int m = 1<<14;
    int k = 1<<14;
    int n = 1<<14;
    printf("MM Size: m = %d, k = %d, n = %d\n", m, k, n);

    cublasStatus_t stat;   // cuBLAS functions status
    cublasHandle_t handle; // cuBLAS context

    // malloc host memory
    float *h_A, *h_B, *gpuRef;
    h_A = (float *)malloc(m * k * sizeof(float));
    h_B = (float *)malloc(k * n * sizeof(float)); 
    // hostRef = (float *)malloc(m * n * sizeof(float)); 
    gpuRef = (float *)malloc(m * n * sizeof(float));

    // initialize data at host side
    initialInt (h_A, m * k);
    initialInt (h_B, k * n);

    // memset(hostRef, 0, m * n * sizeof(float));
    memset(gpuRef, 0, m * n * sizeof(float));

    // malloc device global memory
    float *d_MatA, *d_MatB, *d_MatC;
    cudaMalloc((void **)&d_MatA, m * k * sizeof(float)); 
    cudaMalloc((void **)&d_MatB, k * n * sizeof(float)); 
    cudaMalloc((void **)&d_MatC, m * n * sizeof(float));

    // transfer data from host to device
    cudaMemcpy(d_MatA, h_A, m * k * sizeof(float), cudaMemcpyHostToDevice); 
    cudaMemcpy(d_MatB, h_B, k * n * sizeof(float), cudaMemcpyHostToDevice);

    // initialize CUBLAS context
    stat = cublasCreate(&handle); 

    float alpha = 1.0f;
    float beta = 0.5f;

    // invoke CUBLAS
    stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, d_MatB, n,
                        d_MatA, k, &beta, d_MatC, n);

    // cublasSetStream(handle, stream);

    // copy kernel result back to host side
    cudaMemcpy(gpuRef, d_MatC, m * n * sizeof(float), cudaMemcpyDeviceToHost);

    // free device global memory 
    cudaFree(d_MatA); 
    cudaFree(d_MatB); 
    cudaFree(d_MatC);

    // destroy CUBLAS context
    cublasDestroy(handle); 

    // free host memory 
    free(h_A); 
    free(h_B); 
    free(gpuRef);

    return EXIT_SUCCESS;
}