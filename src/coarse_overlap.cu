#include <cublas_v2.h>
#include <cstdio>
#include <cuda_runtime.h>
#include <stdio.h>
#include "./common.h"

/*
 * Check potential performance improvement using 4 streams. Assume the second matrix given is
 * stored in a column major way. Without copying back the result matrix from device to host.
 */

#define NSTREAM 4

void initialData(float *ip, int size)
{
    int i;

    for(i = 0; i < size; i++)
    {
        ip[i] = (float)(rand() & 0xFF) / 10.0f;
    }
}

int main(int argc, char *argv[]) 
{
    printf("> %s Starting...\n", argv[0]);

    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp; 
    CHECK(cudaGetDeviceProperties(&deviceProp, dev)); 
    printf("Using Device %d: %s\n", dev, deviceProp.name); 
    CHECK(cudaSetDevice(dev));

    // set up max connectioin
    char * iname = "CUDA_DEVICE_MAX_CONNECTIONS";
    setenv (iname, "8", 1);
    char *ivalue =  getenv (iname);
    printf ("> %s = %s\n", iname, ivalue);
    printf ("> with streams = %d\n", NSTREAM);

    // set up data size of matrix 
    int m = 1<<14;
    int k = 1<<14;
    int n = 1<<14;
    printf("MM Size: m = %d, k = %d, n = %d\n", m, k, n);

    // calculalte data size of one square matix
    int nElem = m * n;
    printf("> square matrix size = %d\n", nElem);
    size_t nBytes = nElem * sizeof(float);

    // malloc pinned host memory for async memcpy
    float *h_A, *h_B;
    cudaHostAlloc((void**)&h_A, nBytes, cudaHostAllocDefault);
    cudaHostAlloc((void**)&h_B, nBytes, cudaHostAllocDefault);

    // initialize data at host side
    initialData(h_A, m * k);
    initialData(h_B, k * n);

    // malloc device global memory
    float *d_MatA, *d_MatB, *d_MatC;
    cudaMalloc((void **)&d_MatA, m * k * sizeof(float)); 
    cudaMalloc((void **)&d_MatB, k * n * sizeof(float)); 
    cudaMalloc((void **)&d_MatC, m * n * sizeof(float)); 

    // initialize CUBLAS context
    cublasHandle_t handles[NSTREAM]; // cuBLAS handles for each stream

    float alpha = 1.0f;
    float beta = 0.0f;

    // initialise streams
    cudaStream_t stream[NSTREAM];
    for (int i = 0; i < NSTREAM; ++i)
    {
        CHECK(cudaStreamCreate(&stream[i]));
        cublasCreate(&(handles[i]));
    }

    // initiate all work on the device asynchronously in depth-first order
    for (int i = 0; i < NSTREAM; ++i)
    {
        cublasSetStream(handles[i], stream[i]);
        cudaMemcpy2DAsync(d_MatA + (i/2) * (nElem/2), k * sizeof(float), h_A + (i/2) * (nElem/2), k * sizeof(float), k * sizeof(float), m/2, cudaMemcpyHostToDevice, stream[i]);
        cudaMemcpy2DAsync(d_MatB + (i%2) * (n/2), n * sizeof(float), h_B + (i%2) * (n/2), n * sizeof(float), (n/2) * sizeof(float), k, cudaMemcpyHostToDevice, stream[i]);
        cublasSgemm(handles[i], CUBLAS_OP_T, CUBLAS_OP_T, m/2, n/2, k, &alpha, d_MatA + (i/2) * (nElem/2), k, d_MatB + (i%2) * (n/2), n, &beta, d_MatC + (i%2) * (nElem/2) + (i/2) * (n/2), n);
    }

    // free device global memory 
    cudaFree(d_MatA); 
    cudaFree(d_MatB); 
    cudaFree(d_MatC);

    // destroy CUBLAS context
    for (int i = 0; i < NSTREAM; ++i) cublasDestroy(handles[i]); 

    // free host memory 
    cudaFreeHost(h_A); 
    cudaFreeHost(h_B); 

    return EXIT_SUCCESS;
}