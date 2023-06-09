#include <cublas_v2.h>
#include <cstdio>
#include <cuda_runtime.h>
#include <stdio.h>
#include "./common.h"

/*
 * Check potential performance improvement using 4 streams but
 * does not check for correctness. Assume the second matrix given is
 * stored in a column major way.
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

    // grid parallel operation
    int iElem = nElem / NSTREAM;
    size_t iBytes = iElem * sizeof(float);

    // malloc pinned host memory for async memcpy
    float *h_A, *h_B, *gpuRef[NSTREAM];
    // float *h_A, *h_B, *gpuRef[NSTREAM], *h_B_copy;
    cudaHostAlloc((void**)&h_A, nBytes, cudaHostAllocDefault);
    cudaHostAlloc((void**)&h_B, nBytes, cudaHostAllocDefault);
    // cudaMalloc((void **)&h_B_copy, nBytes); 
    for (int i = 0; i < NSTREAM; i++)
    cudaHostAlloc((void**)&gpuRef[i], iBytes, cudaHostAllocDefault);

    // initialize data at host side
    initialData (h_A, m * k);
    initialData (h_B, k * n);
    // cudaMemcpy(&h_B_copy, &h_B, nBytes, cudaMemcpyHostToHost);
    for (int i = 0; i < NSTREAM; i++)
    memset(gpuRef[i], 0, iBytes);

    // initialize CUBLAS context
    cublasStatus_t stat;   // cuBLAS functions status
    cublasHandle_t handle; // cuBLAS context
    stat = cublasCreate(&handle); 

    float alpha = 1.0f;
    float beta = 0.0f;

    // tramsform the second matrix to be column major
    // cublasSgeam( handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, &alpha, h_B_copy, n, &beta, h_B_copy, m, h_B, m );

    // malloc device global memory
    float *d_MatA, *d_MatB, *d_MatC[NSTREAM];
    cudaMalloc((void **)&d_MatA, m * k * sizeof(float)); 
    cudaMalloc((void **)&d_MatB, k * n * sizeof(float)); 
    for (int i = 0; i < NSTREAM; i++)
    cudaMalloc((void **)&d_MatC[i], iBytes);

    // initialise streams
    cudaStream_t stream[NSTREAM];
    for (int i = 0; i < NSTREAM; ++i)
    {
        CHECK(cudaStreamCreate(&stream[i]));
    }

    int sBytes = iBytes * 2;
    // initiate all work on the device asynchronously in depth-first order
    for (int i = 0; i < NSTREAM; ++i)
    {
        int iaoffset = (i % 2) * iElem * 2;
        int iboffset = !(i % 2) * iElem * 2;
        if (i < 2)
        {
            cudaMemcpyAsync(&d_MatA[iaoffset], &h_A[iaoffset], sBytes,
                              cudaMemcpyHostToDevice, stream[i]);
            cudaMemcpyAsync(&d_MatB[iboffset], &h_B[iboffset], sBytes,
                                cudaMemcpyHostToDevice, stream[i]);
        }
        // cublas TODO
        cudaMemcpyAsync(&gpuRef[i], &d_MatC[i], iBytes,
                              cudaMemcpyDeviceToHost, stream[i]);
    }

    // free device global memory 
    cudaFree(d_MatA); 
    cudaFree(d_MatB); 
    for (int i = 0; i < NSTREAM; i++)
    cudaFree(d_MatC[i]);

    // destroy CUBLAS context
    cublasDestroy(handle); 

    // free host memory 
    free(h_A); 
    free(h_B); 
    // free(h_B_copy);
    for (int i = 0; i < NSTREAM; i++)
    free(gpuRef[i]);

    return EXIT_SUCCESS;
}