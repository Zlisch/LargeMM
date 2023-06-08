#include <cublas_v2.h>
#include <cstdio>
#include <cuda_runtime.h>
#include <stdio.h>
#include "./common.h"

#define NSTREAM 4

/*
 * CU_BLAS_T test passed.
 */

void initialData(float *ip, int size)
{
    int i;

    for(i = 0; i < size; i++)
    {
        ip[i] = (float)(rand() & 0xFF) / 10.0f;
    }
}

void checkResult(float *hostRef, float *gpuRef,   int N)
{
    double epsilon = 1.0E-8;
    bool match = 1;

    for (int i = 0; i < N; i++)
    {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon)
        {
            match = 0;
            printf("Arrays do not match!\n");
            printf("host %5.2f gpu %5.2f at %d\n", hostRef[i], gpuRef[i], i);
            break;
        }
    }

    if (match) printf("Arrays match.\n\n");
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

    // set up testing
    int m = 4;
    int n = 2;
    int k = 3;

    // Matrix A (LHS) is:
    // |  7  | 8  | 9  | 
    // |  10 | 11 | 12 | 
    // |  13 | 14 | 15 | 
    // |  16 | 17 | 18 | 
    // test store in row major
    float h_A[] = {7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18};

    // Matrix B (RHS) is:
    // |  1 |  2 |
    // |  3 |  4 |
    // |  5 |  6 |
    // test store in row major
    float h_B[] = {1, 2, 3, 4, 5, 6};

    // Here are the results we expect, from hand calculations:
    // (1 * 7) + (3 * 8) + (5 * 9) = 76
    // (2 * 7) + (4 * 8) + (6 * 9) = 100
    // (1 * 10) + (3 * 11) + (5 * 12) = 103
    // (2 * 10) + (4 * 11) + (6 * 12) = 136
    // (1 * 13) + (3 * 14) + (5 * 15) = 130
    // (2 * 13) + (4 * 14) + (6 * 15) = 172
    // (1 * 16) + (3 * 17) + (5 * 18) = 157
    // (2 * 16) + (4 * 17) + (6 * 18) = 208
    // That means matrix C should be:
    // |  76 | 100 |
    // | 103 | 136 |
    // | 130 | 172 |
    // | 157 | 208 |
    // test store in col major
    float expected_gpuRef[] = {76, 103, 130, 157, 100, 136, 172, 208};

    // malloc host memory
    float *gpuRef = (float *)malloc(m * n * sizeof(float));
    // initialize data at host side
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
    cublasStatus_t stat;   // cuBLAS functions status
    cublasHandle_t handle; // cuBLAS contextv
    stat = cublasCreate(&handle); 

    float alpha = 1.0f;
    float beta = 0.0f;

    // invoke CUBLAS
    // test row maj
    stat = cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, m, n, k, &alpha, d_MatA, k,
                        d_MatB, n, &beta, d_MatC, m);

    // copy kernel result back to host side
    cudaMemcpy(gpuRef, d_MatC, m * n * sizeof(float), cudaMemcpyDeviceToHost);

    // check device results
    checkResult(expected_gpuRef, gpuRef, m * n);

    // free device global memory 
    cudaFree(d_MatA); 
    cudaFree(d_MatB); 
    cudaFree(d_MatC);

    // destroy CUBLAS context
    cublasDestroy(handle); 

    // free host memory  
    free(gpuRef);
}