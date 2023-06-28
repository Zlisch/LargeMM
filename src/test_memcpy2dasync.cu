#include <cublas_v2.h>
#include <cstdio>
#include <cuda_runtime.h>
#include <stdio.h>
#include "./common.h"

/*
 * Check whether cudaMemcpy2DAsync uses row major or colmun major.
 */

#define NSTREAM 1

void initialData(float *ip, int size)
{
    int i;

    for(i = 0; i < size; i++)
    {
        ip[i] = (float)(rand() & 0xFF) / 10.0f;
    }
}

void checkResult(float *hostRef, float *gpuRef, int N)
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
    int m = 3;
    int n = 2;
    int dpitch = 2; // row major
    int spitch = 3; // row major

    // The test matrix A is:
    // |  7  | 8  | 9  | 
    // |  10 | 11 | 12 | 
    // |  13 | 14 | 15 | 
    // |  16 | 17 | 18 | 
    // test store in row major
    float h_A[] = {7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18};

    // The sub-matrix we expect:
    // |  7 | 8  |
    // | 10 | 11 |
    // | 13 | 14 |
    // test store in row major
    float expected_gpuRef[] = {7, 8, 10, 13, 14};

    // malloc host memory
    float *gpuRef = (float *)malloc(m * n * sizeof(float));
    // initialize data at host side
    memset(gpuRef, 0, m * n * sizeof(float));

    // malloc device global memory
    float *d_MatA;
    cudaMalloc((void **)&d_MatA, m * n * sizeof(float));

    // transfer data from host to device
    CHECK(cudaMemcpy2DAsync(d_MatA, dpitch, h_A, spitch, n, m, cudaMemcpyHostToDevice, 0));

    // copy memcpy result back to host side
    cudaMemcpy(gpuRef, d_MatA, m * n * sizeof(float), cudaMemcpyDeviceToHost);

    // check device results
    checkResult(expected_gpuRef, gpuRef, m * n);

    // free device global memory 
    cudaFree(d_MatA); 

    // free host memory 
    free(gpuRef);
}