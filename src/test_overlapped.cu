#include <cublas_v2.h>
#include <cstdio>
#include <cuda_runtime.h>
#include <stdio.h>
#include "./common.h"

#define NSTREAM 4

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

void printMatrix(float *C, int N)
{
    printf("\n");
    for (int i = 0; i < N; i++)
    {
        printf("(%f)\b\b", C[i]);
    }
    printf("\n");
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
    int m = 4;
    int k = 4;
    int n = 4;
    printf("MM Size: m = %d, k = %d, n = %d\n", m, k, n);

    // calculalte data size of one square matix
    int nElem = m * n;
    printf("> square matrix size = %d\n", nElem);
    size_t nBytes = nElem * sizeof(float);

    // The test matrix A is:
    // |  1  | 2  | 3  | 4  |
    // |  5  | 6  | 7  | 8  |
    // |  9  | 10 | 11 | 12 | 
    // |  13 | 14 | 15 | 16 | 
    // test store in row major
    float h_A[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};

    // The test matrix ABis:
    // |  4  | 2  | 3  | 4  |
    // |  5  | 1  | 7  | 1  |
    // |  9  | 10 | 17 | 12 | 
    // |  13 | 14 | 15 | 10 | 
    // test store in row major
    float h_B[] = {4, 2, 3, 4, 5, 1, 7, 1, 9, 10, 17, 12, 13, 14, 15, 10};

    // The result we expect:
    // |  93  | 90  | 128 | 82  |
    // |  217 | 198 | 296 | 190 |
    // |  341 | 306 | 464 | 298 | 
    // |  465 | 414 | 632 | 406 | 
    // should be stored in col major
    float expected_gpuRef[] = {93, 217, 341, 465, 90, 198, 306, 414, 128, 296, 464, 632, 82, 190, 298, 406};

    // malloc host memory
    float *gpuRef = (float *)malloc(m * n * sizeof(float));
    // initialize data at host side
    memset(gpuRef, 0, m * n * sizeof(float));

    // initialize CUBLAS context
    cublasStatus_t stat;   // cuBLAS functions status
    cublasHandle_t handle; // cuBLAS context
    stat = cublasCreate(&handle); 

    float alpha = 1.0f;
    float beta = 0.0f;

    // malloc device global memory
    float *d_MatA, *d_MatB, *d_MatC;
    cudaMalloc((void **)&d_MatA, m * k * sizeof(float)); 
    cudaMalloc((void **)&d_MatB, k * n * sizeof(float)); 
    cudaMalloc((void **)&d_MatC, m * n * sizeof(float)); 

    // for synchronize the streams and the default stream
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    // initialise streams
    cudaStream_t stream[NSTREAM];
    for (int i = 0; i < NSTREAM; ++i)
    {
        CHECK(cudaStreamCreate(&stream[i]));
    }

    CHECK(cudaEventRecord(start, 0));

    // initiate all work on the device asynchronously in depth-first order
    for (int i = 0; i < NSTREAM; ++i)
    {
        if (i == 0 || i == 2) 
            cudaMemcpy2DAsync(d_MatA + (i/2) * (nElem/2), k * sizeof(float), h_A + (i/2) * (nElem/2), k * sizeof(float), k * sizeof(float), m/2, cudaMemcpyHostToDevice, stream[i]);
        if (i == 0 || i == 1)
            cudaMemcpy2DAsync(d_MatB + (i%2) * (n/2), n * sizeof(float), h_B + (i%2) * (n/2), n * sizeof(float), (n/2) * sizeof(float), k, cudaMemcpyHostToDevice, stream[i]);
        stat = cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, m/2, n/2, k, &alpha, d_MatA + (i/2) * (nElem/2), k, d_MatB + (i%2) * (n/2), n, &beta, d_MatC + (i%2) * (nElem/2) + (i/2) * (n/2), n);
    }

    CHECK(cudaEventRecord(stop, 0));
    CHECK(cudaEventSynchronize(stop));

    // transfer data from host to device
    // general case for A
    // CHECK(cudaMemcpy2DAsync(d_MatA + (3/2) * (nElem/2), k * sizeof(float), h_A + (3/2) * (nElem/2), k * sizeof(float), k * sizeof(float), m/2, cudaMemcpyHostToDevice, 0));
    // general case for B
    // CHECK(cudaMemcpy2DAsync(d_MatB + (3%2) * (n/2), n * sizeof(float), h_B + (3%2) * (n/2), n * sizeof(float), (n/2) * sizeof(float), k, cudaMemcpyHostToDevice, 0););

    // copy memcpy result back to host side
    cudaMemcpy(gpuRef, d_MatC, m * n * sizeof(float), cudaMemcpyDeviceToHost);

    // check device results
    // checkResult(expected_gpuRef, gpuRef, m * n);

    // print returned matrix
    printf("The gpu matrix:\n");
    printMatrix(gpuRef, 4*4);

    // free device global memory 
    cudaFree(d_MatA); 
    cudaFree(d_MatB); 
    cudaFree(d_MatC);

    // destroy CUBLAS context
    cublasDestroy(handle);

    // free host memory 
    free(gpuRef);

    return EXIT_SUCCESS;
}