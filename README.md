# LargeMM

## About The Project

This project aims for building a C/C++ application based on the cuBLAS library to improve the performance of large matricies multiplication. The application would take the existing single GPU algorithm for large matricies multiplication and parallelize it over multiple GPUs. This will all be done within the Extreme Scale Electronic Structure Software (EXESS) and will enable the program to scale on larger node counts. By the end of the project we will have successfully designed a multi-GPU algorithm for Q-Next, will have implemented the algorithm and optimized it for close to peak performance operation and finally benchmarks against the state-of-the-art codes.


### Built With

* [CUDA](https://developer.nvidia.com/cuda-toolkit)
* [cuBLAS](https://docs.nvidia.com/cuda/cublas/)

## Environment

The default environment for this application are [Gadi](https://nci.org.au/our-systems/hpc-systems) and `cuda` version >= `11.6.1`. To run the application, use the `run.sh` provided or build your own script.