cublas_dgemm: cublas_dgemm.cu
	nvcc -std=c++11 cublas_dgemm.cu -o cublas_dgemm -lcublas -arch=sm_60

check_device_info: checkDeviceInfo.cu
	nvcc -std=c++11 checkDeviceInfo.cu -o check_device_info -lcublas -arch=sm_60