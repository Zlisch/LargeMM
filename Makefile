binaries=cublas_dgemm check_device_info test_natural test_transpose

cublas_dgemm: ./src/cublas_dgemm.cu
	nvcc -std=c++11 ./src/cublas_dgemm.cu -o cublas_dgemm -lcublas -arch=sm_60

check_device_info: ./src/checkDeviceInfo.cu
	nvcc -std=c++11 ./src/checkDeviceInfo.cu -o check_device_info -lcublas -arch=sm_60

cublas_dgemm: ./src/test_natural.cu
	nvcc -std=c++11 ./src/test_natural.cu -o test_natural -lcublas -arch=sm_60

cublas_dgemm: ./src/test_transpose.cu
	nvcc -std=c++11 ./src/test_transpose.cu -o test_transpose -lcublas -arch=sm_60

.PHONY: clean
clean:
	rm -f $(binaries) *.o