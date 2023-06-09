binaries=./bin/cublas_dgemm ./bin/check_device_info ./bin/test_natural ./bin/test_transpose

./bin/cublas_dgemm: ./src/cublas_dgemm.cu
	nvcc -std=c++11 ./src/cublas_dgemm.cu -o ./bin/cublas_dgemm -lcublas -arch=sm_60

./bin/check_device_info: ./src/checkDeviceInfo.cu
	nvcc -std=c++11 ./src/checkDeviceInfo.cu -o ./bin/check_device_info -lcublas -arch=sm_60

./bin/cublas_dgemm: ./src/test_natural.cu
	nvcc -std=c++11 ./src/test_natural.cu -o ./bin/test_natural -lcublas -arch=sm_60

./bin/cublas_dgemm: ./src/test_transpose.cu
	nvcc -std=c++11 ./src/test_transpose.cu -o ./bin/test_transpose -lcublas -arch=sm_60

.PHONY: clean
clean:
	rm -f $(binaries) *.o