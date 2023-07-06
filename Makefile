binaries=./bin/cublas_dgemm ./bin/check_device_info ./bin/test_natural ./bin/test_transpose ./bin/test_memcpy2dasync ./bin/coarse_overlap ./bin/test_overlapped
all: $(binaries)

./bin/cublas_dgemm: ./src/cublas_dgemm.cu
	nvcc -std=c++11 ./src/cublas_dgemm.cu -o ./bin/cublas_dgemm -lcublas -arch=sm_60

./bin/check_device_info: ./src/checkDeviceInfo.cu
	nvcc -std=c++11 ./src/checkDeviceInfo.cu -o ./bin/check_device_info -lcublas -arch=sm_60

./bin/test_natural: ./src/test_natural.cu
	nvcc -std=c++11 ./src/test_natural.cu -o ./bin/test_natural -lcublas -arch=sm_60

./bin/test_transpose: ./src/test_transpose.cu
	nvcc -std=c++11 ./src/test_transpose.cu -o ./bin/test_transpose -lcublas -arch=sm_60

./bin/test_memcpy2dasync: ./src/test_memcpy2dasync.cu
	nvcc -std=c++11 ./src/test_memcpy2dasync.cu -o ./bin/test_memcpy2dasync -lcublas -arch=sm_60

./bin/coarse_overlap: ./src/coarse_overlap.cu
	nvcc -std=c++11 ./src/coarse_overlap.cu -o ./bin/coarse_overlap -lcublas -arch=sm_60

./bin/test_overlapped: ./src/coarse_overlap.cu
	nvcc -std=c++11 ./src/test_overlapped.cu -o ./bin/test_overlapped -lcublas -arch=sm_60

.PHONY: clean
clean:
	rm -f $(binaries) *.o