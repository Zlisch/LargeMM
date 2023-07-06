#!/bin/bash

module load cuda/11.6.1
module load openmpi/4.0.1
module load hdf5/1.12.1
module load gcc/10.3.0
module load cmake/3.24.2

# run
LOGDIR="./logs"
if [[ ! -d "$LOGDIR" ]]
then
  mkdir "$LOGDIR"
fi

BINDIR="./bin"
if [[ ! -d "$BINDIR" ]]
then
  mkdir "$BINDIR"
fi

make clean
make all

# ./check_device_info &>> $LOGDIR/output.txt
# ./cublas_dgemm &>> $LOGDIR/output.txt
# ./bin/test_overlapped &>> $LOGDIR/output.txt

# nvprof --print-gpu-trace ./bin/test_natural &>> $LOGDIR/output.txt
# nvprof --print-gpu-trace ./bin/test_transpose &>> $LOGDIR/output.txt
# nvprof --print-gpu-trace ./bin/coarse_overlap &>> $LOGDIR/output.txt
# nvprof --print-gpu-trace ./bin/test_overlapped &>> $LOGDIR/output.txt

# cuda-memcheck ./bin/coarse_overlap &>> $LOGDIR/output.txt

# nsys profile --stats=true cublas_dgemm
nsys profile --stats=true ./bin/coarse_overlap