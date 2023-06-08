#!/bin/bash

module load cuda/11.6.1
module load openmpi/4.0.1
module load hdf5/1.12.1
module load gcc/10.3.0
module load cmake/3.24.2

make

# run
LOGDIR="./logs"
if [[ ! -d "$LOGDIR" ]]
then
  mkdir "$LOGDIR"
fi
# ./check_device_info &>> $LOGDIR/output.txt
# ./cublas_dgemm &>> $LOGDIR/output.txt
./test_transpose &>> $LOGDIR/output.txt

# nvprof --print-gpu-trace ./cublas_dgemm &>> $LOGDIR/output.txt

# nsys profile --stats=true cublas_dgemm