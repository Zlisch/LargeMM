#!/bin/bash
 
#PBS -l ncpus=48
#PBS -l mem=190GB
#PBS -l jobfs=200GB
#PBS -q normal
#PBS -P a00
#PBS -l walltime=02:00:00
#PBS -l storage=gdata/a00+scratch/a00
#PBS -l wd

module load cuda/11.6.1
module load openmpi/4.0.1
module load hdf5/1.12.1
module load gcc/10.3.0
module load cmake/3.24.2

# build
# DIR="./build"

# if [ -d "$DIR" ]
# then
#   echo "Directory ${DIR} exists"
#   cd "$DIR"
# else
#   echo "Directory ${DIR} does not exist"
#   mkdir "$DIR"
#   cd "$DIR"
#   CUDAARCHS=70 cmake ..
# fi
make

# run
LOGDIR="./logs"
if [[ ! -d "$LOGDIR" ]]
then
  mkdir "$LOGDIR"
fi
./check_device_info &>> $LOGDIR/output.txt
./cublas_dgemm &>> $LOGDIR/output.txt
nvprof --print-gpu-trace ./cublas_dgemm &>> $LOGDIR/output.txt