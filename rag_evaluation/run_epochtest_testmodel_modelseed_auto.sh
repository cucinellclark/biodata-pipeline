#!/bin/bash -l
#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l walltime=1:00:00
#PBS -l filesystems=home:eagle
#PBS -j oe
#PBS -q debug 
#PBS -A argonne_tpc


# ------------------
JOB_ID=$PBS_JOBID # Keep track of which job this is

# -------------------
# Instructions on multi-node PyTorch on Polaris
# https://docs.alcf.anl.gov/polaris/data-science-workflows/frameworks/pytorch/
#
# Below settings are taken from there:

# Enable the use of CollNet plugin.
# CollNet allows for "in-network reductions" - apparently
# the switches themselves can do computation as the data is being shared (???)
export NCCL_COLLNET_ENABLE=1

# Use GPU Direct RDMA when GPU and NIC are on the same NUMA node. 
# Traffic will go through the CPU.
export NCCL_NET_GDR_LEVEL=PHB

# ----------------

# Enable GPU-MPI (if supported by application)
export MPICH_GPU_SUPPORT_ENABLED=1


# Information on multi node setup
export MASTER_ADDR=`head -n 1 $PBS_NODEFILE` # The first node in the list is the master node
export MASTER_PORT=29400 # Default pytorch port
NNODES=`wc -l < $PBS_NODEFILE`
NRANKS_PER_NODE=1 #$(nvidia-smi -L | wc -l)
NDEPTH=8
NTHREADS=1 # We use torch.distributed to spawn child processes

NTOTRANKS=$(( NNODES * NRANKS_PER_NODE ))
echo "NUM_OF_NODES= ${NNODES} TOTAL_NUM_RANKS= ${NTOTRANKS} RANKS_PER_NODE= ${NRANKS_PER_NODE} THREADS_PER_RANK= ${NTHREADS}"

# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Set up MMAE environment
#source "$ROOT/init.sh"
source "/home/cc8dm/RAG-eval/init.sh"
#export SOURCE_DATASET_NAME=TCGA

echo "Executing in $PBS_O_WORKDIR/.."
cd "$PBS_O_WORKDIR"
echo `pwd`
echo

source setup_test_model_params.sh 

my_command="python3 RAG-eval-test_model.py --input_dir $INPUT_DIR --terms_file $TERMS_FILE --testing_data $TESTING_DATA --model_name $MODEL_NAME --persist_dir $CHROMA_DIR --test_output $OUTPUT_FILE --adapter_dir $ADAPTER_DIR"

echo "Epoch $epoch"
echo $my_command
eval $my_command
