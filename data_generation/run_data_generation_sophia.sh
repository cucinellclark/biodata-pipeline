#!/bin/bash -l
#PBS -l select=1:system=sophia
#PBS -l walltime=12:00:00
#PBS -l filesystems=home:eagle
#PBS -j oe
#PBS -q single-node 
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

cd /lus/eagle/projects/argonne_tpc/chia-clark 

# Set up MMAE environment
source "lm_eval_init_sophia.sh"
source "setup_test_model_params_llm_harness_sophia.sh"
echo "Executing in $PBS_O_WORKDIR/.."
cd "$PBS_O_WORKDIR"
echo `pwd`
echo

adapter="bioset_result"
OUTPUT_DIR="bvbrc_json_narrative_output_llama3/${adapter}"
DATA_DIR="/home/cc8dm/RAG-eval/BiodataTestingRubric/bv-brc/parsed_genome_data"

echo "Before: $(ls ${OUTPUT_DIR}/ | wc -l)" >> ${adapter}.counts
cmd="python3 generate_prompts_from_json_multithread.py -j ${DATA_DIR}/${adapter}.json -o ${OUTPUT_DIR}/${adapter}"
echo $cmd
eval $cmd
echo "After: $(ls ${OUTPUT_DIR}/ | wc -l)" >> ${adapter}.counts
