#!/bin/bash -l
#PBS -l select=8:system=sophia
#PBS -l walltime=5:00:00
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
#source "$ROOT/init.sh"
source "lm_eval_init_sophia.sh"
source "setup_test_model_params_llm_harness_sophia.sh"
#MODEL_NAME=/lus/eagle/projects/argonne_tpc/chia-llama2/autotrain/Llama-2-7b-chat-hf
#OUTPUT_FILE=/lus/eagle/projects/argonne_tpc/chia-clark/BiodataTestingRubric/bvbrc_json_tests/llm_harness/bioset_result_
#ADAPTER_DIR=/lus/eagle/projects/argonne_tpc/cucinell/BiodataTestingRubric/json_tests/bioset_result_adapters/TMP_RESULTS_Llama-2-7b-chat-hf_
#TASK=winogrande
#NUM_SHOT=5
#EPOCH=0.2
#export SOURCE_DATASET_NAME=TCGA
EPOCH1=$EPOCH
EPOCH2=$(echo "$EPOCH + 1.0" | bc | sed 's/^\./0./')
EPOCH3=$(echo "$EPOCH + 2.0" | bc | sed 's/^\./0./')
EPOCH4=$(echo "$EPOCH + 3.0" | bc | sed 's/^\./0./')
EPOCH5=$(echo "$EPOCH + 4.0" | bc | sed 's/^\./0./')
EPOCH6=$(echo "$EPOCH + 5.0" | bc | sed 's/^\./0./')
EPOCH7=$(echo "$EPOCH + 6.0" | bc | sed 's/^\./0./')
EPOCH8=$(echo "$EPOCH + 7.0" | bc | sed 's/^\./0./')

echo "Executing in $PBS_O_WORKDIR/.."
cd "$PBS_O_WORKDIR"
echo `pwd`
echo

adapter1=${ADAPTER_DIR}${EPOCH1}
echo $adapter1
if [ -d $adapter1 ]; then
	cmd1="CUDA_VISIBLE_DEVICES=0 lm_eval --model hf \
		--batch_size=4 \
		--model_args pretrained=${MODEL_NAME},peft=${adapter1},trust_remote_code=True \
		--tasks ${TASK} \
		--num_fewshot ${NUM_SHOT} \
		--output_path ${OUTPUT_FILE}${EPOCH1}_${TASK}_${NUM_SHOT}.json > ${OUTPUT_FILE}${EPOCH1}_${TASK}_${NUM_SHOT}.debug"

	echo "Epoch $EPOCH1"
	echo $cmd1
	eval $cmd1 &

fi

adapter2=${ADAPTER_DIR}${EPOCH2}
echo $adapter2
if [ -d $adapter2 ]; then
	cmd2="CUDA_VISIBLE_DEVICES=1 lm_eval --model hf \
		--batch_size=4 \
		--model_args pretrained=${MODEL_NAME},peft=${adapter_2},trust_remote_code=True \
		--tasks ${TASK} \
		--num_fewshot ${NUM_SHOT} \
		--output_path ${OUTPUT_FILE}${EPOCH2}_${TASK}_${NUM_SHOT}.json > ${OUTPUT_FILE}${EPOCH2}_${TASK}_${NUM_SHOT}.debug"

	echo "Epoch $EPOCH2"
	echo $cmd2
	eval $cmd2 &
fi

adapter3=${ADAPTER_DIR}${EPOCH3}
echo $adapter3
if [ -d $adapter3 ]; then
	cmd3="CUDA_VISIBLE_DEVICES=2 lm_eval --model hf \
		--batch_size=4 \
		--model_args pretrained=${MODEL_NAME},peft=${adapter3},trust_remote_code=True \
		--tasks ${TASK} \
		--num_fewshot ${NUM_SHOT} \
		--output_path ${OUTPUT_FILE}${EPOCH3}_${TASK}_${NUM_SHOT}.json > ${OUTPUT_FILE}${EPOCH3}_${TASK}_${NUM_SHOT}.debug"

	echo "Epoch $EPOCH3"
	echo $cmd3
	eval $cmd3 &
fi

adapter4=${ADAPTER_DIR}${EPOCH4}
echo $adapter4
if [ -d $adapter4 ]; then
	cmd4="CUDA_VISIBLE_DEVICES=3 lm_eval --model hf \
		--batch_size=4 \
		--model_args pretrained=${MODEL_NAME},peft=${adapter4},trust_remote_code=True \
		--tasks ${TASK} \
		--num_fewshot ${NUM_SHOT} \
		--output_path ${OUTPUT_FILE}${EPOCH4}_${TASK}_${NUM_SHOT}.json > ${OUTPUT_FILE}${EPOCH4}_${TASK}_${NUM_SHOT}.debug"

	echo "Epoch $EPOCH4"
	echo $cmd4
	eval $cmd4 &
fi

adapter5=${ADAPTER_DIR}${EPOCH5}
echo $adapter5
if [ -d $adapter5 ]; then
	cmd5="CUDA_VISIBLE_DEVICES=4 lm_eval --model hf \
		--batch_size=4 \
		--model_args pretrained=${MODEL_NAME},peft=${adapter5},trust_remote_code=True \
		--tasks ${TASK} \
		--num_fewshot ${NUM_SHOT} \
		--output_path ${OUTPUT_FILE}${EPOCH5}_${TASK}_${NUM_SHOT}.json > ${OUTPUT_FILE}${EPOCH5}_${TASK}_${NUM_SHOT}.debug"

	echo "Epoch $EPOCH5"
	echo $cmd5
	eval $cmd5 &
fi

adapter6=${ADAPTER_DIR}${EPOCH6}
echo $adapter6
if [ -d $adapter6 ]; then
	cmd6="CUDA_VISIBLE_DEVICES=5 lm_eval --model hf \
		--batch_size=4 \
		--model_args pretrained=${MODEL_NAME},peft=${adapter6},trust_remote_code=True \
		--tasks ${TASK} \
		--num_fewshot ${NUM_SHOT} \
		--output_path ${OUTPUT_FILE}${EPOCH6}_${TASK}_${NUM_SHOT}.json > ${OUTPUT_FILE}${EPOCH6}_${TASK}_${NUM_SHOT}.debug"

	echo "Epoch $EPOCH6"
	echo $cmd6
	eval $cmd6 &
fi

adapter7=${ADAPTER_DIR}${EPOCH7}
echo $adapter7
if [ -d $adapter7 ]; then
	cmd7="CUDA_VISIBLE_DEVICES=6 lm_eval --model hf \
		--batch_size=4 \
		--model_args pretrained=${MODEL_NAME},peft=${adapter7},trust_remote_code=True \
		--tasks ${TASK} \
		--num_fewshot ${NUM_SHOT} \
		--output_path ${OUTPUT_FILE}${EPOCH7}_${TASK}_${NUM_SHOT}.json > ${OUTPUT_FILE}${EPOCH7}_${TASK}_${NUM_SHOT}.debug"

	echo "Epoch $EPOCH7"
	echo $cmd7
	eval $cmd7 &
fi

adapter8=${ADAPTER_DIR}${EPOCH8}
echo $adapter8
if [ -d $adapter8 ]; then
	cmd8="CUDA_VISIBLE_DEVICES=7 lm_eval --model hf \
		--batch_size=4 \
		--model_args pretrained=${MODEL_NAME},peft=${adapter8},trust_remote_code=True \
		--tasks ${TASK} \
		--num_fewshot ${NUM_SHOT} \
		--output_path ${OUTPUT_FILE}${EPOCH8}_${TASK}_${NUM_SHOT}.json > ${OUTPUT_FILE}${EPOCH8}_${TASK}_${NUM_SHOT}.debug"

	echo "Epoch $EPOCH8"
	echo $cmd8
	eval $cmd8 &
fi

wait
