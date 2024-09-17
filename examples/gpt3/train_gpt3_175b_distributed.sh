 #!/bin/bash

# Runs the "175B" parameter model

export CUDA_DEVICE_MAX_CONNECTIONS=1
export MEGATRON_LOGGING_LEVEL=10

GPUS_PER_NODE=4
# Change for multinode config
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
#MASTER_ADDR="localhost"
MASTER_PORT=48994
NUM_NODES=4
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))
echo "Master Address : "$MASTER_ADDR
CHECKPOINT_PATH=$1 #<Specify path>
TENSORBOARD_LOGS_PATH=$2 #<Specify path>
VOCAB_FILE=$3 #<Specify path to file>/gpt2-vocab.json
DATA_PATH=$4 #<Specify path and file prefix>_text_document
#MASTER_NODE_IP=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NUM_NODES 
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
    --rdzv_id $SLURM_JOB_ID
    --rdzv_endpoint $MASTER_ADDR:29500
    --rdzv_backend c10d
)

GPT_MODEL_ARGS=(
    --num-layers 16 
    --hidden-size 2048 
    --num-attention-heads 8 
    --seq-length 4096 
    --max-position-embeddings 4096
    --seed 42
    --log-throughput
)

TRAINING_ARGS=(
    --num-workers 8 
    #--no-mmap-bin-files
    --micro-batch-size 60 
    #--global-batch-size 240 
    #--rampup-batch-size 20 10 250000 
    #--dataloader-type single
    --train-samples 10000000 
    --weight-decay 0.1 
    --adam-beta1 0.9 
    --adam-beta2 0.95 
    --init-method-std 0.006 
    --clip-grad 1.0 
    --bf16
    --lr 1.0e-4 
    --lr-decay-style cosine 
    --min-lr 1.0e-5
    --lr-warmup-fraction .001 
    #--lr-decay-iters 430000 
    --use-flash-attn
)

MODEL_PARALLEL_ARGS=(
	--tensor-model-parallel-size 4
	--pipeline-model-parallel-size 2
)

DATA_ARGS=(
    --data-path $DATA_PATH 
    --tokenizer-type HuggingFacePretrainedTokenizer
    --tokenizer-model $VOCAB_FILE
    --split 975,24,1
    --vocab-size 50304
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 100
    --save-interval 10000 
    --eval-interval 1000 
    --save $CHECKPOINT_PATH 
    --load $CHECKPOINT_PATH 
    --eval-iters 10
    --tensorboard-dir $TENSORBOARD_LOGS_PATH 
)

srun torchrun ${DISTRIBUTED_ARGS[@]} ../../Megatron-LM/pretrain_gpt.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]}
