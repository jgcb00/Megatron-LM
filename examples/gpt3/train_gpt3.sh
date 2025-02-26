 #!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1

CHECKPOINT_PATH=$1 #<Specify path>
TENSORBOARD_LOGS_PATH=$2 #<Specify path>
VOCAB_FILE=$3 #<Specify path to file>/gpt2-vocab.json
DATA_PATH=$4 #<Specify path and file prefix>_text_document

GPT_MODEL_ARGS=(
    --num-layers 12
    --hidden-size 768
    --num-attention-heads 6
    --seq-length 1024
    --max-position-embeddings 1024
    --seed 42
)

TRAINING_ARGS=(
    --num-workers 1
    --micro-batch-size 8
    --train-samples 12207050
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.95
    --init-method-std 0.006
    --clip-grad 1.0
    --bf16
    --lr 3.0e-4
    --lr-decay-style cosine
    --min-lr 1.0e-5
    --lr-warmup-fraction .001
    #--lr-decay-iters 430000 
    #--use-flash-attn
    #--use-distributed-optimizer
    # --sequence-parallel
    #--overlap-param-gather 
    #--overlap-grad-reduce 
    --normalization RMSNorm
)

MODEL_PARALLEL_ARGS=(
	--tensor-model-parallel-size 1
	--pipeline-model-parallel-size 1
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
    --ckpt-format torch
    --log-validation-ppl-to-tensorboard
    --log-memory-to-tensorboard
    --log-world-size-to-tensorboard
    --log-throughput
)

torchrun --standalone --nnodes=1 --nproc-per-node=1 ../../Megatron-LM/pretrain_gpt.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]}
