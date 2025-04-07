#!/bin/bash
#SBATCH --nodes=1           # number of nodes
#SBATCH --ntasks-per-node=1 # number of tasks per node
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1         # number of gpus per node
#SBATCH --time=24:00:00              # time limits: here 1 hour
#SBATCH --error=logs/experiment1.err            # standard error file
#SBATCH --output=logs/experiment1.out           # standard output file
#SBATCH --account=BOOST_LCustodi       # account name
#SBATCH --partition=boost_usr_prod # partition name for prod

module load gcc/12.2.0 python/3.11.6--gcc--8.5.0 cuda/12.1 cudnn cutensor/1.5.0.3--gcc--12.2.0-cuda-12.1

source /leonardo_work/BOOST_LCustodi/script/training/torch2.5_training_env/bin/activate

export HF_DATASETS_CACHE="/leonardo_work/BOOST_LCustodi/hf_cache"
export WANDB_MODE=offline

GPUS_PER_NODE=1
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=48994
NUM_NODES=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l)
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))
echo "Master Address : "$MASTER_ADDR" | "$NUM_NODES" Nodes | World Size : "$WORLD_SIZE

EXP_NAME="dragon-megatron-test1"
RANDOM_UUID=$(uuidgen)

CHECKPOINT_PATH="${EXP_NAME}_${RANDOM_UUID}"
TENSORBOARD_LOGS_PATH="${CHECKPOINT_PATH}/tensorboard/"
VOCAB_FILE="../../prepared_data/experimental_tokenizer.json"
DATA_PATH="../../prepared_data/zyda_full_text_document"
    
echo "Master Address : "$MASTER_ADDR" | "$NUM_NODES" Nodes | World Size : "$WORLD_SIZE

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NUM_NODES 
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
    --rdzv_id $SLURM_JOB_ID
    --rdzv_endpoint $MASTER_ADDR:29500
    --rdzv_backend c10d
)

DRAGON_ARCH_ARGS=(
    --num-layers 8
    --hidden-size 512
    --ffn_hidden_size 2048
    --group-query-attention
    --num-attention-heads 8
    --num-query-groups 4 # same as n_kv_heads
    --seq-length 1024
    --sliding-window-attention 256
    --max-position-embeddings 1024
    --position-embedding-type rope
    --rotary-base 10000
    --rotary-percent 1.0
    --hidden-dropout 0.0
    --attention-dropout 0.0
    --no-add-bias-linear
    --normalization RMSNorm
    --norm-epsilon 1e-5
    --qk-layernorm
    --init-method-std 0.006 # deepseek init (init_method being None, it will default to torch.nn.init.normal_(mean=0.0, std=init_method_std))
    #--output_layer_init_method # we will have to get rid of this one
    --squared-relu
    --untie-embeddings-and-output-weights
    --spec megatron.core.models.dragon.dragon_layer_specs dragon_stack_spec
    --use-mcore-models # dont know what this is for. seems depreacted
    --no-create-attention-mask-in-dataloader
    --seed 42 # anointed by divine providence
)

# todo: hparams for fusion, activation recomputation, fp8 megatron/core/transformer/dragon_config.py lines 147-234

TRAINING_ARGS=(
    --num-workers $num_workers
    --micro-batch-size 16 # batchsize per device instance
    # no global-batch-size, it is calculated as micro-batch-size * data-parallel-size
    --train-samples 12207050
    --adam-beta1 0.9
    --adam-beta2 0.95
    --adam-eps 1e-8
    --clip-grad 1.0
    --bf16
    --lr 1.0e-3
    --lr-decay-style WSD
    --lr-wsd-decay-style linear
    --lr_warmup_samples 54931 # 0.0045 * train_samples
    --lr_wsd_decay_samples 1831057 # 15% of train_samples
    --lr-warmup-init 0.0
    --min-lr 0.0
    --weight-decay 0.1
    --weight-decay-incr-style constant
    --use-flash-attn
    #--use-distributed-optimizer
    #--sequence-parallel
    #--slw_warmup_steps 1000
)

MODEL_PARALLEL_ARGS=(
    --data-parallel-size 1
	--tensor-model-parallel-size 1
	--pipeline-model-parallel-size 1
)

DATA_ARGS=(
    --data-path $DATA_PATH
    --tokenizer-type HuggingFacePretrainedTokenizer
    --tokenizer-model $VOCAB_FILE
    --split 975,24,1
    --vocab-size 50304
    --make-vocab-size-divisible-by 128
)

#todo: inference args

EVAL_AND_LOGGING_ARGS=(
    --log-interval 100
    --save-interval 10000
    --eval-interval 1000
    --eval-iters 10
    --save $CHECKPOINT_PATH
    --load $CHECKPOINT_PATH
    --ckpt-format torch
    --tensorboard-dir $TENSORBOARD_LOGS_PATH
    --log-validation-ppl-to-tensorboard
    --log-memory-to-tensorboard
    --log-world-size-to-tensorboard
    --log-params-norm # maybe disable that one for perf
    --log-throughput
    --log-progress
    --wandb-project migration
    --wandb-exp-name $EXP_NAME
    --wandb-save-dir ../wandb
)

srun torchrun ${DISTRIBUTED_ARGS[@]} ../pretrain_dragon.py \
    ${DRAGON_ARCH_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]}
