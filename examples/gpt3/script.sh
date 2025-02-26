#!/bin/bash

#SBATCH --nodes=1           # number of nodes
#SBATCH --ntasks-per-node=30 # number of tasks per node
#SBATCH --time=24:00:00              # time limits: here 1 hour
#SBATCH --error=train_model_1B_100BT.err            # standard error file
#SBATCH --output=train_model_1B_100BT.out           # standard output file
#SBATCH --account=BOOST_LCustodi       # account name
#SBATCH --partition=boost_qos_dbg # partition name for test
##SBATCH --partition=boost_usr_prod # partition name for prod



bash examples/gpt3/train_gpt3.sh \
    gpt3/ckpt/ \
    tensorboard/logs \
    experimental_tokenizer.json \
    zyda_data/