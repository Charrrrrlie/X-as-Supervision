#!/bin/bash

# Assign positional arguments to variables
partition=$1
num_gpu=$2
config_path=$3
checkpoint_path=$4
eval_mode=$5
extra_tag_info=$6

num_cpu=$((num_gpu * 6))

# default eval_mode "best"
if [ -z "$eval_mode" ]; then
    eval_mode="best"
fi

if [ ! -d "slurm_output" ]; then
    mkdir -p "slurm_output"
fi

OMP_NUM_THREADS=10 sbatch --gres=gpu:$num_gpu -n 1 --cpus-per-task=$num_cpu -p $partition -A $partition -o slurm_output/log.eval.out.%j\
    eval.sh $num_gpu $config_path $checkpoint_path $eval_mode $extra_tag_info