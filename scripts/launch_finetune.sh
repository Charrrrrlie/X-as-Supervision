#!/bin/bash

# Assign positional arguments to variables
partition=$1
num_gpu=$2
config_path=$3
checkpoint=$4
extra_tag_info=$5

num_cpu=$((num_gpu * 10))
num_cpu=$((num_cpu > 48 ? 48 : num_cpu))

if [ ! -d "slurm_output" ]; then
    mkdir -p "slurm_output"
fi

OMP_NUM_THREADS=10 sbatch --gres=gpu:$num_gpu -n 1 --cpus-per-task=$num_cpu -p $partition -A $partition -o slurm_output/log.out.%j\
    finetune.sh $num_gpu $config_path $checkpoint $extra_tag_info