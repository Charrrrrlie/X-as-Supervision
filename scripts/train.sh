#!/bin/bash

# Assign positional arguments to variables
num_gpu=$1
config_path=$2
extra_tag_info=$3

# Navigate to the root directory where train.py is located
# Assuming the script is in ./scripts/ relative to the root
cd ..

module load anaconda/2022.10
module load cuda/11.8
source activate ~/Envs/hpe3D

export PYTHONUNBUFFERED=1

# rand master-port
port=$(( ( RANDOM % 1000 )  + 10000 ))

# if scale in $config_path, use train2d3d.py
if [[ $config_path == *"TikTok"* ]]; then
    command="torchrun --nproc-per-node=$num_gpu --master_port=$port train2d3d.py --config scripts/$config_path"
else
    command="torchrun --nproc-per-node=$num_gpu --master_port=$port train.py --config scripts/$config_path"
fi

# Check if extra_tag_info is provided and is not empty
if [[ -n $extra_tag_info ]]; then
    command+=" --extra_tag $extra_tag_info"
fi

# Execute the command
echo "Running command: $command"
eval $command

# Optionally, return to the original directory
cd - 
