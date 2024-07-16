#!/bin/bash
export PYTHONPATH=$PYTHONPATH:../
samples_per_run=10000

# Check if the first argument is provided and is a numeric value
if [[ -z "$1" || ! "$1" =~ ^[0-9]+$ ]]; then
    echo "Error: Argument not provided or not a valid number."
    echo "Usage: $0 <numeric_argument>"
    exit 1
fi

# Use $(( )) for arithmetic calculations
seed_value=$(($1 * samples_per_run / 100))

# Run the Python script
/home/pghosh/miniconda3/envs/alpha_geo/bin/python demo.py \
--out /is/cluster/scratch/pghosh/dataset/alpha_geo/arithmetic/arithmetic_probs_depth_2_base_funcs_65/$1.csv \
--n $samples_per_run \
--seed $seed_value \
--pbar False \
--overwrite True

## Echo the command with variables expanded to show which command was executed
#echo "/home/pghosh/miniconda3/envs/alpha_geo/bin/python demo.py --out /is/cluster/scratch/pghosh/dataset/alpha_geo/arithmetic/$1.csv -n $samples_per_run --seed $seed_value --pbar False --overwrite True"
