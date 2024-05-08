#!/usr/bin/env bash
source ~/reinforcement/alphageometry/LLM_finetuner/verbalization_venv3/bin/activate
set -eux

GPUS_PER_NODE=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
export ACCELERATE_LOG_LEVEL=info

accelerate launch --config_file ~/reinforcement/alphageometry/LLM_finetuner/distributed_tests/distributed_example.yaml \
    --num_processes $((NUM_MACHINES * GPUS_PER_NODE)) \
    --num_machines "$NUM_MACHINES" \
    --rdzv_backend c10d \
    --main_process_ip "$MASTER_IP" \
    --main_process_port 29500 \
    --machine_rank "$MACHINE_RANK" \
    "$@"
    # LLM_finetuner/distributed_tests/wait_for_everyone.py
    # --cpu \
    # --num_cpu_threads_per_process 2 \