#!/usr/bin/env bash

# Run accelerate on two nodes, hardcoded IP
set -eux

num_nodes=2
GPUS_PER_NODE=1
head_node_ip=127.0.0.1
# head_node_ip=172.22.2.161
# head_node_ip=108.23.24.32
export ACCELERATE_LOG_LEVEL=info
source ~/reinforcement/alphageometry/LLM_finetuner/verbalization_venv3/bin/activate
accelerate launch --config_file LLM_finetuner/distributed_tests/distributed_example.yaml \
    --num_processes $((num_nodes * GPUS_PER_NODE)) \
    --num_machines $num_nodes \
    --rdzv_backend c10d \
    --main_process_ip $head_node_ip \
    --main_process_port 29500 \
    "$@"
    # LLM_finetuner/distributed_tests/wait_for_everyone.py
    # --cpu \
    # --num_cpu_threads_per_process 2 \