#!/usr/bin/env bash

# source ~/reinforcement/alphageometry/LLM_finetuner/setup_commands.sh

source ~/reinforcement/alphageometry/LLM_finetuner/verbalization_venv3/bin/activate
export VERB_RUN_DIR=~/reinforcement/LLM_finetuner/runs/verbalization

export PATH=$PATH:~/reinforcement/alphageometry/LLM_finetuner

# source ~/reinforcement/alphageometry/LLM_finetuner/copy_hf_model_to_ram.sh models--meta-llama--Llama-2-7b-hf
# source ~/reinforcement/alphageometry/LLM_finetuner/copy_hf_model_to_ram.sh models--gpt2