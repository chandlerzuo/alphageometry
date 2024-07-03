#!/usr/bin/env bash

# Run standalone (no args) as
# source ~/reinforcement/alphageometry/LLM_finetuner/setup_env.sh

# source ~/reinforcement/alphageometry/LLM_finetuner/copy_hf_model_to_ram.sh models--meta-llama--Llama-2-7b-chat-hf
# source ~/reinforcement/alphageometry/LLM_finetuner/copy_hf_model_to_ram.sh models--gpt2
source ~/reinforcement/alphageometry/LLM_finetuner/copy_hf_model_to_ram.sh models--meta-llama--Meta-Llama-3-8B-Instruct
source ~/reinforcement/alphageometry/LLM_finetuner/verbalization_venv3/bin/activate

# export HF_HOME=/fast/mmordig/huggingface_cache/ # very slow
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

exec "$@"
# equivalent, https://askubuntu.com/questions/525767/what-does-an-exec-command-do
# empty exec mostly used to perform redirections

# if [ "$#" -eq 0 ]; then
#     echo "No arguments provided, so no exec"
# else
#     exec "$@"
# fi