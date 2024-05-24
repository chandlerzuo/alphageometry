#!/usr/bin/env bash

source ~/reinforcement/alphageometry/LLM_finetuner/copy_hf_model_to_ram.sh models--meta-llama--Llama-2-7b-chat-hf
source ~/reinforcement/alphageometry/LLM_finetuner/copy_hf_model_to_ram.sh models--gpt2
source ~/reinforcement/alphageometry/LLM_finetuner/verbalization_venv3/bin/activate

exec "$@"