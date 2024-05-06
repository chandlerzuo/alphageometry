#!/usr/bin/env bash

shopt -s expand_aliases
alias launch_condor_job="~/reinforcement/alphageometry/LLM_finetuner/launch_condor_job_new.py \
    160 --max_running_price -1 ---" #--dry 

run_dir_name=run_03052024

for num_train_samples in 1000 -1; do

# llama2 with peft
# launch_condor_job ~/reinforcement/alphageometry/LLM_finetuner/run_with_accelerate.sh \
#   ~/reinforcement/alphageometry/LLM_finetuner/sft_finetuning.py \
#   --overwrite_output_dir \
#   --use_peft \
#   --per_device_train_batch_size 32 \
#   --per_device_eval_batch_size 32 \
#   --model_name_or_path meta-llama/Llama-2-7b-chat-hf \
#   --max_eval_samples 400 \
#   --explicit_eos_str '[END]' \
#   --extra_tokens_file ~/reinforcement/alphageometry/assets/def-patterns-desc.yml \
#   --output_dir ~/reinforcement/alphageometry/LLM_finetuner/runs/verbalization/training/${run_dir_name}/{model_name}_{max_train_samples}ex_peft{use_peft} \
#   --dataset_name ~/reinforcement/alphageometry/LLM_finetuner/runs/verbalization/datasets/alpha_geo_processed \
#   --config ~/reinforcement/alphageometry/LLM_finetuner/trl_sft_config.yml \
#   --max_train_samples "$num_train_samples" \
#   --num_train_epochs 100000

# gpt2 with peft
launch_condor_job ~/reinforcement/alphageometry/LLM_finetuner/run_with_accelerate.sh \
  ~/reinforcement/alphageometry/LLM_finetuner/sft_finetuning.py \
  --overwrite_output_dir \
  --use_peft \
  --per_device_train_batch_size 64 \
  --per_device_eval_batch_size 64 \
  --model_name_or_path meta-llama/Llama-2-7b-hf \
#  --model_name_or_path gpt2 \
  --max_eval_samples 400 \
  --explicit_eos_str '[END]' \
  --extra_tokens_file ~/reinforcement/alphageometry/assets/def-patterns-desc.yml \
  --output_dir /is/cluster/fast/pghosh/ouputs/alpha_geo/${run_dir_name}/{model_name}_{max_train_samples}ex_peft{use_peft} \
  --dataset_name /is/cluster/scratch/pghosh/dataset/alpha_geo_processed \
  --config ~/reinforcement/alphageometry/LLM_finetuner/trl_sft_config.yml \
  --max_train_samples "$num_train_samples" \
  --num_train_epochs 100000
# does not work with peft (cublas error on 95Gb GPU)
#   --load_in_8bit \

## gpt2 without peft
#launch_condor_job ~/reinforcement/alphageometry/LLM_finetuner/run_with_accelerate.sh \
#  ~/reinforcement/alphageometry/LLM_finetuner/sft_finetuning.py \
#  --overwrite_output_dir \
#  --per_device_train_batch_size 64 \
#  --per_device_eval_batch_size 64 \
#  --model_name_or_path gpt2 \
#  --max_eval_samples 400 \
#  --explicit_eos_str '[END]' \
#  --extra_tokens_file ~/reinforcement/alphageometry/assets/def-patterns-desc.yml \
#  --output_dir ~/reinforcement/alphageometry/LLM_finetuner/runs/verbalization/training/${run_dir_name}/{model_name}_{max_train_samples}ex_peft{use_peft} \
#  --dataset_name ~/reinforcement/alphageometry/LLM_finetuner/runs/verbalization/datasets/alpha_geo_processed \
#  --config ~/reinforcement/alphageometry/LLM_finetuner/trl_sft_config.yml \
#  --max_train_samples "$num_train_samples" \
#  --num_train_epochs 100000

done

# llama: use_peft: bs=64 requires 83 GB, exceeding A100

#   use_peft: --load_in_8bit runs into Cublas error

#   without peft:
#   --bf16 \
#   --per_device_train_batch_size 1 \
#   --per_device_eval_batch_size 1 \
#   not working
#   --eval_steps 10 \

#   --eval_steps 10, max train samples, num epochs, delete output dir, prepare bash script,
#   git push

# --load_in_8bit: requires use_peft
# num train samples 1000, 