#!/usr/bin/env bash

# ~/reinforcement/alphageometry/LLM_finetuner/run_jobs_02052024.sh

shopt -s expand_aliases

# interactive (NO condor-submit), single node, single gpu
alias launch_condor_job="~/reinforcement/alphageometry/LLM_finetuner/setup_env.sh"

# interactive (NO condor-submit), single node, multi-gpu
# alias launch_condor_job="~/reinforcement/alphageometry/LLM_finetuner/setup_env.sh \
#   ~/reinforcement/alphageometry/LLM_finetuner/accelerate_launch_singlenode.sh"

# condor-submit, single node, multi-gpu
# alias launch_condor_job="~/reinforcement/alphageometry/LLM_finetuner/condor_submit_with_extra_args.py 100 \
#     --submission-file ~/reinforcement/alphageometry/LLM_finetuner/sft_singlenode.sub \
#     --- \
#     ~/reinforcement/alphageometry/LLM_finetuner/setup_env.sh \
#     ~/reinforcement/alphageometry/LLM_finetuner/accelerate_launch_singlenode.sh"
    # --dry \

# condor-submit, multiple nodes, multiple GPUs
# alias launch_condor_job="~/reinforcement/alphageometry/LLM_finetuner/condor_submit_with_extra_args.py 100 \
#     --submission-file ~/reinforcement/alphageometry/LLM_finetuner/sft_distributed.sub \
#     --- \
#     ~/reinforcement/alphageometry/LLM_finetuner/setup_env.sh \
#     ~/reinforcement/alphageometry/LLM_finetuner/accelerate_launch_distributed.sh"
    # --dry \

# run_dir_name=run_03052024
# run_dir_name=run_09052024
run_dir_name=run_27052024_3 # interactive, single node
# run_dir_name=run_27052024 # distributed
# run_dir_name=run_27052024_3 # interactive, single node

cd ~/reinforcement/alphageometry/LLM_finetuner/runs/verbalization/training/${run_dir_name} # for wandb save directory

num_train_samples=256
# num_train_samples=10
# for num_train_samples in 192 100000; do #-1; do #1000; do # -1; do

# model_name=gpt2
# model_name=meta-llama/Llama-2-7b-chat-hf
model_name=meta-llama/Meta-Llama-3-8B-Instruct
# for model_name in meta-llama/Llama-2-7b-chat-hf gpt2; do
# # llama2 with peft
launch_condor_job \
  ~/reinforcement/alphageometry/LLM_finetuner/sft_finetuning.py \
  --overwrite_output_dir \
  --use_peft \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 32 \
  --model_name_or_path "${model_name}" \
  --max_eval_samples 400 \
  --max_generate_samples 3 \
  --logging_strategy steps --logging_steps 50 \
  --gen_steps 20 \
  --resume_from_checkpoint false \
  --learning_rate 0.00001 \
  --output_dir ~/reinforcement/alphageometry/LLM_finetuner/runs/verbalization/training/${run_dir_name}/{model_name}_{max_train_samples}ex_peft{use_peft} \
  --dataset_name ~/reinforcement/alphageometry/LLM_finetuner/runs/verbalization/datasets/alpha_geo_arrow \
  --config ~/reinforcement/alphageometry/LLM_finetuner/trl_sft_config.yml \
  --max_train_samples "$num_train_samples" \
  --num_train_epochs 100000
# done
# done

# extra tokens does not seem to work well due to tokenization problems
# --extra_tokens_file ~/reinforcement/alphageometry/assets/def-patterns-desc.yml \
# todo: do model generate with a dataloader (so it works with accelerate), why is it so slow and memory so low compared to previously?
  # --explicit_eos_str '[END]' \
# --model_name_or_path gpt2 \
  # --per_device_train_batch_size 64 \
  # --per_device_eval_batch_size 64 \
  # --model_name_or_path meta-llama/Llama-2-7b-chat-hf \
  # --logging_strategy steps --logging_steps 50 \
  # --gen_steps 200 \
  # --dataset_test_name train \
  # --extra_tokens_file ~/reinforcement/alphageometry/assets/def-patterns-desc.yml \

# # gpt2 with peft
# launch_condor_job \
#   ~/reinforcement/alphageometry/LLM_finetuner/sft_finetuning.py \
#   --overwrite_output_dir \
#   --use_peft \
#   --per_device_train_batch_size 64 \
#   --per_device_eval_batch_size 64 \
#   --model_name_or_path gpt2 \
#   --max_eval_samples 400 \
#   --max_generate_samples 10 \
#   --explicit_eos_str '[END]' \
#   --extra_tokens_file ~/reinforcement/alphageometry/assets/def-patterns-desc.yml \
#   --output_dir ~/reinforcement/alphageometry/LLM_finetuner/runs/verbalization/training/${run_dir_name}/{model_name}_{max_train_samples}ex_peft{use_peft} \
#   --dataset_name ~/reinforcement/alphageometry/LLM_finetuner/runs/verbalization/datasets/alpha_geo_arrow \
#   --config ~/reinforcement/alphageometry/LLM_finetuner/trl_sft_config.yml \
#   --max_train_samples "$num_train_samples" \
#   --num_train_epochs 100000
# # does not work with peft (cublas error on 95Gb GPU)
# #   --load_in_8bit \

# # gpt2 without peft
# launch_condor_job \
#   ~/reinforcement/alphageometry/LLM_finetuner/sft_finetuning.py \
#   --overwrite_output_dir \
#   --per_device_train_batch_size 64 \
#   --per_device_eval_batch_size 64 \
#   --model_name_or_path gpt2 \
#   --max_eval_samples 400 \
#   --max_generate_samples 10 \
#   --explicit_eos_str '[END]' \
#   --extra_tokens_file ~/reinforcement/alphageometry/assets/def-patterns-desc.yml \
#   --output_dir ~/reinforcement/alphageometry/LLM_finetuner/runs/verbalization/training/${run_dir_name}/{model_name}_{max_train_samples}ex_peft{use_peft} \
#   --dataset_name ~/reinforcement/alphageometry/LLM_finetuner/runs/verbalization/datasets/alpha_geo_arrow \
#   --config ~/reinforcement/alphageometry/LLM_finetuner/trl_sft_config.yml \
#   --max_train_samples "$num_train_samples" \
#   --num_train_epochs 100000


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