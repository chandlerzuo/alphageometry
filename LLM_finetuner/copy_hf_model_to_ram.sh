#!/usr/bin/env bash
# will be sourced, so don't set: set -eu #x

## todo: error happens when sourcing this: ls, mkdir not found, maybe due to "module load cuda/12.1"
##CONDOR request_cpus=1
##CONDOR request_memory=1000
##CONDOR +JobBatchName = "ddddd"
##CONDOR log = /home/mmordig/joblogs/job-$(ClusterId)-$(ProcId).$$([NumJobStarts]).log
##CONDOR output = /home/mmordig/joblogs/job-$(ClusterId)-$(ProcId).$$([NumJobStarts]).out
##CONDOR error = /home/mmordig/joblogs/job-$(ClusterId)-$(ProcId).$$([NumJobStarts]).err
# pwd
# mkdir -p tttt
# launch_condor_job 30 --max_running_price -1 --- ~/reinforcement/HumbleAttemptAtGeneralAI/geometry_translation/new/copy_hf_model_to_ram.sh
##CONDOR periodic_remove = NumRestarts > 0

# add a new model
# huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct
# cp -r /home/mmordig/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct /fast/mmordig/huggingface_cache/hub

# source ~/reinforcement/HumbleAttemptAtGeneralAI/geometry_translation/new/copy_hf_model_to_ram.sh models--meta-llama--Llama-2-7b-hf

# rm -rf /dev/shm/huggingface
# hf_home=/dev/shm/huggingface/
hf_home=$TMPDIR/huggingface/
echo "Copying hf cache to tempdir instead!!"
new_cache_dir=$hf_home/hub/
model=$1

mkdir -p "$new_cache_dir"
echo "Content of $new_cache_dir:"
ls "$new_cache_dir"
# !cp -r ~/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/ /dev/shm/huggingface/hub/
# !cp -r ~/.cache/huggingface/hub/version.txt /dev/shm/huggingface/hub/ # otherwise migrating the cache
# !cp -r ~/.cache/huggingface/hub /dev/shm/huggingface/
# !cp -r ~/.cache/huggingface/hub/models--gpt2/ /dev/shm/huggingface/hub/

copied_models_filename=$new_cache_dir/copied_models.txt
# check if model name appears in copied_models_filename
function was_copied() {
    [ -f "$copied_models_filename" ] && grep -q "$1" "$copied_models_filename"
}

# if [ -d "$new_cache_dir""$model" ]; then
if was_copied "$model"; then
    echo "already copied model $model"
else
    echo "copying model $model to RAM"
    cp -r ~/.cache/huggingface/hub/version.txt "$new_cache_dir" # otherwise migrating the cache
    # unsafe: cp -r ~/.cache/huggingface/token "$new_cache_dir"token # unsafe, set an env variable, see below
    cp -r ~/.cache/huggingface/hub/"$model" "$new_cache_dir"
fi
echo "$model" >> "$copied_models_filename"

# split "export HF_TOKEN", see https://www.shellcheck.net/wiki/SC2155
HF_TOKEN=$(cat ~/.cache/huggingface/token)
export HF_TOKEN

# export HF_HOME=/dev/shm/huggingface/
export HF_HOME=$hf_home
# export HF_HOME=/fast/mmordig/huggingface_cache/



