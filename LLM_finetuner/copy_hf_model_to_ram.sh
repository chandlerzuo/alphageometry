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

# source ~/reinforcement/HumbleAttemptAtGeneralAI/geometry_translation/new/copy_hf_model_to_ram.sh models--meta-llama--Llama-2-7b-hf

# rm -rf /dev/shm/huggingface
ls /dev/shm
mkdir -p /dev/shm/huggingface/hub/
# !cp -r ~/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/ /dev/shm/huggingface/hub/
# !cp -r ~/.cache/huggingface/hub/version.txt /dev/shm/huggingface/hub/ # otherwise migrating the cache
# !cp -r ~/.cache/huggingface/hub /dev/shm/huggingface/
# !cp -r ~/.cache/huggingface/hub/models--gpt2/ /dev/shm/huggingface/hub/

if [ -f /dev/shm/huggingface/hub/"$1" ]; then
    echo "already copied model $1"
else
    cp -r ~/.cache/huggingface/hub/version.txt /dev/shm/huggingface/hub/ # otherwise migrating the cache
    # unsafe: cp -r ~/.cache/huggingface/token /dev/shm/huggingface/hub/token # unsafe, set an env variable, see below
    cp -r ~/.cache/huggingface/hub/"$1" /dev/shm/huggingface/hub/
fi

# split "export HF_TOKEN", see https://www.shellcheck.net/wiki/SC2155
HF_TOKEN=$(cat ~/.cache/huggingface/token)
export HF_TOKEN

export HF_HOME=/dev/shm/huggingface/



