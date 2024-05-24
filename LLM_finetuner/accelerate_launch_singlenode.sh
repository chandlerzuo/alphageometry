#!/usr/bin/env bash

num_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
extra_args=""
if [ "$num_gpus" -gt 1 ]; then
    # required, otherwise uses one gpu only even when num_processes is specified
    extra_args="--multi_gpu"
fi
accelerate launch --config_file ~/reinforcement/alphageometry/LLM_finetuner/example_accelerate_config.yaml $extra_args --num_processes "$num_gpus" "$@"


# OLD
# ##CONDOR request_cpus=20
# ##CONDOR request_memory=256000
# ##CONDOR request_disk=100G
# ##CONDOR +JobBatchName = "verb_sft"
# ##CONDOR log = /home/mmordig/joblogs/job-$(ClusterId)-$(ProcId).$$([NumJobStarts]).log
# ##CONDOR output = /home/mmordig/joblogs/job-$(ClusterId)-$(ProcId).$$([NumJobStarts]).out
# ##CONDOR error = /home/mmordig/joblogs/job-$(ClusterId)-$(ProcId).$$([NumJobStarts]).err
# ##CONDOR request_gpus=4
# ##CONDOR requirements = (TARGET.CUDAGlobalMemoryMb > 50000) && (Machine != "g101.internal.cluster.is.localnet")

# ##-CONDOR +MaxRuntime = 4500
# # see # https://atlas.is.localnet/confluence/display/IT/Limit+the+running+price+of+a+job
# # https://atlas.is.localnet/confluence/display/IT/How+to+limit+the+running+time+of+a+job+in+the+cluster
# ##-CONDOR +MaxRunningPrice = 30
# ##-CONDOR +RunningPriceExceededAction = "restart"
# # Maximum expected execution time for the job, in seconds
# # Number of retries before giving up
# ##-CONDOR MaxTime = 3600
# ##-CONDOR NumRetries = 5
# ##-CONDOR periodic_hold = (JobStatus =?= 2) && ((CurrentTime - JobCurrentStartDate) >= $(MaxTime))
# ##-CONDOR periodic_hold_reason = ifThenElse(JobRunCount <= $(NumRetries), "Job runtime exceeded", "Job runtime exceeded, no more retries left")
# ##-CONDOR periodic_hold_subcode = ifThenElse(JobRunCount <= $(NumRetries), 1, 2)
# ##-CONDOR periodic_release = ( (JobStatus =?= 5) && (HoldReasonCode =?= 3) && (HoldReasonSubCode =?= 1) )
# # HoldReasonCode =?= 3: job was put on hold
# # periodic release: whether to add job back to the queue
 
# # Uncomment this line if you want the jobs automatically removed from the queue
# # periodic_remove = ( (JobStatus =?= 5) && (HoldReasonCode =?= 3) && (HoldReasonSubCode =?= 2) )


# # condor_submit_autokill 20 -append 'request_cpus=8' -append 'request_memory=128GB' -append 'request_disk=100GB' -append 'request_gpus=1' -append 'requirements = (TARGET.CUDAGlobalMemoryMb > 50000) && (Machine != "g101.internal.cluster.is.localnet")' -i
# # condor_submit_autokill 20 -append 'request_cpus=16' -append 'request_memory=256GB' -append 'request_disk=100GB' -append 'request_gpus=2' -append 'requirements = (TARGET.CUDAGlobalMemoryMb > 50000) && (Machine != "g101.internal.cluster.is.localnet")' -i

# # alias launch_condor_job=/home/mmordig/reinforcement/HumbleAttemptAtGeneralAI/geometry_translation/new/launch_condor_job_new.py
# # launch_condor_job 30 --max_running_price -1 --- bash ~/reinforcement/HumbleAttemptAtGeneralAI/geometry_translation/new/accelerate_launch_singlenode.sh
# # or
# # bash ~/reinforcement/HumbleAttemptAtGeneralAI/geometry_translation/new/accelerate_launch_singlenode.sh


# #tmux
# source ~/.bashrc
# pwd
# set -eux
# # module load cuda/12.1
# # source /etc/profile.d/modules.sh # seems to be overwriting the path
# # module load cuda/12.1 # seems to mess up the path
# source ~/reinforcement/alphageometry/LLM_finetuner/copy_hf_model_to_ram.sh models--meta-llama--Llama-2-7b-chat-hf
# source ~/reinforcement/alphageometry/LLM_finetuner/copy_hf_model_to_ram.sh models--gpt2
# source ~/reinforcement/alphageometry/LLM_finetuner/verbalization_venv3/bin/activate
# function accelerate_cmd() {
#     accelerate launch --config_file ~/reinforcement/alphageometry/LLM_finetuner/example_accelerate_config.yaml --multi_gpu --num_processes "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" "$@"
# }
# ## move there, no locks supported, export HF_DATASETS_CACHE=/fast/mmordig/hf_cache/datasets
# accelerate_cmd "$@"