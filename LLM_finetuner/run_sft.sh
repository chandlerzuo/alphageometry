#!/usr/bin/env bash

##CONDOR request_cpus=8
##CONDOR request_memory=128000
##CONDOR request_disk=100G
##CONDOR +JobBatchName = "verb_sft"
##CONDOR log = /home/mmordig/joblogs/job-$(ClusterId)-$(ProcId).$$([NumJobStarts]).log
##CONDOR output = /home/mmordig/joblogs/job-$(ClusterId)-$(ProcId).$$([NumJobStarts]).out
##CONDOR error = /home/mmordig/joblogs/job-$(ClusterId)-$(ProcId).$$([NumJobStarts]).err
##CONDOR request_gpus=1
##CONDOR requirements = (TARGET.CUDAGlobalMemoryMb > 50000) && (Machine != "g101.internal.cluster.is.localnet")

##CONDOR +MaxRuntime = 4500
# see # https://atlas.is.localnet/confluence/display/IT/Limit+the+running+price+of+a+job
# https://atlas.is.localnet/confluence/display/IT/How+to+limit+the+running+time+of+a+job+in+the+cluster
##CONDOR +MaxRunningPrice = 30
##CONDOR +RunningPriceExceededAction = "restart"
# Maximum expected execution time for the job, in seconds
# Number of retries before giving up
##CONDOR MaxTime = 3600
##CONDOR NumRetries = 5
##CONDOR periodic_hold = (JobStatus =?= 2) && ((CurrentTime - JobCurrentStartDate) >= $(MaxTime))
##CONDOR periodic_hold_reason = ifThenElse(JobRunCount <= $(NumRetries), "Job runtime exceeded", "Job runtime exceeded, no more retries left")
##CONDOR periodic_hold_subcode = ifThenElse(JobRunCount <= $(NumRetries), 1, 2)
##CONDOR periodic_release = ( (JobStatus =?= 5) && (HoldReasonCode =?= 3) && (HoldReasonSubCode =?= 1) )
# HoldReasonCode =?= 3: job was put on hold
# periodic release: whether to add job back to the queue
 
# Uncomment this line if you want the jobs automatically removed from the queue
# periodic_remove = ( (JobStatus =?= 5) && (HoldReasonCode =?= 3) && (HoldReasonSubCode =?= 2) )


# condor_submit_autokill 20 -append 'request_cpus=8' -append 'request_memory=128GB' -append 'request_disk=100GB' -append 'request_gpus=1' -append 'requirements = (TARGET.CUDAGlobalMemoryMb > 50000) && (Machine != "g101.internal.cluster.is.localnet")' -i
# condor_submit_autokill 20 -append 'request_cpus=16' -append 'request_memory=256GB' -append 'request_disk=100GB' -append 'request_gpus=2' -append 'requirements = (TARGET.CUDAGlobalMemoryMb > 50000) && (Machine != "g101.internal.cluster.is.localnet")' -i

# alias launch_condor_job=/home/mmordig/reinforcement/HumbleAttemptAtGeneralAI/geometry_translation/new/launch_condor_job_new.py
# launch_condor_job 30 --max_running_price -1 --- bash ~/reinforcement/HumbleAttemptAtGeneralAI/geometry_translation/new/run_sft.sh
# or
# bash ~/reinforcement/HumbleAttemptAtGeneralAI/geometry_translation/new/run_sft.sh


#tmux
source ~/.bashrc
pwd
set -eux
# module load cuda/12.1
# source /etc/profile.d/modules.sh # seems to be overwriting the path
# module load cuda/12.1 # seems to mess up the path
source ~/reinforcement/HumbleAttemptAtGeneralAI/geometry_translation/new/copy_hf_model_to_ram.sh models--meta-llama--Llama-2-7b-hf
source ~/reinforcement/HumbleAttemptAtGeneralAI/geometry_translation/new/verbalization_venv3/bin/activate
## move there, no locks supported, export HF_DATASETS_CACHE=/fast/mmordig/hf_cache/datasets
exp_type=
# exp_type=_small
python ~/reinforcement/HumbleAttemptAtGeneralAI/geometry_translation/new/trl_adaptations/sft_adapted.py \
  --overwrite_output_dir \
  --output_dir /fast/mmordig/general_ai_rl/alphageom_project/verbalization/training/exp${exp_type} \
  --dataset_name /fast/mmordig/general_ai_rl/alphageom_project/verbalization/datasets/alpha_geo${exp_type}_processed \
  --config ~/reinforcement/HumbleAttemptAtGeneralAI/geometry_translation/new/trl_sft_config.yml