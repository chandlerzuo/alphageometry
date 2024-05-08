#!/usr/bin/env bash

# choose parallel universe to make sure jobs start at the same time
##CONDOR universe=parallel
##CONDOR machine_count = 2
# add -$(Node) to log, otuput
##CONDOR Executable = /bin/cat
##CONDOR Arguments = Num nodes: $(Node)

##CONDOR request_cpus=1
##CONDOR request_memory=16000
##CONDOR request_disk=1G
##CONDOR +JobBatchName = "test_distributed"
##CONDOR log = /home/mmordig/joblogs/job-$(ClusterId)-$(ProcId)-$(Node).$$([NumJobStarts]).log
##CONDOR output = /home/mmordig/joblogs/job-$(ClusterId)-$(ProcId)-$(Node).$$([NumJobStarts]).out
##CONDOR error = /home/mmordig/joblogs/job-$(ClusterId)-$(ProcId)-$(Node).$$([NumJobStarts]).err
##CONDOR request_gpus=1
##CONDOR requirements = (TARGET.CUDAGlobalMemoryMb > 50000) && (Machine != "g101.internal.cluster.is.localnet")

alias launch_condor_job="~/reinforcement/alphageometry/LLM_finetuner/launch_condor_job_new.py 50 --max_running_price -1 ---" #--dry 
launch_condor_job /home/mmordig/reinforcement/alphageometry/LLM_finetuner/tests/test_distributed.sh

source ~/.bashrc
pwd
set -eux

source ~/reinforcement/alphageometry/LLM_finetuner/verbalization_venv3/bin/activate