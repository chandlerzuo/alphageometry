#!/usr/bin/env bash

# script to launch multi-node jobs and wait until all jobs are ready

set -eux

##CONDOR request_cpus=1
##CONDOR request_memory=3000
##CONDOR request_disk=1G
##CONDOR +JobBatchName = "test_distributed"
##CONDOR log = /home/mmordig/joblogs/job-$(ClusterId)-$(ProcId)-$(Node).$$([NumJobStarts]).log
##CONDOR output = /home/mmordig/joblogs/job-$(ClusterId)-$(ProcId)-$(Node).$$([NumJobStarts]).out
##CONDOR error = /home/mmordig/joblogs/job-$(ClusterId)-$(ProcId)-$(Node).$$([NumJobStarts]).err
##-CONDOR request_gpus=1
##-CONDOR requirements = (TARGET.CUDAGlobalMemoryMb > 50000) && (Machine != "g101.internal.cluster.is.localnet")
##CONDOR NumRanks = 2
##CONDOR +WantIOProxy = true
##CONDOR $(Cluster) $(Process) $(NumRanks)
##CONDOR queue $(NumRanks)

# inspired by mpi script: see directory /usr/local/share/htcondor_mpi/, https://atlas.is.localnet/confluence/display/IT/How+to+run+MPI+programs+on+the+cluster
# requires:
# +WantIOProxy = true
# case-sensitive: don't use (can't chirp error): +want_io_proxy = true

# alias launch_condor_job="~/reinforcement/alphageometry/LLM_finetuner/launch_condor_job_new.py \
#     50 --max_running_price -1 ---" #--dry 
# launch_condor_job ~/reinforcement/alphageometry/LLM_finetuner/distributed_tests/condor_wait_for_others.sh

export _CONDOR_CLUSTERNO=$1
shift
export _CONDOR_PROCNO=$1
shift
export _CONDOR_NPROCS=$1
shift

echo "Starting job $_CONDOR_PROCNO / $_CONDOR_NPROCS on $(hostname)"

export _CONDOR_REMOTE_SPOOL_DIR=/scratch/rendezvous/$_CONDOR_CLUSTERNO
mkdir -p "$_CONDOR_REMOTE_SPOOL_DIR"

CONDOR_CHIRP=$(condor_config_val libexec)
CONDOR_CHIRP=$CONDOR_CHIRP/condor_chirp

ip_address=$(hostname -I | awk '{print $1}')
echo "Job $_CONDOR_PROCNO: IP address: $ip_address"

echo "Job $_CONDOR_PROCNO: Writing ip address to contact file"
thisrun=$($CONDOR_CHIRP get_job_attr EnteredCurrentStatus)
# append to file (in a thread-safe fashion?)
echo "$_CONDOR_PROCNO $ip_address $(hostname) $thisrun" |
	$CONDOR_CHIRP put -mode cwa - "$_CONDOR_REMOTE_SPOOL_DIR"/contact
_TEST=$?
if [ $_TEST -ne 0 ]
then
	echo error $_TEST chirp putting contact info back to submit machine
	exit 255
fi

echo "Job $_CONDOR_PROCNO: Waiting for all jobs to start"
done=0
num_attempts=0
local_contact_file="$_CONDOR_SCRATCH_DIR"/contact

while [ $done -eq 0 ]; do
    $CONDOR_CHIRP fetch "$_CONDOR_REMOTE_SPOOL_DIR"/contact "$local_contact_file"
    lines=$(wc -l "$local_contact_file" | awk '{print $1}')
    if [ "$lines" -eq "$_CONDOR_NPROCS" ]; then
        done=1
    else
        num_attempts=$((num_attempts+1))
        if [ $num_attempts -gt 1000 ]; then
            echo "Job $_CONDOR_PROCNO: Timeout waiting for all jobs to start"
            exit 1
        fi
        sleep 1
    fi
done

# print time
echo "Job $_CONDOR_PROCNO: All jobs have started at $(date)"

echo "Job $_CONDOR_PROCNO: Getting master ip from contact file: "
cat "$local_contact_file"

# get master ip identified by proc 0
# look at first column: if equal to 0, return second column
master_ip=$(awk '$1 == 0 {print $2}' "$local_contact_file")
echo "Job $_CONDOR_PROCNO: Master IP: $master_ip"

export MACHINE_RANK=$_CONDOR_PROCNO
export MASTER_IP=$master_ip
export NUM_MACHINES=$_CONDOR_NPROCS
exec "$@"