# Distributed Setup With Accelerate

We show how to use accelerate in distributed mode (multiple nodes with multiple GPUs).

Using htcondor's `universe=parallel` to have all jobs start at the same time does not work (waiting forever) because the scheduler will only allocate it once all nodes are ready at the same time. So, it is better to do a busy wait adding one node at a time.
Therefore, we use the `queue` command and wait until all nodes are ready.

We provide a default accelerate config. You can overwrite it with your own config:
```bash
accelerate config --config_file LLM_finetuner/distributed_tests/distributed_example.yaml
```

## Automatic Setup
We launch n processes that are part of the same job via the `queue` command in the condor submission file. Then, we wait until all nodes are ready (to avoid the accelerate timeout). Finally, we run `accelerate launch` (wrapped in a script):
```{bash}
condor_submit_bid 50 ~/reinforcement/alphageometry/LLM_finetuner/distributed_tests/distributed_accelerate_example.sub
```

## Manual Setup
Here, you request the jobs interactively and wait until they are ready

Single node: we don't pass a machine rank:
```{bash}
accelerate launch --config_file LLM_finetuner/distributed_tests/distributed_example.yaml LLM_finetuner/distributed_tests/wait_for_everyone.py --num_machines 1
```

Distributed (multiple nodes):
```{bash}
https://github.com/huggingface/accelerate/blob/main/examples/slurm/submit_multinode.sh

# machine 1
condor_submit_bid 50 -append 'request_cpus=2' -append 'request_memory=16GB' -append 'request_gpus=1' -i
source ~/reinforcement/alphageometry/LLM_finetuner/verbalization_venv3/bin/activate
LLM_finetuner/distributed_tests/run_accelerate_simple.sh LLM_finetuner/distributed_tests/wait_for_everyone.py --machine_rank 0

# machine 2
condor_submit_bid 50 -append 'request_cpus=2' -append 'request_memory=16GB' -append 'request_gpus=1' -i
source ~/reinforcement/alphageometry/LLM_finetuner/verbalization_venv3/bin/activate
LLM_finetuner/distributed_tests/run_accelerate_simple.sh LLM_finetuner/distributed_tests/wait_for_everyone.py --machine_rank 1
```
