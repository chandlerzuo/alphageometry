# Fine-Tuning LLMs for Natural to Formal Translation

## Set up venv
```{bash}

cd ~/reinforcement/alphageometry/LLM_finetuner
python3.10 -m venv verbalization_venv3
source ~/reinforcement/alphageometry/LLM_finetuner/verbalization_venv3/bin/activate
# pip uninstall -y LLM_finetuner
pip install -e ~/reinforcement/alphageometry/
python -c "import LLM_finetuner; import LLM_finetuner.utils; LLM_finetuner.utils.subset_dataset"
# alternatively (then make sure to adapt your python path)
pip install --upgrade pip
pip install -r requirements.txt

wandb login
huggingface-cli login
```

Prepare with
```
mkdir -p /fast/mmordig/general_ai_rl/alphageom_project
ln -s /fast/mmordig/general_ai_rl/alphageom_project ~/reinforcement/alphageometry/LLM_finetuner/runs
ln -s /is/cluster/fast/mmordig/general_ai_rl/alphageom_project/ ~/reinforcement/alphageometry/LLM_finetuner/cluster_runs
mkdir -p ~/reinforcement/alphageometry/LLM_finetuner/runs/verbalization/{training,datasets,predictions}
export VERB_RUN_DIR=~/reinforcement/alphageometry/LLM_finetuner/runs/verbalization

# mikado
mkdir -p ~/reinforcement/alphageometry/LLM_finetuner/runs
mkdir -p ~/reinforcement/alphageometry/LLM_finetuner/runs/verbalization/{training,datasets,predictions}
ln -s alpha_geo_small alpha_geo
ln -s alpha_geo_small_processed alpha_geo_processed
export VERB_RUN_DIR=~/reinforcement/alphageometry/LLM_finetuner/runs/verbalization
```

For debugging:
```{bash}
wandb disable
wandb enabled
# whether to upload or not (but not disable)
wandb offline
wandb online
export WANDB_LOG_MODEL=True # upload model at end
export WANDB_WATCH=all # gradients (by default), all (gradients + parameters), false
export WANDB_PROJECT=verbalization
```

## Download model
Download the models with the huggingface CLI:
```{bash}
mkdir /fast/mmordig/hf_cache
cp -r ~/.cache/huggingface/ /fast/mmordig/hf_cache

# cannot download directly to this directory due to locking not available
# export HF_HOME=/fast/mmordig/hf_cache/huggingface
source ~/reinforcement/alphageometry/LLM_finetuner/verbalization_venv3/bin/activate
huggingface-cli download meta-llama/Meta-Llama-3-8B
```

## Dataset Preparation
```{bash}
cd $VERB_RUN_DIR/datasets
# scratch is very slow, https://atlas.is.localnet/confluence/display/IT/Cluster+storage
## to slow: rsync -avzh /is/cluster/scratch/pghosh/dataset/alpha_geo alpha_geo
## cp -r /is/cluster/scratch/pghosh/dataset/alpha_geo .
mkdir splits
find /is/cluster/scratch/pghosh/dataset/alpha_geo/ -type f -printf "%P\n" | split -l 1000 - splits/split_
for f in splits/split_*; do rsync -a --files-from="$f" /is/cluster/scratch/pghosh/dataset/alpha_geo/ alpha_geo/ & done; wait

# merge into fewer csvs
echo "dummy" > alpha_geo_header.csv
head -n 1 alpha_geo/nl_fl_dataset_1.csv >> alpha_geo_header.csv
cat alpha_geo_header.csv
## tail -q -n +2 alpha_geo_header.csv alpha_geo/nl_fl_dataset_{1..2000}.csv > alpha_geo_fewer/chunks1.csv

ls -1 alpha_geo | wc -l # --> 31952

mkdir alpha_geo_fewer_chunks
for i in {0..15}; do
    start=$((2000*$i+1))
    end=$((2000*($i+1)))
    tail -q -n +2 alpha_geo_header.csv alpha_geo/nl_fl_dataset_{$start..$end}.csv > alpha_geo_fewer_chunks/chunks$i.csv &
done

# on mikado, instead do
cd $VERB_RUN_DIR/datasets
cp -r /is/cluster-test/fast/mmordig/general_ai_rl/alphageom_project/verbalization/datasets/alpha_geo_small/ .
```

Preprocess the dataset into a HF dataset:
```{bash}
# condor_submit_autokill 25 -append 'request_cpus=20' -append 'request_memory=100GB' -append 'request_disk=100GB' -i

# tmux
source ~/reinforcement/alphageometry/LLM_finetuner/verbalization_venv3/bin/activate
python ~/reinforcement/alphageometry/LLM_finetuner/create_dataset.py \
    /fast/mmordig/general_ai_rl/alphageom_project/verbalization/datasets/alpha_geo_fewer_chunks \
    /fast/mmordig/general_ai_rl/alphageom_project/verbalization/datasets/alpha_geo_processed

# small dataset
source ~/reinforcement/alphageometry/LLM_finetuner/verbalization_venv3/bin/activate
python ~/reinforcement/alphageometry/LLM_finetuner/create_dataset.py \
    /fast/mmordig/general_ai_rl/alphageom_project/verbalization/datasets/alpha_geo_small \
    /fast/mmordig/general_ai_rl/alphageom_project/verbalization/datasets/alpha_geo_small_processed

# on mikado
source ~/reinforcement/alphageometry/LLM_finetuner/verbalization_venv3/bin/activate
python ~/reinforcement/alphageometry/LLM_finetuner/create_dataset.py \
    ~/reinforcement/alphageometry/LLM_finetuner/runs/verbalization/datasets/alpha_geo_small \
    ~/reinforcement/alphageometry/LLM_finetuner/runs/verbalization/datasets/alpha_geo_small_processed
```

New:
```
python ~/reinforcement/alphageometry/LLM_finetuner/convert_dataset.py \
    ~/reinforcement/alphageometry/LLM_finetuner/runs/verbalization/datasets/alpha_geo_small \
    ~/reinforcement/alphageometry/LLM_finetuner/runs/verbalization/datasets/alpha_geo_small_arrow

ln -s ~/reinforcement/alphageometry/LLM_finetuner/runs/verbalization/datasets/alpha_geo_small_arrow ~/reinforcement/alphageometry/LLM_finetuner/runs/verbalization/datasets/alpha_geo_arrow
```

## Fine-Tuning
Train the model on GPUs:

```{bash}

# condor_submit_autokill 20 -append 'request_cpus=8' -append 'request_memory=128GB' -append 'request_disk=100GB' -append 'request_gpus=1' -append 'requirements = (TARGET.CUDAGlobalMemoryMb > 50000) && (Machine != "g101.internal.cluster.is.localnet")' -i
# condor_submit_autokill 20 -append 'request_cpus=16' -append 'request_memory=256GB' -append 'request_disk=100GB' -append 'request_gpus=2' -append 'requirements = (TARGET.CUDAGlobalMemoryMb > 50000) && (Machine != "g101.internal.cluster.is.localnet")' -i
tmux
# wandb enabled
# wandb online

ls ~/.cache/huggingface/hub/
source ~/reinforcement/alphageometry/LLM_finetuner/copy_hf_model_to_ram.sh models--meta-llama--Llama-2-7b-hf
source ~/reinforcement/alphageometry/LLM_finetuner/copy_hf_model_to_ram.sh models--meta-llama--Llama-2-7b-chat-hf
source ~/reinforcement/alphageometry/LLM_finetuner/copy_hf_model_to_ram.sh models--gpt2

# source ~/reinforcement/alphageometry/LLM_finetuner/setup_commands.sh
accelerate config

# python \
source ~/reinforcement/alphageometry/LLM_finetuner/verbalization_venv3/bin/activate
function accelerate_cmd() {
    accelerate launch --config_file ~/reinforcement/alphageometry/LLM_finetuner/example_accelerate_config.yaml --multi_gpu --num_processes "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" "$@"
}


source ~/reinforcement/alphageometry/LLM_finetuner/verbalization_venv3/bin/activate
python ~/reinforcement/alphageometry/LLM_finetuner/make_model_predictions.py \
    --dataset_name ~/reinforcement/alphageometry/LLM_finetuner/runs/verbalization/datasets/alpha_geo_processed \
    --dataset_test_name test \
    --max_predict_samples 5 \
    --model_name_or_path ~/reinforcement/alphageometry/LLM_finetuner/runs/verbalization/training/run_09052024/Llama-2-7b-chat-hf_-1ex_peftTrue \
    --out_filename ~/reinforcement/alphageometry/LLM_finetuner/runs/verbalization/predictions/run_09052024_{model_name}_{max_predict_samples}samples.txt \
    --max_new_tokens 70
    


source ~/reinforcement/alphageometry/LLM_finetuner/verbalization_venv3/bin/activate
python ~/reinforcement/alphageometry/LLM_finetuner/make_model_predictions.py \
    --dataset_name ~/reinforcement/alphageometry/LLM_finetuner/runs/verbalization/datasets/alpha_geo_processed \
    --dataset_test_name train \
    --max_predict_samples 1 \
    --model_name_or_path ~/reinforcement/alphageometry/LLM_finetuner/runs/verbalization/training/run_09052024/gpt2_-1ex_peftTrue/ \
    --out_filename ~/reinforcement/alphageometry/LLM_finetuner/runs/verbalization/predictions/run_09052024_{model_name}_{max_predict_samples}samples.txt \
    --max_new_tokens 70

#--use_peft \

#source ~/reinforcement/alphageometry/LLM_finetuner/verbalization_venv3/bin/activate
source ~/reinforcement/alphageometry/LLM_finetuner/setup_env.sh
num_train_samples=10
run_dir_name=debug11
python ~/reinforcement/alphageometry/LLM_finetuner/sft_finetuning.py \
  --overwrite_output_dir \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --eval_steps 10 \
  --model_name_or_path meta-llama/Llama-2-7b-chat-hf \
  --max_eval_samples 400 \
  --explicit_eos_str '[END]' \
  --extra_tokens_file ~/reinforcement/alphageometry/assets/def-patterns-desc.yml \
  --output_dir ~/reinforcement/alphageometry/LLM_finetuner/runs/verbalization/training/${run_dir_name}/{model_name}_{max_train_samples}ex_peft{use_peft} \
  --dataset_name ~/reinforcement/alphageometry/LLM_finetuner/runs/verbalization/datasets/alpha_geo_processed \
  --config ~/reinforcement/alphageometry/LLM_finetuner/trl_sft_config.yml \
  --max_train_samples "$num_train_samples" \
  --num_train_epochs 100000




source ~/reinforcement/alphageometry/LLM_finetuner/verbalization_venv3/bin/activate
num_train_samples=10
run_dir_name=debug112
python ~/reinforcement/alphageometry/LLM_finetuner/sft_finetuning.py \
  --overwrite_output_dir \
  --use_peft \
  --per_device_train_batch_size 64 \
  --per_device_eval_batch_size 64 \
  --model_name_or_path gpt2 \
  --eval_steps 10 \
  --evaluation_strategy steps \
  --max_eval_samples 400 \
  --explicit_eos_str '[END]' \
  --extra_tokens_file ~/reinforcement/alphageometry/assets/def-patterns-desc.yml \
  --output_dir ~/reinforcement/alphageometry/LLM_finetuner/runs/verbalization/training/${run_dir_name}/{model_name}_{max_train_samples}ex_peft{use_peft} \
  --dataset_name ~/reinforcement/alphageometry/LLM_finetuner/runs/verbalization/datasets/alpha_geo_arrow \
  --config ~/reinforcement/alphageometry/LLM_finetuner/trl_sft_config.yml \
  --max_train_samples "$num_train_samples" \
  --num_train_epochs 100000

  



accelerate_cmd \
  ~/reinforcement/alphageometry/LLM_finetuner/sft_finetuning.py \
  --overwrite_output_dir \
  --model_name_or_path gpt2 \
  --max_eval_samples 400 \
  --output_dir ~/reinforcement/alphageometry/LLM_finetuner/runs/verbalization/training/overfit_single_nocompl/{model_name}_{max_train_samples}ex \
  --dataset_name ~/reinforcement/alphageometry/LLM_finetuner/runs/verbalization/datasets/alpha_geo_processed \
  --config ~/reinforcement/alphageometry/LLM_finetuner/trl_sft_config.yml \
  --max_train_samples 1000 \
  --num_train_epochs 800
# accelerate_cmd \


python ~/reinforcement/alphageometry/LLM_finetuner/sft_finetuning.py \
  --overwrite_output_dir \
  --model_name_or_path gpt2 \
  --use_peft=True \
  --max_eval_samples 400 \
  --output_dir ~/reinforcement/alphageometry/LLM_finetuner/runs/verbalization/training/overfit_single_nocompl/{model_name}_{max_train_samples}ex_peft{use_peft} \
  --dataset_name ~/reinforcement/alphageometry/LLM_finetuner/runs/verbalization/datasets/alpha_geo_processed \
  --config ~/reinforcement/alphageometry/LLM_finetuner/trl_sft_config.yml \
  --max_train_samples 200 \
  --num_train_epochs 100000

source ~/reinforcement/alphageometry/LLM_finetuner/verbalization_venv3/bin/activate
python ~/reinforcement/alphageometry/LLM_finetuner/make_model_predictions.py \
    --dataset_name ~/reinforcement/alphageometry/LLM_finetuner/runs/verbalization/datasets/alpha_geo_processed \
    --dataset_test_name train \
    --max_predict_samples 1 \
    --model_name_or_path ~/reinforcement/alphageometry/LLM_finetuner/runs/verbalization/training/overfit_single_nocompl/gpt2_2ex \
    --out_filename ~/reinforcement/alphageometry/LLM_finetuner/runs/verbalization/predictions/overfitsingle_{model_name}_{max_predict_samples}samples.txt \
    --max_new_tokens 70

source ~/reinforcement/alphageometry/LLM_finetuner/verbalization_venv3/bin/activate
python ~/reinforcement/alphageometry/LLM_finetuner/make_model_predictions.py \
    --dataset_name ~/reinforcement/alphageometry/LLM_finetuner/runs/verbalization/datasets/alpha_geo_processed \
    --dataset_test_name train \
    --max_predict_samples 1 \
    --model_name_or_path ~/reinforcement/alphageometry/LLM_finetuner/runs/verbalization/training/run_02052024/Llama-2-7b-chat-hf_1000ex_peftTrue \
    --out_filename ~/reinforcement/alphageometry/LLM_finetuner/runs/verbalization/predictions/overfitsingle_{model_name}_{max_predict_samples}samples.txt \
    --max_new_tokens 70
```

Testing:
```{bash}
source ~/reinforcement/alphageometry/LLM_finetuner/verbalization_venv3/bin/activate
python ~/reinforcement/alphageometry/LLM_finetuner/sft_finetuning.py \
  --overwrite_output_dir \
  --model_name_or_path gpt2 \
  --per_device_eval_batch_size 32 \
  --per_device_train_batch_size 32 \
  --use_peft=True \
  --explicit_eos_str '[END]' \
  --extra_tokens_file ~/reinforcement/alphageometry/assets/def-patterns-desc.yml \
  --max_eval_samples 400 \
  --output_dir ~/reinforcement/alphageometry/LLM_finetuner/runs/verbalization/training/overfit_single_nocompl/{model_name}_{max_train_samples}ex_peft{use_peft}_extratokens_eos \
  --dataset_name ~/reinforcement/alphageometry/LLM_finetuner/runs/verbalization/datasets/alpha_geo_processed \
  --config ~/reinforcement/alphageometry/LLM_finetuner/trl_sft_config.yml \
  --max_train_samples 32 \
  --num_train_epochs 10000
```




Old:
```{bash}
# condor_submit_autokill 20 -append 'request_cpus=8' -append 'request_memory=128GB' -append 'request_disk=100GB' -append 'request_gpus=1' -append 'requirements = (TARGET.CUDAGlobalMemoryMb > 50000) && (Machine != "g101.internal.cluster.is.localnet")' -i
# condor_submit_autokill 20 -append 'request_cpus=16' -append 'request_memory=256GB' -append 'request_disk=100GB' -append 'request_gpus=2' -append 'requirements = (TARGET.CUDAGlobalMemoryMb > 50000) && (Machine != "g101.internal.cluster.is.localnet")' -i
tmux
# wandb enabled
# wandb online
# rm -rf /fast/mmordig/general_ai_rl/alphageom_project/verbalization/training/exp_small/Llama-2-7b-hf
#module load cuda/12.1
source ~/reinforcement/alphageometry/LLM_finetuner/copy_hf_model_to_ram.sh models--meta-llama--Llama-2-7b-hf
source ~/reinforcement/alphageometry/LLM_finetuner/verbalization_venv3/bin/activate
## move there, no locks supported, export HF_DATASETS_CACHE=/fast/mmordig/hf_cache/datasets
exp_type=_small
#exp_type=
python ~/reinforcement/alphageometry/LLM_finetuner/sft_finetuning.py \
  --overwrite_output_dir \
  --output_dir /fast/mmordig/general_ai_rl/alphageom_project/verbalization/training/exp${exp_type}/{model_name} \
  --dataset_name /fast/mmordig/general_ai_rl/alphageom_project/verbalization/datasets/alpha_geo${exp_type}_processed \
  --config ~/reinforcement/alphageometry/LLM_finetuner/trl_sft_config.yml \
  --dataloader_num_workers 0

# mikado
source ~/reinforcement/alphageometry/LLM_finetuner/verbalization_venv3/bin/activate
exp_type=_small
python ~/reinforcement/alphageometry/LLM_finetuner/sft_finetuning.py \
  --overwrite_output_dir \
  --model_name_or_path meta-llama/Llama-2-7b-chat-hf \
  --output_dir ~/reinforcement/alphageometry/LLM_finetuner/runs/verbalization/training/exp${exp_type}/{model_name} \
  --dataset_name ~/reinforcement/alphageometry/LLM_finetuner/runs/verbalization/datasets/alpha_geo${exp_type}_processed \
  --config ~/reinforcement/alphageometry/LLM_finetuner/trl_sft_config.yml \
  
source ~/reinforcement/alphageometry/LLM_finetuner/copy_hf_model_to_ram.sh models--meta-llama--Llama-2-7b-chat-hf
  --model_name_or_path gpt2 \

# overfit on one sample on mikado
# rm -rf ~/reinforcement/alphageometry/LLM_finetuner/runs/verbalization/training/overfit_single
source ~/reinforcement/alphageometry/LLM_finetuner/verbalization_venv3/bin/activate
exp_type=_small
python ~/reinforcement/alphageometry/LLM_finetuner/sft_finetuning.py \
  --overwrite_output_dir \
  --model_name_or_path meta-llama/Llama-2-7b-hf \
  --output_dir ~/reinforcement/alphageometry/LLM_finetuner/runs/verbalization/training/overfit_single_nocompl/{model_name}_1ex \
  --max_eval_samples 1 \
  --dataset_name ~/reinforcement/alphageometry/LLM_finetuner/runs/verbalization/datasets/alpha_geo${exp_type}_processed \
  --config ~/reinforcement/alphageometry/LLM_finetuner/trl_sft_config.yml \
  --max_train_samples 1 \
  --num_train_epochs 150


# accelerate
accelerate config
accelerate launch ~/reinforcement/alphageometry/LLM_finetuner/sft_finetuning.py \
  --overwrite_output_dir \
  --model_name_or_path gpt2 \
  --max_eval_samples 1 \
  --use_peft=True \
  --output_dir ~/reinforcement/alphageometry/LLM_finetuner/runs/verbalization/training/overfit_single_nocompl/{model_name}_withpeft \
  --dataset_name ~/reinforcement/alphageometry/LLM_finetuner/runs/verbalization/datasets/alpha_geo${exp_type}_processed \
  --config ~/reinforcement/alphageometry/LLM_finetuner/trl_sft_config.yml \
  --max_train_samples 200 \
  --num_train_epochs 1000



# with peft
python ~/reinforcement/alphageometry/LLM_finetuner/sft_finetuning.py \
  --overwrite_output_dir \
  --model_name_or_path gpt2 \
  --max_eval_samples 1 \
  --use_peft=True \
  --output_dir ~/reinforcement/alphageometry/LLM_finetuner/runs/verbalization/training/overfit_single_nocompl/{model_name}_withpeft \
  --dataset_name ~/reinforcement/alphageometry/LLM_finetuner/runs/verbalization/datasets/alpha_geo${exp_type}_processed \
  --config ~/reinforcement/alphageometry/LLM_finetuner/trl_sft_config.yml \
  --max_train_samples 1 \
  --num_train_epochs 1000

python ~/reinforcement/alphageometry/LLM_finetuner/sft_finetuning.py \
  --overwrite_output_dir \
  --model_name_or_path gpt2 \
  --max_eval_samples 400 \
  --output_dir ~/reinforcement/alphageometry/LLM_finetuner/runs/verbalization/training/overfit_single_nocompl/{model_name}_{max_train_samples}ex \
  --dataset_name ~/reinforcement/alphageometry/LLM_finetuner/runs/verbalization/datasets/alpha_geo${exp_type}_processed \
  --config ~/reinforcement/alphageometry/LLM_finetuner/trl_sft_config.yml \
  --max_train_samples 2 \
  --num_train_epochs 800

# works, does not properly end with 2 samples, but otherwise fine
python ~/reinforcement/alphageometry/LLM_finetuner/sft_finetuning.py \
  --overwrite_output_dir \
  --model_name_or_path gpt2 \
  --max_eval_samples 1 \
  --output_dir ~/reinforcement/alphageometry/LLM_finetuner/runs/verbalization/training/overfit_single_nocompl/{model_name} \
  --dataset_name ~/reinforcement/alphageometry/LLM_finetuner/runs/verbalization/datasets/alpha_geo${exp_type}_processed \
  --config ~/reinforcement/alphageometry/LLM_finetuner/trl_sft_config.yml \
  --max_train_samples 1 \
  --num_train_epochs 150
  
  
  --model_name_or_path gpt2 \

python ~/reinforcement/alphageometry/LLM_finetuner/make_model_predictions.py \
    --dataset_name ~/reinforcement/alphageometry/LLM_finetuner/runs/verbalization/datasets/alpha_geo_processed \
    --dataset_test_name train \
    --max_predict_samples 1 \
    --model_name_or_path ~/reinforcement/alphageometry/LLM_finetuner/runs/verbalization/training/overfit_single_nocompl/gpt2 \
    --out_filename ~/reinforcement/alphageometry/LLM_finetuner/runs/verbalization/predictions/overfitsingle_{model_name}_{max_predict_samples}samples.txt \
    --max_new_tokens 70
    
--model_name_or_path ~/reinforcement/alphageometry/LLM_finetuner/runs/verbalization/training/overfit_single_nocompl/Llama-2-7b-chat-hf/ \
--model_name_or_path ~/reinforcement/alphageometry/LLM_finetuner/runs/verbalization/training/overfit_single/Llama-2-7b-hf \
  --model_name_or_path meta-llama/Llama-2-7b-chat-hf \
```

Multi-GPU:
```{bash}

todo:

accelerate config
accelerate test
accelerate launch <script_and_args>
```

## Generation
Generate predictions:
```{bash}
source ~/reinforcement/alphageometry/LLM_finetuner/verbalization_venv3/bin/activate
exp_type=_small
model=gpt2
python ~/reinforcement/alphageometry/LLM_finetuner/make_model_predictions.py \
    ~/reinforcement/alphageometry/LLM_finetuner/runs/verbalization/training/exp${exp_type}/${model} \
    ~/reinforcement/alphageometry/LLM_finetuner/runs/verbalization/datasets/alpha_geo${exp_type}_processed \
    ~/reinforcement/alphageometry/LLM_finetuner/runs/verbalization/predictions/exp${exp_type}/${model}_predictions.txt 5
```

Deploy the model (note: different generation params than for prediction):
```{bash}
source ~/reinforcement/alphageometry/LLM_finetuner/verbalization_venv3/bin/activate
exp_type=_small
model=gpt2
python ~/reinforcement/alphageometry/LLM_finetuner/deploy_model.py \
    ~/reinforcement/alphageometry/LLM_finetuner/runs/verbalization/training/exp${exp_type}/${model}
```


## Notes
Do not use `is/scratch` (HDD and full), but `/fast` (SSD).

Not getting faster with batch size larger than 32 (I think)
Using packing
how to copy big models (using home is not very convenient)

LLMs don't really have a context length, they just don't work very well for out-of-distribution, so the context length is the typical length found in the training data

seeding: setting `seed, data_seed`, does this prevent duplicate data?

```{bash}
MultiGPU training:

The input hidden states seems to be silently casted in float32, this might be related to the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in torch.bfloat16.
/lustre/home/mmordig/reinforcement/alphageometry/LLM_finetuner/verbalization_venv3/lib/python3.10/site-packages/torch/nn/parallel/_functions.py:68: UserWarning: Was asked to gather along dimension 0, but all input tensors were scalars; will instead unsqueeze and return a vector.
```

tokenize on the fly or in advance?

huggingface 401: credential helper not running on machine

todo: use mlm and DataCollatorForCompletionOnlyLM, their mlm argument does nothing

`$$([NumJobStarts])` wildcard value not filled

module load not working

https://www.philschmid.de/getting-started-pytorch-2-0-transformers: chooses eval batch size smaller than train batch size, why? has the eval dataset larger length?

Current warnings:
happens with gpt2: 
```{bash}
To disable this warning, you can either:
        - Avoid using `tokenizers` before the fork if possible
        - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
The input hidden states seems to be silently casted in float32, this might be related to the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in torch.bfloat16.
```

## ToDo

```
from accelerate import PartialState
device_string = PartialState().process_index
model = AutoModelForCausalLM.from_pretrained(
     ...
    device_map={'':device_string}
)
```

Additionally, in the SFTTrainer, we support pre-tokenized datasets if they are datasets.Dataset or datasets.IterableDataset. In other words, if such a dataset has a column of input_ids, no further processing (tokenization or packing) will be done, and the dataset will be used as-is. This can be useful if you have pretokenized your dataset outside of this script and want to re-use it directly.

condor_free