# Fine-Tuning LLMs for Natural to Formal Translation

## Set up venv
```{bash}
cd ~/reinforcement/HumbleAttemptAtGeneralAI/geometry_translation/new
python3.10 -m venv verbalization_venv3
source ~/reinforcement/alphageometry/LLM_finetuner/verbalization_venv3/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

wandb login
huggingface-cli login
```

Prepare with
```
mkdir -p /fast/mmordig/general_ai_rl/alphageom_project/verbalization/{training,datasets,predictions}
export VERB_RUN_DIR=/fast/mmordig/general_ai_rl/alphageom_project/verbalization

# mikado
mkdir -p /home/mmordig/reinforcement/HumbleAttemptAtGeneralAI/runs/verbalization/{training,datasets,predictions}
export VERB_RUN_DIR=/home/mmordig/reinforcement/HumbleAttemptAtGeneralAI/runs/verbalization
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
source ~/reinforcement/HumbleAttemptAtGeneralAI/geometry_translation/new/verbalization_venv3/bin/activate
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
source ~/reinforcement/HumbleAttemptAtGeneralAI/geometry_translation/new/verbalization_venv3/bin/activate
python ~/reinforcement/HumbleAttemptAtGeneralAI/geometry_translation/new/create_dataset.py \
    /fast/mmordig/general_ai_rl/alphageom_project/verbalization/datasets/alpha_geo_fewer_chunks \
    /fast/mmordig/general_ai_rl/alphageom_project/verbalization/datasets/alpha_geo_processed

# small dataset
source ~/reinforcement/HumbleAttemptAtGeneralAI/geometry_translation/new/verbalization_venv3/bin/activate
python ~/reinforcement/HumbleAttemptAtGeneralAI/geometry_translation/new/create_dataset.py \
    /fast/mmordig/general_ai_rl/alphageom_project/verbalization/datasets/alpha_geo_small \
    /fast/mmordig/general_ai_rl/alphageom_project/verbalization/datasets/alpha_geo_small_processed

# on mikado
source ~/reinforcement/HumbleAttemptAtGeneralAI/geometry_translation/new/verbalization_venv3/bin/activate
python ~/reinforcement/HumbleAttemptAtGeneralAI/geometry_translation/new/create_dataset.py \
    /home/mmordig/reinforcement/HumbleAttemptAtGeneralAI/runs/verbalization/datasets/alpha_geo_small \
    /home/mmordig/reinforcement/HumbleAttemptAtGeneralAI/runs/verbalization/datasets/alpha_geo_small_processed
```

## Fine-Tuning
Train the model on GPUs:
```{bash}
# condor_submit_autokill 20 -append 'request_cpus=8' -append 'request_memory=128GB' -append 'request_disk=100GB' -append 'request_gpus=1' -append 'requirements = (TARGET.CUDAGlobalMemoryMb > 50000) && (Machine != "g101.internal.cluster.is.localnet")' -i
# condor_submit_autokill 20 -append 'request_cpus=16' -append 'request_memory=256GB' -append 'request_disk=100GB' -append 'request_gpus=2' -append 'requirements = (TARGET.CUDAGlobalMemoryMb > 50000) && (Machine != "g101.internal.cluster.is.localnet")' -i
tmux
# wandb enabled
# wandb online
# rm -rf /fast/mmordig/general_ai_rl/alphageom_project/verbalization/training/exp_small/Llama-2-7b-hf
#module load cuda/12.1
source ~/reinforcement/HumbleAttemptAtGeneralAI/geometry_translation/new/copy_hf_model_to_ram.sh models--meta-llama--Llama-2-7b-hf
source ~/reinforcement/HumbleAttemptAtGeneralAI/geometry_translation/new/verbalization_venv3/bin/activate
## move there, no locks supported, export HF_DATASETS_CACHE=/fast/mmordig/hf_cache/datasets
exp_type=_small
#exp_type=
python ~/reinforcement/HumbleAttemptAtGeneralAI/geometry_translation/new/trl_adaptations/sft_adapted.py \
  --overwrite_output_dir \
  --output_dir /fast/mmordig/general_ai_rl/alphageom_project/verbalization/training/exp${exp_type}/{model_name} \
  --dataset_name /fast/mmordig/general_ai_rl/alphageom_project/verbalization/datasets/alpha_geo${exp_type}_processed \
  --config ~/reinforcement/HumbleAttemptAtGeneralAI/geometry_translation/new/trl_sft_config.yml \
  --dataloader_num_workers 0

# mikado
source ~/reinforcement/HumbleAttemptAtGeneralAI/geometry_translation/new/verbalization_venv3/bin/activate
exp_type=_small
python ~/reinforcement/HumbleAttemptAtGeneralAI/geometry_translation/new/trl_adaptations/sft_adapted.py \
  --overwrite_output_dir \
  --model_name_or_path meta-llama/Llama-2-7b-chat-hf \
  --output_dir /home/mmordig/reinforcement/HumbleAttemptAtGeneralAI/runs/verbalization/training/exp${exp_type}/{model_name} \
  --dataset_name /home/mmordig/reinforcement/HumbleAttemptAtGeneralAI/runs/verbalization/datasets/alpha_geo${exp_type}_processed \
  --config ~/reinforcement/HumbleAttemptAtGeneralAI/geometry_translation/new/trl_sft_config.yml \
  
source ~/reinforcement/HumbleAttemptAtGeneralAI/geometry_translation/new/copy_hf_model_to_ram.sh models--meta-llama--Llama-2-7b-chat-hf
  --model_name_or_path gpt2 \
```
```{bash}


python ~/reinforcement/HumbleAttemptAtGeneralAI/geometry_translation/new/trl_adaptations/sft_adapted.py \
  --overwrite_output_dir \
  --model_name_or_path gpt2 \
  --max_eval_samples 400 \
  --output_dir /home/mmordig/reinforcement/HumbleAttemptAtGeneralAI/runs/verbalization/training/overfit_single_nocompl/{model_name}_{max_train_samples}ex \
  --dataset_name /home/mmordig/reinforcement/HumbleAttemptAtGeneralAI/runs/verbalization/datasets/alpha_geo${exp_type}_processed \
  --config ~/reinforcement/HumbleAttemptAtGeneralAI/geometry_translation/new/trl_sft_config.yml \
  --max_train_samples 2 \
  --num_train_epochs 800


ln -s /fast/mmordig/general_ai_rl/alphageom_project ~/reinforcement/HumbleAttemptAtGeneralAI/runs

python ~/reinforcement/HumbleAttemptAtGeneralAI/geometry_translation/new/trl_adaptations/sft_adapted.py \
  --overwrite_output_dir \
  --model_name_or_path gpt2 \
  --use_peft=True \
  --max_eval_samples 400 \
  --output_dir /home/mmordig/reinforcement/HumbleAttemptAtGeneralAI/runs/verbalization/training/overfit_single_nocompl/{model_name}_{max_train_samples}ex_peft{use_peft} \
  --dataset_name /home/mmordig/reinforcement/HumbleAttemptAtGeneralAI/runs/verbalization/datasets/alpha_geo_processed \
  --config ~/reinforcement/HumbleAttemptAtGeneralAI/geometry_translation/new/trl_sft_config.yml \
  --max_train_samples 200 \
  --num_train_epochs 100000



# overfit on one sample on mikado
# rm -rf /home/mmordig/reinforcement/HumbleAttemptAtGeneralAI/runs/verbalization/training/overfit_single
source ~/reinforcement/HumbleAttemptAtGeneralAI/geometry_translation/new/verbalization_venv3/bin/activate
exp_type=_small
python ~/reinforcement/HumbleAttemptAtGeneralAI/geometry_translation/new/trl_adaptations/sft_adapted.py \
  --overwrite_output_dir \
  --model_name_or_path meta-llama/Llama-2-7b-hf \
  --output_dir /home/mmordig/reinforcement/HumbleAttemptAtGeneralAI/runs/verbalization/training/overfit_single_nocompl/{model_name}_1ex \
  --max_eval_samples 1 \
  --dataset_name /home/mmordig/reinforcement/HumbleAttemptAtGeneralAI/runs/verbalization/datasets/alpha_geo${exp_type}_processed \
  --config ~/reinforcement/HumbleAttemptAtGeneralAI/geometry_translation/new/trl_sft_config.yml \
  --max_train_samples 1 \
  --num_train_epochs 150


# accelerate
accelerate config
accelerate launch ~/reinforcement/HumbleAttemptAtGeneralAI/geometry_translation/new/trl_adaptations/sft_adapted.py \
  --overwrite_output_dir \
  --model_name_or_path gpt2 \
  --max_eval_samples 1 \
  --use_peft=True \
  --output_dir /home/mmordig/reinforcement/HumbleAttemptAtGeneralAI/runs/verbalization/training/overfit_single_nocompl/{model_name}_withpeft \
  --dataset_name /home/mmordig/reinforcement/HumbleAttemptAtGeneralAI/runs/verbalization/datasets/alpha_geo${exp_type}_processed \
  --config ~/reinforcement/HumbleAttemptAtGeneralAI/geometry_translation/new/trl_sft_config.yml \
  --max_train_samples 200 \
  --num_train_epochs 1000



# with peft
python ~/reinforcement/HumbleAttemptAtGeneralAI/geometry_translation/new/trl_adaptations/sft_adapted.py \
  --overwrite_output_dir \
  --model_name_or_path gpt2 \
  --max_eval_samples 1 \
  --use_peft=True \
  --output_dir /home/mmordig/reinforcement/HumbleAttemptAtGeneralAI/runs/verbalization/training/overfit_single_nocompl/{model_name}_withpeft \
  --dataset_name /home/mmordig/reinforcement/HumbleAttemptAtGeneralAI/runs/verbalization/datasets/alpha_geo${exp_type}_processed \
  --config ~/reinforcement/HumbleAttemptAtGeneralAI/geometry_translation/new/trl_sft_config.yml \
  --max_train_samples 1 \
  --num_train_epochs 1000

python ~/reinforcement/HumbleAttemptAtGeneralAI/geometry_translation/new/trl_adaptations/sft_adapted.py \
  --overwrite_output_dir \
  --model_name_or_path gpt2 \
  --max_eval_samples 400 \
  --output_dir /home/mmordig/reinforcement/HumbleAttemptAtGeneralAI/runs/verbalization/training/overfit_single_nocompl/{model_name}_{max_train_samples}ex \
  --dataset_name /home/mmordig/reinforcement/HumbleAttemptAtGeneralAI/runs/verbalization/datasets/alpha_geo${exp_type}_processed \
  --config ~/reinforcement/HumbleAttemptAtGeneralAI/geometry_translation/new/trl_sft_config.yml \
  --max_train_samples 2 \
  --num_train_epochs 800

# works, does not properly end with 2 samples, but otherwise fine
python ~/reinforcement/HumbleAttemptAtGeneralAI/geometry_translation/new/trl_adaptations/sft_adapted.py \
  --overwrite_output_dir \
  --model_name_or_path gpt2 \
  --max_eval_samples 1 \
  --output_dir /home/mmordig/reinforcement/HumbleAttemptAtGeneralAI/runs/verbalization/training/overfit_single_nocompl/{model_name} \
  --dataset_name /home/mmordig/reinforcement/HumbleAttemptAtGeneralAI/runs/verbalization/datasets/alpha_geo${exp_type}_processed \
  --config ~/reinforcement/HumbleAttemptAtGeneralAI/geometry_translation/new/trl_sft_config.yml \
  --max_train_samples 1 \
  --num_train_epochs 150
  
  
  --model_name_or_path gpt2 \

python ~/reinforcement/HumbleAttemptAtGeneralAI/geometry_translation/new/make_model_predictions.py \
    --dataset_name ~/reinforcement/HumbleAttemptAtGeneralAI/runs/verbalization/datasets/alpha_geo${exp_type}_processed \
    --model_name_or_path /home/mmordig/reinforcement/HumbleAttemptAtGeneralAI/runs/verbalization/training/overfit_single_nocompl/gpt2 \
    --dataset_test_name train \
    --max_predict_samples 1 \
    --out_filename /home/mmordig/reinforcement/HumbleAttemptAtGeneralAI/runs/verbalization/predictions/exp${exp_type}/{model_name}_overfit_single_predictions.txt
    
--model_name_or_path /home/mmordig/reinforcement/HumbleAttemptAtGeneralAI/runs/verbalization/training/overfit_single_nocompl/Llama-2-7b-chat-hf/ \
--model_name_or_path /home/mmordig/reinforcement/HumbleAttemptAtGeneralAI/runs/verbalization/training/overfit_single/Llama-2-7b-hf \
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
source ~/reinforcement/HumbleAttemptAtGeneralAI/geometry_translation/new/verbalization_venv3/bin/activate
exp_type=_small
model=gpt2
python ~/reinforcement/HumbleAttemptAtGeneralAI/geometry_translation/new/make_model_predictions.py \
    ~/reinforcement/HumbleAttemptAtGeneralAI/runs/verbalization/training/exp${exp_type}/${model} \
    ~/reinforcement/HumbleAttemptAtGeneralAI/runs/verbalization/datasets/alpha_geo${exp_type}_processed \
    /home/mmordig/reinforcement/HumbleAttemptAtGeneralAI/runs/verbalization/predictions/exp${exp_type}/${model}_predictions.txt 5
```

Deploy the model (note: different generation params than for prediction):
```{bash}
source ~/reinforcement/HumbleAttemptAtGeneralAI/geometry_translation/new/verbalization_venv3/bin/activate
exp_type=_small
model=gpt2
python ~/reinforcement/HumbleAttemptAtGeneralAI/geometry_translation/new/deploy_model.py \
    ~/reinforcement/HumbleAttemptAtGeneralAI/runs/verbalization/training/exp${exp_type}/${model}
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
/lustre/home/mmordig/reinforcement/HumbleAttemptAtGeneralAI/geometry_translation/new/verbalization_venv3/lib/python3.10/site-packages/torch/nn/parallel/_functions.py:68: UserWarning: Was asked to gather along dimension 0, but all input tensors were scalars; will instead unsqueeze and return a vector.
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