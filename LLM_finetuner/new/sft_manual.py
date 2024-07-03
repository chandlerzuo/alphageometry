#%%

# python /home/mmordig/reinforcement/alphageometry/LLM_finetuner/new/sft_manual.py \
#     --overwrite_output_dir \
#     --use_peft \
#     --per_device_train_batch_size 64 \
#     --per_device_eval_batch_size 64 \
#     --model_name_or_path gpt2 \
#     --eval_steps 10 \
#     --evaluation_strategy steps \
#     --max_eval_samples 400 \
#     --dataset_name ~/reinforcement/alphageometry/LLM_finetuner/runs/verbalization/datasets/alpha_geo_arrow \
#     --config ~/reinforcement/alphageometry/LLM_finetuner/trl_sft_config.yml \
#     --output_dir test11 \
#     --max_generate_samples 2 \
#     --num_train_epochs 3
    
#     # --explicit_eos_str '[END]' \
#     # --extra_tokens_file ~/reinforcement/alphageometry/assets/def-patterns-desc.yml \
#     # --output_dir ~/reinforcement/alphageometry/LLM_finetuner/runs/verbalization/training/${run_dir_name}/{model_name}_{max_train_samples}ex_peft{use_peft} \
#     # --max_train_samples "$num_train_samples" \
        
        
import contextlib
import functools
import logging
import math
import os
import sys
import numpy as np
from transformers import (AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, get_scheduler,
    DataCollatorWithPadding,
    GenerationConfig
)
from trl import TrlParser
# from accelerate import set_seed
import accelerate
from datasets import load_dataset

import torch
import wandb

from LLM_finetuner.dataset import create_inputs_labels_for_generation, format_question_answer_batch_for_training
from LLM_finetuner.utils import check_hf_token_available, set_pad_token_if_not_set, subset_dataset
from LLM_finetuner.new.arg_dataclasses import SFTArguments, ModelArguments
from LLM_finetuner.new.utils import handle_fraction

# if is interactive session, import tqdm properly
# see https://stackoverflow.com/questions/2356399/tell-if-python-is-in-interactive-mode
# import __main__
# if not hasattr(__main__, '__file__'):
#     from tqdm.notebook import tqdm
# else: 
#     from tqdm import tqdm
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false" # to avoid warnings
assert torch.cuda.is_available(), "cuda not available"

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

model_name = "gpt2"
seed = 1
learning_rate = 1e-6
epochs = 3

print("Args:", sys.argv)
# attention: if values are set to the same value as their default value, they can be overridden by the config file
sys.argv = ["dummy", '--overwrite_output_dir', '--use_peft', '--per_device_train_batch_size', '4', '--per_device_eval_batch_size', '4', '--model_name_or_path', 'gpt2', '--eval_steps', '10', '--evaluation_strategy', 'steps', '--max_eval_samples', '2', '--dataset_name', '/home/mmordig/reinforcement/alphageometry/LLM_finetuner/runs/verbalization/datasets/alpha_geo_arrow', '--config', '/home/mmordig/reinforcement/alphageometry/LLM_finetuner/trl_sft_config.yml', '--output_dir', 'test11', '--num_train_epochs', '2']
args, training_args, model_args = TrlParser((SFTArguments, TrainingArguments, ModelArguments)).parse_args_and_config()

check_hf_token_available()
accelerate.utils.set_seed(seed)

from transformers import GPT2LMHeadModel, GPT2TokenizerFast # for intellisense
tokenizer: GPT2TokenizerFast = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
model: GPT2LMHeadModel = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path)
# model = GPT2LMHeadModel.from_pretrained(model_args.model_name_or_path)
device = "cuda"
model.to(device)
set_pad_token_if_not_set(model, tokenizer)

# todo
# training loop
# checkpointing
# wandb
# resume from, set seed for dataloaders
# peft
# accelerate

wandb.init(project="sft", config=vars(args), mode="disabled")

dataset = load_dataset(args.dataset_name)
train_dataset = subset_dataset(dataset[args.dataset_train_name], args.max_train_samples)
eval_dataset = subset_dataset(dataset[args.dataset_eval_name], args.max_eval_samples)
assert training_args.max_steps == -1 # never set
num_training_steps = training_args.num_train_epochs * len(train_dataset)

train_dataset = train_dataset.map(
    functools.partial(format_question_answer_batch_for_training, tokenizer=tokenizer), 
    batched=True
)
logger.info(f"Example datapoints train: {train_dataset[:2]}")
def tokenize_batch(batch):
    return tokenizer(batch["text"])
train_dataset = train_dataset.map(tokenize_batch, batched=True)
# or see Trainer._set_signature_columns_if_needed, ._remove_unused_columns
train_dataset = train_dataset.remove_columns(["text", "fl_statement", "nl_statement"])

data_collator = DataCollatorWithPadding(tokenizer)
from torch.utils.data import DataLoader
train_dataloader = DataLoader(
    train_dataset, 
    batch_size=training_args.per_device_train_batch_size, 
    num_workers=training_args.dataloader_num_workers,
    collate_fn=data_collator,
    pin_memory=True,
    shuffle=True
)
eval_dataloader = DataLoader(
    eval_dataset, 
    batch_size=training_args.per_device_eval_batch_size, 
    num_workers=training_args.dataloader_num_workers,
    collate_fn=data_collator,
    pin_memory=True,
    shuffle=False
)
# train_dataset
# next(iter(train_dataloader))

logger.info(f"Example datapoints eval: {eval_dataset[:2]}")
#%%
eval_dataset = create_inputs_labels_for_generation(eval_dataset, tokenizer=tokenizer)
eval_dataset[:2]
len(eval_dataset)
#%%

# gen_config = args.eval_generation_config
gen_config = GenerationConfig(
    predict_with_generate=True, max_new_tokens=70, num_beams=2
)
@contextlib.contextmanager
def set_padding_side(tokenizer, padding_side):
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = padding_side
    yield
    tokenizer.padding_side = old_padding_side
    
# tokenizer.padding_side = "left"
# encoded_inputs = tokenizer(eval_dataset[:2]["text"], return_tensors="pt", padding=True, truncation=True)
# encoded_inputs = encoded_inputs.to(device)
# encoded_generations = model.generate(**encoded_inputs, generation_config=gen_config)
# tokenizer.batch_decode(encoded_generations, skip_special_tokens=True)

def model_generate_on_batch(batch):
    with set_padding_side(tokenizer, "left"):
        encoded_inputs = tokenizer(batch["text"], return_tensors="pt", padding=True, truncation=True)
    encoded_inputs = encoded_inputs.to(model.device)
    encoded_generations = model.generate(**encoded_inputs, generation_config=gen_config)
    return {"generation": tokenizer.batch_decode(encoded_generations, skip_special_tokens=True)}
# generations = eval_dataset.map(
#     model_generate_on_batch, batched=True, batch_size=training_args.per_device_eval_batch_size,
#     remove_columns=set(eval_dataset.column_names) - {"labels"}
# )
# wandb.Table(dataframe=generations.to_pandas())
#%%
# tokenizer.add_special_tokens

#%%
optimizer = torch.optim.AdamW(model.parameters(), lr=training_args.learning_rate)
scheduler = get_scheduler(
    training_args.lr_scheduler_type, optimizer=optimizer, 
    num_warmup_steps=handle_fraction(training_args.warmup_steps, num_training_steps), 
    num_training_steps=num_training_steps,
    scheduler_specific_kwargs=training_args.lr_scheduler_kwargs,
)

def extract_loss(outputs):
    return outputs["loss"] if isinstance(outputs, dict) else outputs[0]

# ignore keys at inference
ignore_keys = None
if ignore_keys is None:
    if hasattr(model, "config"):
        ignore_keys = getattr(model.config, "keys_to_ignore_at_inference", [])
    else:
        ignore_keys = []
logger.info(f"ignore keys: {ignore_keys}")
def extract_logits(outputs):
    if isinstance(outputs, dict):
        logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
    else:
        logits = outputs[1:]
    return logits

# class Controller:
#     should_evaluate = False
#     should_log = False
#     should_save = False
#     def on_epoch_end(self, epoch, logs):
#         self.should_evaluate = True
#         self.should_log = True
#         self.should_save = True
# for epoch in range(training_args.num_train_epochs):

from transformers.trainer_pt_utils import EvalLoopContainer, find_batch_size
from transformers.trainer_utils import EvalPrediction, denumpify_detensorize
compute_metrics = None
def run_evaluation():
    metric_key_prefix = "eval"
    batch_size = eval_dataloader.batch_size
    
    all_losses = EvalLoopContainer(training_args.eval_do_concat_batches, padding_index=-100)
    all_preds = EvalLoopContainer(training_args.eval_do_concat_batches, padding_index=-100)
    all_labels = EvalLoopContainer(training_args.eval_do_concat_batches, padding_index=-100)
    # all_inputs = EvalLoopContainer(training_args.eval_do_concat_batches, padding_index=-100)
    
    model.eval()
    with torch.no_grad():
        for inputs in eval_dataloader:
            observed_batch_size = find_batch_size(inputs)
            # if batch_size is None:
            #     batch_size = observed_batch_size
                
            outputs = model(**inputs)
            losses = extract_loss(outputs)
            losses = losses.repeat(batch_size) # repeat, so that averaging gives the right result again
            predictions = extract_logits(outputs)
            # all_losses.add(losses)
            all_losses.add(loss)
            all_preds.add(predictions)
            all_labels.add(inputs["labels"])
    
    # may need to do this more often, so move into for loop
    all_losses.to_cpu_and_numpy()
    all_preds.to_cpu_and_numpy()
    all_labels.to_cpu_and_numpy()
    # all_inputs.to_cpu_and_numpy()
        
    all_losses = all_losses.get_arrays()
    all_preds = all_preds.get_arrays()
    all_labels = all_labels.get_arrays()
    # all_inputs = all_inputs.get_arrays()
    
    if compute_metrics is not None:
        metrics = compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels))
    else:
        metrics = {}
    metrics = denumpify_detensorize(metrics)
    if isinstance(all_losses, list) and all_losses:
        metrics[f"{metric_key_prefix}_loss"] = np.concatenate(all_losses).mean().item()
    elif isinstance(all_losses, np.ndarray):
        metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()

    for key in list(metrics.keys()):
        if not key.startswith(f"{metric_key_prefix}_"):
            metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)
    # return metrics
    logger.info(f"Metrics: {metrics}")
    wandb.log(metrics)

for epoch in tqdm(range(math.ceil(training_args.num_train_epochs)), desc="Epoch"):
    # logger.info(f"Training epoch {epoch}")
    
    for batch in tqdm(train_dataloader, desc="Batch"):
        model.train()
        batch = batch.to(device)
        if "labels" not in batch:
            batch["labels"] = batch["input_ids"]
        outputs = model(**batch)
        loss = extract_loss(outputs)
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        model.zero_grad()
        
        # evaluation
        run_evaluation()
                
        # evaluation via text generation        
        generations = eval_dataset.map(
            model_generate_on_batch, batched=True, batch_size=training_args.per_device_eval_batch_size,
            remove_columns=set(eval_dataset.column_names) - {"labels"}
        )
        wandb.log(wandb.Table(dataframe=generations.to_pandas()))
    
logger.info("Done training")
# %%
