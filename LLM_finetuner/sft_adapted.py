# Perform supervised fine-tuning
# Modified from /home/mmordig/reinforcement/HumbleAttemptAtGeneralAI/geometry_translation/new/verbalization_venv3/lib/python3.10/site-packages/trl/commands/scripts/sft.py
# also see /home/mmordig/reinforcement/venvs/rlenv/lib/python3.10/site-packages/trl/commands/cli.py

# flake8: noqa
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
# regular:
python examples/scripts/sft.py \
    --model_name_or_path="facebook/opt-350m" \
    --report_to="wandb" \
    --learning_rate=1.41e-5 \
    --per_device_train_batch_size=64 \
    --gradient_accumulation_steps=16 \
    --output_dir="sft_openassistant-guanaco" \
    --logging_steps=1 \
    --num_train_epochs=3 \
    --max_steps=-1 \
    --push_to_hub \
    --gradient_checkpointing \

# peft:
python examples/scripts/sft.py \
    --model_name_or_path="facebook/opt-350m" \
    --report_to="wandb" \
    --learning_rate=1.41e-5 \
    --per_device_train_batch_size=64 \
    --gradient_accumulation_steps=16 \
    --output_dir="sft_openassistant-guanaco" \
    --logging_steps=1 \
    --num_train_epochs=3 \
    --max_steps=-1 \
    --push_to_hub \
    --gradient_checkpointing \
    --use_peft \
    --lora_r=64 \
    --lora_alpha=16
"""
from dataclasses import asdict, dataclass, field
import logging
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import os, sys; sys.path.append(os.path.join(os.path.dirname(__file__), "..")) #todo, or __vsc_ipynb_file__ in jupyter
from utils import set_pad_token_if_not_set, subset_dataset
from question_answer_utils import get_question_answer_to_chat_formatter, response_template

# TRL_USE_RICH = os.environ.get("TRL_USE_RICH", False)
TRL_USE_RICH = True

from trl.commands.cli_utils import init_zero_verbose, SftScriptArguments, TrlParser

if TRL_USE_RICH:
    # init_zero_verbose()
    FORMAT = "%(message)s"

    from rich.console import Console
    from rich.logging import RichHandler

import torch
from datasets import load_dataset, load_from_disk

from tqdm.rich import tqdm
from transformers import AutoTokenizer, TrainingArguments, AutoModelForCausalLM
from transformers.trainer_utils import get_last_checkpoint
from transformers.models.llama import tokenization_llama_fast

from trl import (
    ModelConfig,
    RichProgressCallback,
    SFTTrainer,
    DataCollatorForCompletionOnlyLM,
    setup_chat_format,
    get_peft_config,
    get_quantization_config,
    get_kbit_device_map,
)

tqdm.pandas()

@dataclass
class SftScriptArgumentsExtra(SftScriptArguments):
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    train_completions_only: bool = field(
        default=False, 
        metadata={"help": "Whether to train on completions only (i.e. mask out the rest in the loss computation)"}
    )

if TRL_USE_RICH:
    # NOTE: ignored if using init_zero_verbose() because it calls logging.basicConfig already!!
    logging.basicConfig(format=FORMAT, datefmt="[%X]", handlers=[RichHandler()], level=logging.INFO)


if __name__ == "__main__":
    logger = logging
    
    parser = TrlParser((SftScriptArgumentsExtra, TrainingArguments, ModelConfig))
    # import sys; sys.argv = "python sft --overwrite_output_dir --output_dir runs.trl_sft --config /home/mmordig/reinforcement/HumbleAttemptAtGeneralAI/geometry_translation/new/trl_chat_finetune.yml".split(" ")
    # print(sys.argv)
    args, training_args, model_config = parser.parse_args_and_config()

    # Force use our print callback
    if TRL_USE_RICH:
        training_args.disable_tqdm = True
        console = Console()

    ################
    # Model & Tokenizer
    ################
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    model = AutoModelForCausalLM.from_pretrained(model_config.model_name_or_path, **model_kwargs)
    # add_eos_token to learn ending properly, not needed with chat format
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path, use_fast=True, add_eos_token=True)
    # tokenizer.pad_token = tokenizer.eos_token
    set_pad_token_if_not_set(model, tokenizer)

    ################
    # Dataset
    ################
    # raw_datasets = load_dataset(args.dataset_name)
    logger.info("Loading raw dataset")
    raw_datasets = load_from_disk(args.dataset_name)
    logger.info("Loaded raw dataset")

    train_dataset = raw_datasets[args.dataset_train_name]
    eval_dataset = raw_datasets[args.dataset_test_name]
    
    train_dataset = subset_dataset(train_dataset, args.max_train_samples)
    eval_dataset = subset_dataset(eval_dataset, args.max_eval_samples)
    
    # adapt model_dir
    
    if "HF_TOKEN" not in os.environ:
        default_token_path = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
        if (default_token_path is None) or (not (Path(default_token_path) / "token").exists()):
            # cannot access restricted models
            logger.warning("HF_TOKEN not available on remote machine")
        
    def adapt_output_dir(output_dir):
        # output_dir = Path(output_dir) / model_config.model_name_or_path.split("/")[-1]
        output_dir = Path(output_dir.format(
            model_name=model_config.model_name_or_path.split("/")[-1], 
            **asdict(args), **asdict(training_args), **asdict(model_config),
        ))
        logger.info(f"Adapted output dir to {output_dir}")
        # print(f"Adapted output dir to {output_dir}")
        output_dir.mkdir(exist_ok=True, parents=True)
        return str(output_dir)
    training_args.output_dir = adapt_output_dir(training_args.output_dir)
    
    def adapt_resume_from_checkpoint(resume_from_checkpoint):
        if resume_from_checkpoint == "latest_if_available":
            # passing resume_from_checkpoint=True will fail if no checkpoint exists
            # so "latest_if_available" loads the most recent one if one exists
            resume_from_checkpoint = get_last_checkpoint(training_args.output_dir) # may be None if nothing found
            logger.info(f"Resuming from previous checkpoint {resume_from_checkpoint} (None means from scratch)")
        return resume_from_checkpoint
    training_args.resume_from_checkpoint = adapt_resume_from_checkpoint(training_args.resume_from_checkpoint)
    
    if training_args.gradient_checkpointing:
        # still seems to be an issue, https://github.com/huggingface/transformers/issues/26969
        logger.info("Setting use_reentrant to False explicitly")
        if training_args.gradient_checkpointing_kwargs is None:
            training_args.gradient_checkpointing_kwargs = {'use_reentrant':False}
        else:
            training_args.gradient_checkpointing_kwargs['use_reentrant'] = False


    # # this chat format is ChatML which is different from the tokenizer's one which may be optimized to the model, so we use that instead
    # model, tokenizer = setup_chat_format(model, tokenizer)

    print(f"Example training data: {train_dataset[:2][args.dataset_text_field]}")
    
    if tokenizer.chat_template is not None:
        logger.info("Detected chat model, formatting according to chat template")
        # assumes user-assistant roles
        formatting_func = get_question_answer_to_chat_formatter(tokenizer, text_column=args.dataset_text_field)
        print(f"Example chat format: {formatting_func(train_dataset[:2])}")
        args.dataset_text_field = None
        
        if args.train_completions_only:
            
            # example to see format
            # model_name_or_path = "meta-llama/Llama-2-7b-chat-hf"
            # # model_name_or_path = "meta-llama/Llama-2-7b-hf"
            # tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)#, add_eos_token=True)
            # # inputs = "Hello world"
            # inputs = [
            #     {"role": "system", "content": "System"},
            #     # must be ordered as user-assistant alternating sequence
            #     {"role": "user", "content": "Question1"},
            #     {"role": "assistant", "content": "Answer1"},
            #     {"role": "user", "content": "Question2"},
            #     {"role": "assistant", "content": "Answer2"},
            # ]
            # print(tokenizer.apply_chat_template(inputs, tokenize=False))
            
            assert isinstance(tokenizer, tokenization_llama_fast.LlamaTokenizerFast), "only supported for llama currently"
            response_template = tokenization_llama_fast.E_INST # [/INST]
            instruction_template = tokenization_llama_fast.B_INST + tokenizer.bos_token # [INST]<s>
            logger.info(f"Using instruction_template '{instruction_template}', response_template '{response_template}'")
    else:
        instruction_template = None
        # if model_config.model_name_or_path == "gpt2":
        if False: 
            logger.info("Explicitly appending EOS token to dataset for gpt2")
            orig_dataset_text_field = args.dataset_text_field # will be set to None later
            formatting_func = lambda batch: [(ex + " " + tokenizer.eos_token) for ex in batch[orig_dataset_text_field]]
            args.dataset_text_field = None
        else:
            formatting_func = None
        
    data_collator = None
    if args.train_completions_only:
        logger.info("Training on completions only")
        assert not args.packing, "cannot use packing when training on completions"
        data_collator = DataCollatorForCompletionOnlyLM(response_template=response_template, instruction_template=instruction_template, tokenizer=tokenizer)

    ################
    # Optional rich context managers
    ###############
    init_context = nullcontext() if not TRL_USE_RICH else console.status("[bold green]Initializing the SFTTrainer...")
    save_context = (
        nullcontext()
        if not TRL_USE_RICH
        else console.status(f"[bold green]Training completed! Saving the model to {training_args.output_dir}")
    )

    ################
    # Training
    ################
    with init_context:
        trainer = SFTTrainer(
            # model=model_config.model_name_or_path,
            # model_init_kwargs=model_kwargs,
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            dataset_text_field=args.dataset_text_field,
            formatting_func=formatting_func,
            max_seq_length=args.max_seq_length,
            tokenizer=tokenizer,
            packing=args.packing,
            peft_config=get_peft_config(model_config),
            callbacks=[RichProgressCallback] if TRL_USE_RICH else None,
        )

    # trainer.train(resume_from_checkpoint=Path(training_args.output_dir).exists())
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    
    # trainer.train(resume_from_checkpoint=False) # todo

    with save_context:
        trainer.save_model(training_args.output_dir)
        
    logger.info(f"Trained model starting from '{model_config.model_name_or_path}' for {training_args.num_train_epochs}")
    logger.info(f"Saved model to '{training_args.output_dir}'")
    logger.info("Done with SFT training")
