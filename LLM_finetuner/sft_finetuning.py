#!/usr/bin/env python


# Perform supervised fine-tuning
#%%
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

#%%
from dataclasses import asdict, dataclass, field
import functools
import logging
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Optional
import yaml
import sys

import LLM_finetuner
from LLM_finetuner.dataset import SYSTEM_MESSAGE, SYSTEM_MESSAGE_JSON, convert_formalalphageom_to_json, create_inputs_labels_for_generation, format_question_answer_batch_for_generation, format_question_answer_batch_for_training, replace_words_with_tokens
from LLM_finetuner.generation_helpers import MakeModelGenerations
from LLM_finetuner.utils import args_as_dict, check_hf_token_available, set_pad_token_if_not_set, subset_dataset, get_model_name_from_name_or_path, add_new_tokens_with_average_init
# from LLM_finetuner.question_answer_utils import extract_question_answer, get_question_answer_to_chat_formatter, response_template

# TRL_USE_RICH = os.environ.get("TRL_USE_RICH", False)
# TRL_USE_RICH = True
TRL_USE_RICH = False

from trl.commands.cli_utils import init_zero_verbose, SftScriptArguments, TrlParser

if TRL_USE_RICH:
    # init_zero_verbose()
    FORMAT = "%(message)s"

    from rich.console import Console
    from rich.logging import RichHandler

import torch
from datasets import load_dataset, load_from_disk, DatasetDict

from tqdm.rich import tqdm
from transformers import AutoTokenizer, TrainingArguments, AutoModelForCausalLM
from transformers.trainer_utils import get_last_checkpoint
from transformers.models.llama import tokenization_llama_fast
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, GenerationConfig

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
    max_generate_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of examples for generation to this "
                "value if set."
            )
        },
    )
    train_completions_only: bool = field(
        default=False, 
        metadata={"help": "Whether to train on completions only (i.e. mask out the rest in the loss computation)"}
    )
    extra_tokens_file: Optional[Path] = field(
        default=None,
        metadata={
            "help": 
                "yaml file with extra tokens to add; entries with no values are randomly init'd, "
                "entries with values are initialized with average embedding of the tokens of its description"
        }
    )
    explicit_eos_str: Optional[str] = field(
        default="",
        metadata={
            "help": 
                "explicit EOS token to add at the end of each sample (since some tokenizers don't "
                "seem to do it even with eos_token); also adds token to model if not present"
        }
    )
    gen_steps: Optional[int] = field(
        default=-1,
        metadata={
            "help": "Generate every gen_steps steps, -1 to disable"
        }
    )
    extra_metadata: Optional[str] = field(
        default=None,
        metadata={
            "help": "metadata, useful to show in wandb"
        }
    )
if TRL_USE_RICH:
    # NOTE: ignored if using init_zero_verbose() because it calls logging.basicConfig already!!
    logging.basicConfig(format=FORMAT, datefmt="[%X]", handlers=[RichHandler()], level=logging.INFO)


logger = logging
def adapt_resume_from_checkpoint(resume_from_checkpoint, output_dir):
    if resume_from_checkpoint == "latest_if_available":
        # passing resume_from_checkpoint=True will fail if no checkpoint exists
        # so "latest_if_available" loads the most recent one if one exists
        resume_from_checkpoint = get_last_checkpoint(output_dir) # may be None if nothing found
    elif resume_from_checkpoint.lower() == "false":
        resume_from_checkpoint = None
    logger.info(f"Resuming from checkpoint {resume_from_checkpoint} (None means from scratch)")
    return resume_from_checkpoint

#%%
def main():
    1
    #%%
    os.environ["WANDB_PROJECT"] = "alphageom_verb1" # used by WandbCallback: when wandb is broken (especially when adding new metrics), so choosing a new project may help
    os.environ["TOKENIZERS_PARALLELISM"] = "false" # to avoid warnings
    check_hf_token_available()

    # sys.argv = ["dummy", '--overwrite_output_dir', '--use_peft', '--per_device_train_batch_size', '64', '--per_device_eval_batch_size', '64', '--model_name_or_path', 'gpt2', '--eval_steps', '10', '--evaluation_strategy', 'steps', '--max_eval_samples', '400', '--max_generate_samples', '2', '--explicit_eos_str', '[END]', '--extra_tokens_file', '/home/mmordig/reinforcement/alphageometry/assets/def-patterns-desc.yml', '--output_dir', '/home/mmordig/reinforcement/alphageometry/LLM_finetuner/runs/verbalization/training/debug112/{model_name}_{max_train_samples}ex_peft{use_peft}', '--dataset_name', '/home/mmordig/reinforcement/alphageometry/LLM_finetuner/runs/verbalization/datasets/alpha_geo_arrow', '--config', '/home/mmordig/reinforcement/alphageometry/LLM_finetuner/trl_sft_config.yml', '--max_train_samples', '10', '--num_train_epochs', '100000']
    # logger.warning("Dummy args set")
    
    parser = TrlParser((SftScriptArgumentsExtra, TrainingArguments, ModelConfig)) # do not subclass TrainingArguments because TrlParser YamlConfigParser hardcodes this class!
    # import sys; sys.argv = "python sft --overwrite_output_dir --output_dir runs.trl_sft --config /home/mmordig/reinforcement/HumbleAttemptAtGeneralAI/geometry_translation/new/trl_chat_finetune.yml".split(" ")
    print(f"Received the following args: {sys.argv}")
    args: SftScriptArgumentsExtra
    training_args: TrainingArguments
    model_config: ModelConfig
    args, training_args, model_config = parser.parse_args_and_config()
    # directly adding extra fields to TrainingArguments is not possible because TrlParser/YamlConfigParser hardcodes TrainingArguments
    training_args.extra_metadata = args.extra_metadata # wandb.config picks it up as metadata for the run
    training_args.gen_steps = args.gen_steps
    
    # todo
    training_args.generation_config = GenerationConfig(predict_with_generate=True, max_new_tokens=70, num_beams=2) #todo
    training_args.predict_with_generate = True
    training_args.generation_max_length = None
    training_args.generation_num_beams = None
    
    # Force use our print callback
    if TRL_USE_RICH:
        training_args.disable_tqdm = True
        console = Console()
    
    #%%
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
    logger.info(f"Loading model {model_config.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(model_config.model_name_or_path, **model_kwargs)
    logger.info(f"Loading tokenizer")
    # add_eos_token to learn ending properly, not needed with chat format
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path, use_fast=True, add_eos_token=True)
    # tokenizer.pad_token = tokenizer.eos_token
    # set_pad_token_if_not_set(model, tokenizer)
    logger.warning("Setting pad to eos token")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    def_to_desc = None
    if args.extra_tokens_file is not None:
        logger.info(f"Loading extra tokens from '{args.extra_tokens_file}'")
        def_to_desc = yaml.safe_load(args.extra_tokens_file.read_text())
        logger.info(f"Vocabulary length before: {len(tokenizer)}")    
        add_new_tokens_with_average_init(model, tokenizer, {f"[{key}]": value for (key, value) in def_to_desc.items()})
        logger.info(f"Vocabulary length after: {len(tokenizer)}")    

    #%%
    ################
    # Dataset
    ################
    assert args.dataset_text_field == "text"
    logger.info("Loading dataset")
    raw_datasets = load_dataset(args.dataset_name)
    logger.info(raw_datasets)
    raw_datasets = DatasetDict({
        "train": subset_dataset(raw_datasets[args.dataset_train_name], args.max_train_samples),
        "eval": subset_dataset(raw_datasets[args.dataset_test_name], args.max_eval_samples),
        "generation_test": subset_dataset(raw_datasets[args.dataset_test_name], args.max_generate_samples),
        "generation_train": subset_dataset(raw_datasets[args.dataset_train_name], args.max_generate_samples),
    })
    logger.info(f"After subsetting: {raw_datasets}")
    if def_to_desc is not None:
        geom_tokens = sorted(list(def_to_desc.keys()), key=lambda x: -len(x))
        raw_datasets = raw_datasets.map(lambda x: {"fl_statement": replace_words_with_tokens(x["fl_statement"], geom_tokens)}, num_proc=8)
    
    convert_to_json = True
    if convert_to_json:
        def json_converter(batch):
            return {"fl_statement": [convert_formalalphageom_to_json(x) for x in batch["fl_statement"]]}
        logger.info("Converting to JSON")
        raw_datasets = raw_datasets.map(json_converter, batched=True)
    
    # raw_datasets = load_from_disk(args.dataset_name)
    # logger.info("Loaded raw dataset")
    extra_formatting_args = args_as_dict(
        system_message=SYSTEM_MESSAGE_JSON if convert_to_json else SYSTEM_MESSAGE, 
        end=args.explicit_eos_str
    )
    train_dataset = raw_datasets["train"].map(
        functools.partial(format_question_answer_batch_for_training, tokenizer=tokenizer, **extra_formatting_args), 
        batched=True
    )
    eval_dataset = raw_datasets["eval"].map(
        functools.partial(format_question_answer_batch_for_training, tokenizer=tokenizer, **extra_formatting_args), 
        batched=True
    )
    if args.max_generate_samples:
        gen_dataset_test = create_inputs_labels_for_generation(raw_datasets["generation_test"], tokenizer=tokenizer, **extra_formatting_args)
        # also evaluate on training set
        gen_dataset_train = create_inputs_labels_for_generation(raw_datasets["generation_train"], tokenizer=tokenizer, **extra_formatting_args)
    else: 
        gen_dataset_test = None
    
    # train_dataset = train_dataset.map(
    #     functools.partial(format_question_answer_batch_for_training, tokenizer=tokenizer), 
    #     batched=True
    # )
    # eval_dataset = create_inputs_labels_for_generation(eval_dataset, tokenizer=tokenizer)

    # train_dataset = raw_datasets[args.dataset_train_name]
    # eval_dataset = raw_datasets[args.dataset_test_name]
    
    # train_dataset = subset_dataset(train_dataset, args.max_train_samples)
    # eval_dataset = subset_dataset(eval_dataset, args.max_eval_samples)
    # eval_dataset = eval_dataset.map(
    #     lambda x: {args.dataset_text_field: "### Question: " + extract_question_answer(x[args.dataset_text_field])[0]}
    # )
    
    print(f"Example datapoints train: {train_dataset[:2]}")
    print(f"Example datapoints eval: {eval_dataset[:2]}")
    logger.info(f'Example train tokenization: {tokenizer.tokenize(train_dataset[0]["text"])}')
    if gen_dataset_test is not None:
        print(f"Example datapoints gen: {gen_dataset_test[:2]}")
        print("Generation text:", gen_dataset_test[0]["text"])
        print("Generation labels:", gen_dataset_test[0]["labels"])
        logger.info(f'Example gen tokenization: {tokenizer.tokenize(gen_dataset_test[0]["text"])}')
    
    logger.info(f"using eos token {args.explicit_eos_str}")
    tokenizer.add_tokens(args.explicit_eos_str, special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))
    
    #%%
    def format_output_dir_path(output_dir):
        # output_dir = Path(output_dir) / get_model_name_from_name_or_path(model_config.model_name_or_path)
        output_dir = Path(output_dir.format(
            model_name=get_model_name_from_name_or_path(model_config.model_name_or_path),
            **asdict(args), **asdict(training_args), **asdict(model_config),
        ))
        logger.info(f"Adapted output dir to {output_dir}")
        # print(f"Adapted output dir to {output_dir}")
        output_dir.mkdir(exist_ok=True, parents=True)
        return str(output_dir)
    run_name_was_default = (training_args.run_name == training_args.output_dir) # if run_name is not set, TrainingArguments.__post_init sets it to the output dir
    training_args.output_dir = format_output_dir_path(training_args.output_dir)
    if run_name_was_default:
        training_args.run_name = training_args.output_dir
        logger.info(f"Also changed run name to {training_args.run_name}")
    
    training_args.resume_from_checkpoint = adapt_resume_from_checkpoint(training_args.resume_from_checkpoint, output_dir=training_args.output_dir)
    
    # TrlParser should normally do it, but does not work due to hardcoding of classes
    if training_args.gradient_checkpointing:
        # still seems to be an issue, https://github.com/huggingface/transformers/issues/26969
        logger.info("Setting use_reentrant to False explicitly")
        if training_args.gradient_checkpointing_kwargs is None:
            training_args.gradient_checkpointing_kwargs = {'use_reentrant':False}
        else:
            training_args.gradient_checkpointing_kwargs['use_reentrant'] = False
    
    formatting_func = None
    assert not args.train_completions_only

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
    
    #%%
    callbacks = []
    if TRL_USE_RICH:
        callbacks.append(RichProgressCallback)
    if gen_dataset_test is not None:
        callbacks.append(MakeModelGenerations(gen_dataset=gen_dataset_test, prefix="eval_"))
        callbacks.append(MakeModelGenerations(gen_dataset=gen_dataset_train, prefix="train_"))
    with init_context:
        trainer = SFTTrainer(
        # trainer = SFTSeq2SeqTrainer(
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
            callbacks=callbacks,
        )

    # #%%
    # training_args.device
    # eval_dataset.column_names
    #%%
    # from LLM_finetuner.generation_helpers import MakeModelGenerations
    # ttt = MakeModelGenerations(gen_dataset=gen_dataset)
    # # model = trainer.accelerator.prepare(model) # to put it on the same device as args.device, can also place input on model.device
    # df = ttt.on_epoch_end(training_args, None, None, model, tokenizer)
    # # imperfect splitting, but hopefully for many chat templates
    # # split by "assistant"
    # #%%
    # # df.iloc[0]["gt_answer"]
    # # df.iloc[0]["pred_answer"]
    # # df.iloc[0]["generation"]
    # df.iloc[0]
    
    #%%
    # return trainer, training_args.resume_from_checkpoint
    # trainer.train(resume_from_checkpoint=Path(training_args.output_dir).exists())
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    
    # trainer.train(resume_from_checkpoint=False) # todo

    with save_context:
        trainer.save_model(training_args.output_dir)
        
    logger.info(f"Trained model starting from '{model_config.model_name_or_path}' for {training_args.num_train_epochs}")
    logger.info(f"Saved model to '{training_args.output_dir}'")
    logger.info("Done with SFT training")

#%%
if __name__ == "__main__":
    main()
    
    
# if args.extra_metadata is None:
#     # not working right now because it needs to be added to training_args (but not supported by TrlParser) to be logged to wandb
#     # training args will be uploaded as config to wandb
#     jobad_file = os.environ.get("_CONDOR_JOB_AD", None)
#     if jobad_file is not None:
#         args.extra_metadata = Path(jobad_file).read_text()

# # this chat format is ChatML which is different from the tokenizer's one which may be optimized to the model, so we use that instead
# model, tokenizer = setup_chat_format(model, tokenizer)

# print(f"Example training data before transformations: {train_dataset[:2][args.dataset_text_field]}")

# if tokenizer.chat_template is not None:
#     logger.info("Detected chat model, formatting according to chat template")
#     # assumes user-assistant roles
#     formatting_func = get_question_answer_to_chat_formatter(tokenizer, text_column=args.dataset_text_field)
#     args.dataset_text_field = None
    
#     if args.train_completions_only:
        
#         # example to see format
#         # model_name_or_path = "meta-llama/Llama-2-7b-chat-hf"
#         # # model_name_or_path = "meta-llama/Llama-2-7b-hf"
#         # tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)#, add_eos_token=True)
#         # # inputs = "Hello world"
#         # inputs = [
#         #     {"role": "system", "content": "System"},
#         #     # must be ordered as user-assistant alternating sequence
#         #     {"role": "user", "content": "Question1"},
#         #     {"role": "assistant", "content": "Answer1"},
#         #     {"role": "user", "content": "Question2"},
#         #     {"role": "assistant", "content": "Answer2"},
#         # ]
#         # print(tokenizer.apply_chat_template(inputs, tokenize=False))
        
#         assert isinstance(tokenizer, tokenization_llama_fast.LlamaTokenizerFast), "only supported for llama currently"
#         response_template = tokenization_llama_fast.E_INST # [/INST]
#         instruction_template = tokenization_llama_fast.B_INST + tokenizer.bos_token # [INST]<s>
#         logger.info(f"Using instruction_template '{instruction_template}', response_template '{response_template}'")
# else:
#     instruction_template = None
#     # if model_config.model_name_or_path == "gpt2":
#     orig_dataset_text_field = args.dataset_text_field # will be set to None later
#     formatting_func = lambda elem: elem[orig_dataset_text_field]
#     args.dataset_text_field = None
    
# if args.explicit_eos_str is not None:
#     assert formatting_func is not None
#     old_formatting_func = formatting_func
#     formatting_func = lambda batch: [(x + " " + args.explicit_eos_str) for x in old_formatting_func(batch)]
    
#     tokenizer.add_tokens(args.explicit_eos_str, special_tokens=True)
#     model.resize_token_embeddings(len(tokenizer))
    
# print(f"Example datapoints: {formatting_func(train_dataset[:2])}")





# data_collator = None
# if args.train_completions_only:
#     logger.info("Training on completions only")
#     assert not args.packing, "cannot use packing when training on completions"
#     data_collator = DataCollatorForCompletionOnlyLM(response_template=response_template, instruction_template=instruction_template, tokenizer=tokenizer)



# class SFTSeq2SeqTrainer(SFTTrainer, Seq2SeqTrainer):
#     def __init__(self, *args, **kwargs):
#         SFTTrainer.__init__(self, *args, **kwargs)
        
#         # copied from Seq2SeqTrainer.__init__
#         # Override self.model.generation_config if a GenerationConfig is specified in args.
#         # Priority: args.generation_config > model.generation_config > default GenerationConfig.
#         if self.args.generation_config is not None:
#             gen_config = self.load_generation_config(self.args.generation_config)
#             self.model.generation_config = gen_config
# %%
