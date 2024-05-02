"""
Make predictions with the model and write to a file (along with the ground-truth)

"""

#%%
from dataclasses import dataclass, field
from typing import Optional
import torch
import more_itertools
import logging
import sys
import textwrap

import tqdm
from transformers import pipeline, HfArgumentParser
from datasets import load_from_disk
import gradio as gr
from contextlib import nullcontext, redirect_stdout

import os, sys; sys.path.append(os.path.join(os.path.dirname(__file__), ".")) #todo
from question_answer_utils import extract_answer, extract_question_prompt, get_question_answer_to_chat_formatter
from utils import load_model_for_inference, setup_logging, subset_dataset

#%%
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
setup_logging()

#%%
# model_name_or_path = sys.argv[1]
# dataset_name = sys.argv[2]
# filename_predictions_out = sys.argv[3]
# max_predict_samples = int(sys.argv[4]) if len(sys.argv) >= 4 else 30

@dataclass
class PredictionArguments:
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": ("The model checkpoint for weights initialization.")},
    )
    dataset_name: str = field(default=None, metadata={"help": "the dataset name or directory"})
    dataset_test_name: str = field(default="test", metadata={"help": "the name of the test set of the dataset"})
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "number of samples to predict, int or fraction of the dataset size"
            )
        },
    )
    out_filename: Optional[str] = field(
        default=None,
        metadata={"help": "the name of the output file, will interpolate {model_name} with last name of model"}
    )
    dataset_text_field: str = field(default="text", metadata={"help": "the text field of the dataset"})
    max_new_tokens: int = field(default=70, metadata={"help": "maximum number of new tokens to generate (per datapoint)"})
    # num_beams: Optional[int] = field()
    
args, = HfArgumentParser((PredictionArguments, )).parse_args_into_dataclasses()
model_name_or_path = args.model_name_or_path
dataset_name = args.dataset_name
dataset_test_name = args.dataset_test_name
filename_predictions_out = args.out_filename.format(model_name=model_name_or_path.split("/")[-1])
max_predict_samples = args.max_predict_samples
dataset_text_field = args.dataset_text_field
max_new_tokens = args.max_new_tokens

#%%
# mikado config
logger.warning("Using dummy args")
model_name_or_path = "/home/mmordig/reinforcement/HumbleAttemptAtGeneralAI/runs/verbalization/training/overfit_single_nocompl/gpt2"
dataset_name = "/home/mmordig/reinforcement/HumbleAttemptAtGeneralAI/runs/verbalization/datasets/alpha_geo_small_processed"
# dataset_test_name = "test"
dataset_test_name = "train" # for overfitting exp
filename_predictions_out = "/home/mmordig/reinforcement/HumbleAttemptAtGeneralAI/runs/verbalization/predictions/exp_small/gpt2_predictions.txt"
max_predict_samples = 2
dataset_text_field = "text"
max_new_tokens = 70
    

#%%
raw_datasets = load_from_disk(dataset_name)
dataset = raw_datasets[dataset_test_name]
dataset = subset_dataset(dataset, n_samples=max_predict_samples)

model, tokenizer = load_model_for_inference(model_name_or_path)

formatting_func = extract_question_prompt
if tokenizer.chat_template is not None:
    logger.info("Detected chat model, formatting according to chat template")
    # assumes user-assistant roles
    formatting_func = get_question_answer_to_chat_formatter(tokenizer, text_column=None, add_generation_prompt=True)
    
logger.info(f"Generating predictions, writing to file '{filename_predictions_out}'")

max_new_tokens = 50
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, num_return_sequences=2, num_beams=4, do_sample=True, max_new_tokens=max_new_tokens)

#%%
def get_data():
    for row in dataset[args.dataset_text_field]:
        # print(row)
        yield formatting_func(row), row

# using a pipe is faster because GPU works in the background while writing to file
# with open(filename_predictions_out, 'w') as f, redirect_stdout(f):
with nullcontext():
    # for out in pipe(KeyDataset(raw_datasets["test"].map(extract_question), "text")):
    query_it, gt_seqs_it = more_itertools.unzip(get_data())
    query_it = (x for x in query_it) # to convert from map to generator (so hf recognizes it as an iterable)
    for out, gt_seq in tqdm.tqdm(zip(pipe(query_it), gt_seqs_it), total=len(dataset), desc="Making predictions"):
        # returns several candidate sequences
        
        if formatting_func is None:
            question_prompt = None
            for (i, candidate) in enumerate(out):
                txt = candidate["generated_text"]
                if question_prompt is None:
                    question_prompt = extract_question_prompt(txt)
                    print("#"*60)
                    # print(textwrap.fill(question_prompt))
                    # print(textwrap.fill(gt_seq))
                    print(textwrap.fill("Query: " + extract_question_prompt(gt_seq)))
                    print(textwrap.fill("Expected answer: " + extract_answer(gt_seq)))
                else:
                    assert question_prompt == extract_question_prompt(txt)
                answer = extract_answer(txt)
                print("#"*30 + f" Candidate {i+1} " + "#"*30)
                extra = ""
                # if len(tokenizer(answer)["input_ids"]) == max_new_tokens:
                # not perfect because tokenizing with question_prompt may lead to different tokenization
                # safer
                if len(tokenizer(txt)["input_ids"]) - len(tokenizer(question_prompt)["input_ids"]) == max_new_tokens:
                    extra = " <MAX token length exceeded>"
                print("\u23CE\n".join(textwrap.wrap(answer)) + extra) # unicode symbol is enter symbol
        else:
            print("#"*60)
            print(textwrap.fill(gt_seq))
            for (i, candidate) in enumerate(out):
                print("#"*30 + f" Candidate {i+1} " + "#"*30)
                print(textwrap.fill(candidate))
                
        sys.stdout.flush()

logger.info(f"Written to file '{filename_predictions_out}'")
# %%
