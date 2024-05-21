"""
Make predictions with the model and write to a file (along with the ground-truth)

"""

#%%
from dataclasses import asdict, dataclass, field
from typing import Optional
import torch
import more_itertools
import logging
import sys
import textwrap
import collections

import tqdm
from transformers import pipeline, HfArgumentParser
from datasets import load_from_disk
import gradio as gr
from contextlib import nullcontext, redirect_stdout

from LLM_finetuner.question_answer_utils import extract_answer, extract_question_prompt, get_question_answer_to_chat_formatter
from LLM_finetuner.utils import load_model_for_inference, setup_logging, subset_dataset, get_model_name_from_name_or_path

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
filename_predictions_out = args.out_filename.format(model_name=get_model_name_from_name_or_path(model_name_or_path), **asdict(args))
max_predict_samples = args.max_predict_samples
dataset_text_field = args.dataset_text_field
max_new_tokens = args.max_new_tokens

#%%
# mikado config
# logger.warning("Using dummy args")
# # model_name_or_path = "/home/mmordig/reinforcement/alphageometry/LLM_finetuner/runs/verbalization/training/overfit_single_nocompl/gpt2"
# # model_name_or_path = "/home/mmordig/reinforcement/alphageometry/LLM_finetuner/runs/verbalization/training/overfit_single_nocompl/gpt2_2ex"
# # model_name_or_path = "/home/mmordig/reinforcement/alphageometry/LLM_finetuner/runs/verbalization/training/overfit_single_nocompl/gpt2_withpeft"
# model_name_or_path = "/home/mmordig/reinforcement/alphageometry/LLM_finetuner/runs/verbalization/training/failed_quote/gpt2_1000ex_peftFalse\'/"
# dataset_name = "/home/mmordig/reinforcement/alphageometry/LLM_finetuner/runs/verbalization/datasets/alpha_geo_small_processed"
# # dataset_test_name = "test"
# dataset_test_name = "train" # for overfitting exp
# filename_predictions_out = "/home/mmordig/reinforcement/alphageometry/LLM_finetuner/runs/verbalization/predictions/exp_small/gpt2_predictions.txt"
# # max_predict_samples = 2
# max_predict_samples = 1
# dataset_text_field = "text"
# max_new_tokens = 70
    

#%%
logger.info(f"Generating predictions, writing to file '{filename_predictions_out}'")

raw_datasets = load_from_disk(dataset_name)
dataset = raw_datasets[dataset_test_name]
dataset = subset_dataset(dataset, n_samples=max_predict_samples)

model, tokenizer = load_model_for_inference(model_name_or_path)

is_chat_model = tokenizer.chat_template is not None
if is_chat_model:
    logger.info("Detected chat model, formatting according to chat template")
    # assumes user-assistant roles
    prompt_extraction_function = get_question_answer_to_chat_formatter(tokenizer, text_column=None, add_generation_prompt=True)
else:
    prompt_extraction_function = extract_question_prompt
    
def extract_extra_cols(batch):
    return {
        "question_prompt": [prompt_extraction_function(item) for item in batch[dataset_text_field]],
        "answer_only": [extract_answer(item) for item in batch[dataset_text_field]],
    }
dataset = dataset.map(extract_extra_cols, batched=True)
logger.info(f"Example datapoint: {dataset[0]}")

# use_cache to avoid recomputing hidden states, see https://discuss.huggingface.co/t/what-is-the-purpose-of-use-cache-in-decoder/958
# max_new_tokens = 70
pipe = pipeline(
    "text-generation", model=model, tokenizer=tokenizer, 
    num_return_sequences=2, num_beams=4, do_sample=True, max_new_tokens=max_new_tokens, use_cache=True,
    return_full_text=False, # answer only
    num_workers=2, 
    # batch_size=2 # triggers a cuda device-side error, maybe related to https://github.com/huggingface/transformers/issues/22546
)

#%%

break_nicely = lambda x: "\u23CE\n".join(textwrap.wrap(x)) # symbol "⏎" for line breaks
# using a pipe/dataset is faster because GPU works in the background while writing to file
logger.info(f"Writing to file '{filename_predictions_out}'")
with open(filename_predictions_out, 'w') as f, redirect_stdout(f):
# with nullcontext():
    for (out, question_prompt, gt_answer) in tqdm.tqdm(zip(pipe(dataset["question_prompt"]), dataset["question_prompt"], dataset["answer_only"])):
        print("#"*80)
        print("Query: ")
        print(break_nicely(question_prompt))
        print("Expected answer: ")
        print(break_nicely(gt_answer))
        # strips whitespace because generated text has leading and trailing whitespace
        out_counted = collections.Counter([candidate["generated_text"].strip() for candidate in out])
        gt_answer = gt_answer.strip()
        print(f"Number of candidates that are equal to expected: {out_counted.get(gt_answer, 0)}")
        print(f"Number of candidates that begin with expected:", sum(out_counted[key] for key in out_counted if key.startswith(gt_answer)))
        # for (i, candidate) in enumerate(out):
        #     candidate_text = candidate["generated_text"]
        for (i, (candidate_text, count)) in enumerate(out_counted.items()):
            # logger.info(f"Generated text: {candidate_text}")
            # answer = extract_answer(candidate_text)
            answer = candidate_text
            print("#"*20 + f" Candidate {i+1} (appears {count} times) " + "#"*20)
            extra = ""
            # not perfect because tokenizing with question_prompt may lead to different tokenization
            if len(tokenizer(answer)["input_ids"]) == max_new_tokens:
                extra = " <MAX token length exceeded>"
            print(break_nicely(answer) + extra)
        # sys.stdout.flush()

logger.info(f"Wrote to file '{filename_predictions_out}'")
# %%
