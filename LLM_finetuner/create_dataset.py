"""
Create the dataset by parsing the csv files and creating "text" field to input to the SFT script.

It is recommended to have few input csv files in the directory, otherwise it is inefficient.
"""

import logging
from pathlib import Path
import sys
from datasets import load_dataset, DatasetDict
import numpy as np

import os, sys; sys.path.append(os.path.join(os.path.dirname(__file__), ".")) #todo
from utils import setup_logging
# from transformers import AutoTokenizer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
setup_logging()


# mikado
# logger.warning("Using dummy args")
# raw_dataset_dir = "/home/mmordig/reinforcement/datasets/alpha_geo_small"
# dataset_output_dir = "/home/mmordig/reinforcement/HumbleAttemptAtGeneralAI/geometry_translation/new/runs/verb_dataset"

# small dataset
# raw_dataset_dir =    "/fast/mmordig/general_ai_rl/alphageom_project/verbalization/datasets/alpha_geo_small"
# dataset_output_dir = "/fast/mmordig/general_ai_rl/alphageom_project/verbalization/datasets/alpha_geo_small_processed"
# full dataset
# raw_dataset_dir =    "/fast/mmordig/general_ai_rl/alphageom_project/verbalization/datasets/alpha_geo_fewer_chunks"
# dataset_output_dir = "/fast/mmordig/general_ai_rl/alphageom_project/verbalization/datasets/alpha_geo_processed"


raw_dataset_dir = sys.argv[1]
dataset_output_dir = sys.argv[2]

# trl supports chat
# if mode == "chat":
#     pass
# else:
#     assert tokenizer.chat_template is None
# # model_name_or_path = sys.argv[3]
# tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

num_proc = 8
# cpu_count = 
# logger.info(f"Detected {cpu_count} CPUs")
# num_proc = max(1, cpu_count-1)

logger.info(f"Read raw dataset from '{raw_dataset_dir}', write to '{dataset_output_dir}'")

num_input_files = len([_ for _ in Path(raw_dataset_dir).iterdir()])
logger.info(f"Found {num_input_files} input files")

dataset = load_dataset("csv", data_dir=raw_dataset_dir)#, nrows=10)
logger.info(f"""Example datapoint: {next(iter(dataset["train"]))}""")

# def formatting_func(row):
#     question = row["nl_statement"]
#     answer = row["fl_statement"]
#     return {"text": f"### Question: {question} ### Answer: {answer}"}
# dataset = dataset.map(formatting_func)

def formatting_func_batched(batch):
    questions = batch["nl_statement"]
    answers = batch["fl_statement"]
    return {"text": [f"### Question: {question} ### Answer: {answer}" for (question, answer) in zip(questions, answers)]}

dataset = dataset.map(formatting_func_batched, batched=True)# batching already sufficiently efficient, num_proc=num_proc)
# col_names = dataset["train"].take(1)._resolve_features().column_names
# logger.info(f"Detected column names {col_names}")
col_names = dataset["train"].column_names # not working for Iterable dataset
dataset = dataset.remove_columns(set(col_names) - {"text"})

rng1, rng2 = np.random.default_rng(seed=1).spawn(2)
dataset = dataset["train"].train_test_split(0.1, generator=rng1)
# make validation set small because it will be loaded at each epoch
dataset_eval = dataset["test"].train_test_split(train_size=min(int(len(dataset["test"]) * 0.5), 400), generator=rng2)
dataset = DatasetDict({
    "train": dataset["train"],
    "val": dataset_eval["train"],
    "test": dataset_eval["test"],
})

logger.info(f"""Dataset example datapoints: {dataset, dataset["train"][:2]}""")

# num_shards = min(num_input_files, 1024)
dataset.save_to_disk(dataset_output_dir, num_proc=8)#, num_shards={split: num_shards for split in dataset.keys()})

logger.info(f"Read raw dataset from '{raw_dataset_dir}', write to '{dataset_output_dir}'")
logger.info("Done")