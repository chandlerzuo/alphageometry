# converts the dataset from csv to huggingface format (arrow)

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

## Args
raw_dataset_dir = sys.argv[1]
dataset_output_dir = sys.argv[2]
num_proc = 8

logger.info(f"Reading raw dataset from '{raw_dataset_dir}', writing to '{dataset_output_dir}'")

num_input_files = len([f for f in Path(raw_dataset_dir).iterdir() if f.suffix == ".csv"])
logger.info(f"Found {num_input_files} input files")

dataset = load_dataset("csv", data_dir=raw_dataset_dir)["train"]
logger.info(f'Removing columns: {set(dataset.column_names) - {"nl_statement", "fl_statement"}}')
dataset = dataset.remove_columns(set(dataset.column_names) - {"nl_statement", "fl_statement"})
logger.info(f"""Example datapoint: {next(iter(dataset))}""")

rng1, rng2 = np.random.default_rng(seed=1).spawn(2)
dataset = dataset.train_test_split(0.1, generator=rng1)
# make validation set small because it will be loaded at each epoch
dataset_eval = dataset["test"].train_test_split(train_size=min(int(len(dataset["test"]) * 0.5), 400), generator=rng2)
dataset = DatasetDict({
    "train": dataset["train"],
    "val": dataset_eval["train"],
    "test": dataset_eval["test"],
})

logger.info(f"""Dataset example datapoints: {dataset["train"][:2]}""")

logger.info(f"Saving the following dataset: {dataset}")
dataset.save_to_disk(dataset_output_dir, num_proc=8)

logger.info(f"Read raw dataset from '{raw_dataset_dir}', write to '{dataset_output_dir}'")
logger.info("Done")