#%%
import numpy as np
import torch
from transformers import DataCollatorWithPadding

from datasets import Dataset, load_dataset, DatasetDict

filename = "runs/datasets/arithmetic/nl_fl.csv"
seed = 42
test_size = 0.1

def prepare_data(filename, test_size=0.1, seed=None, device=None):
    dataset = load_dataset('csv', data_files=filename, split='train')
    print(f"Dataset loaded, of size {len(dataset)}")
    dataset = dataset.rename_columns(({"fl_statement": "formal", "nl_statement": "natural"}))
    dataset = dataset.with_format("torch", device=device)

    generator = np.random.default_rng(seed)
    assert 0 <= 2 * test_size <= 1, f"got {test_size}"
    temp_ds = dataset.train_test_split(test_size=0.1 * 2, shuffle=True, generator=generator)
    train_ds = temp_ds["train"]
    temp_ds2 = temp_ds["test"].train_test_split(test_size=0.5, shuffle=True, generator=generator)
    val_ds = temp_ds2["train"]
    test_ds = temp_ds2["test"]
    
    return DatasetDict({"train": train_ds, "validation": val_ds, "test": test_ds})

device = None
ds = prepare_data(filename, test_size, seed, device)
# %%
ds["train"][2]