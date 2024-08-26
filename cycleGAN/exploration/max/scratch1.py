#%%
import itertools
import json

from model_preparation import AutoEncoderLLM

model = AutoEncoderLLM.from_pretrained("runs/gpt2dec_only/checkpoints")
#%%
model
#%%


def check_keys_in_jsonl(file_path, nrows=None):
    keys = None
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in itertools.islice(file, nrows):
            content = json.loads(line)
            if keys is None:
                keys = set(content.keys())
            assert keys == set(content.keys()), f"Keys do not match: {keys} != {set(content.keys())}"

filename = "/home/mmordig/reinforcement/alphageometry/cycleGAN/runs/datasets/arithmetic/rephrased-nl_fl_dataset_all.jsonl"
check_keys_in_jsonl(filename)#, nrows=10)

# %%

xx = train_datasets[1]
indices = np.array([391, 117, 330, 275, 396, 380, 325, 95]) - len(train_datasets[0])
xx[[i for i in indices if i >= 0]]

#%%


#%%
from transformers import AutoModel
model = AutoModel.from_pretrained("gpt2")
#%%
?model.save_pretrained()
# %%
