import logging
import sys
from typing import Dict
import torch
from transformers.trainer_utils import get_last_checkpoint
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from peft import AutoPeftModelForCausalLM

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def subset_dataset(dataset, n_samples=None):
    if (n_samples is not None) and (n_samples > 0):
        if n_samples < 1:
            # fraction
            dataset = dataset.select(range(int(len(dataset) * n_samples)))
        else:
            dataset = dataset.select(range(min(len(dataset), n_samples)))
    return dataset
    
def infer_checkpoint(checkpoint_dir):
    """first try getting latest checkpoint inside directory;
    if None, it means we passed an actual checkpoint dir, so return that
    """
    x = get_last_checkpoint(checkpoint_dir) # regexp checking for "checkpoint-<number>"
    return checkpoint_dir if x is None else x

def get_model_name_from_name_or_path(model_name_or_path):
    model_name_or_path = str(model_name_or_path)
    if model_name_or_path.endswith("/"):
        model_name_or_path = model_name_or_path[:-1]
    return model_name_or_path.split("/")[-1]

def load_model_for_inference(model_checkpoints_dir):
    model_name_or_path = infer_checkpoint(model_checkpoints_dir)
    logger.info(f"Loading model from '{model_name_or_path}'")
    # model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    # bnb_config = None
    # fails when vocab size changes
    # https://discuss.huggingface.co/t/loading-peft-model-from-checkpoint-leading-into-size-missmatch/71944
    # model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto", quantization_config=bnb_config)
    model = AutoPeftModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto", quantization_config=bnb_config)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path) # don't add eos_token since in generation mode
    tokenizer.padding_side = "left" # for auto-regressive generation
    logger.info(f"Loaded model")
    
    return model, tokenizer

def set_pad_token_if_not_set(model, tokenizer):
    if (tokenizer.pad_token is None) or (tokenizer.eos_token == tokenizer.pad_token):
        # padding token should be distinct from eos_token since it gets ignored in the loss, so
        # the model does not learn to generate the eos token
        # following works well for llama-2 at least
        # see https://discuss.huggingface.co/t/llama2-pad-token-for-batched-inference/48020/3
        tokenizer.add_special_tokens({'pad_token': '[PAD]'}) # this fails when using gpt2 with device assertion
        # tokenizer.pad_token = "[PAD]"
        model.resize_token_embeddings(len(tokenizer))
        
    logger.info(f"Using padding token '{tokenizer.pad_token}'")

def add_new_tokens_with_average_init(model, tokenizer, def_to_desc: Dict[str, str]):
    """
    For each (key, value) of def_to_desc, add "key" as a new token and initialize to average 
    embedding of tokens in value
    
    Only initializes if the tokens are not already present. If only some are present, it raises an error.
    
    Skips None's in values
    """
    prev_num_tokens = len(tokenizer)

    tokens = list(def_to_desc.keys())
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    def_to_desc = {token: def_to_desc[token] for (token, id) in zip(tokens, token_ids) if id == tokenizer.unk_token_id}

    if len(def_to_desc) == 0:
        logger.warning("All tokens are already part of the tokenizer, not modifying them")
        return
    
    logger.warning(f"The following tokens are already known, ignoring them: {[token for (token, id) in zip(tokens, token_ids) if id != tokenizer.unk_token_id]}")
    tokenizer.add_tokens([f"{defn}" for defn in def_to_desc.keys()], special_tokens=False)
    assert len(tokenizer) == prev_num_tokens + len(def_to_desc)

    logger.info(f"Vocabulary size: {len(tokenizer)}")
    model.resize_token_embeddings(len(tokenizer))
    input_embeddings = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()

    for (new_token, token_desc) in def_to_desc.items():
        if token_desc is None: continue
        avg_embedding = input_embeddings(
            tokenizer(token_desc, return_tensors="pt")["input_ids"][0]
        ).mean(dim=0)
        
        token_id = tokenizer.convert_tokens_to_ids(new_token)
        assert token_id != tokenizer.unk_token_id
        # print(token_id, input_embeddings.weight.data.shape)
        input_embeddings.weight.data[token_id] = avg_embedding
        output_embeddings.weight.data[token_id] = avg_embedding
        # llama2 without peft does not have weight tying
        # assert torch.all(input_embeddings.weight.data[token_id] == output_embeddings.weight.data[token_id]), "no weight tying, why?" # may try to set output embeddings
    
def setup_logging():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO,
    )
    
def main():

    def test1():
        from transformers import AutoTokenizer
        model_name_or_path = "gpt2"
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
        print(tokenizer.eos_token)
        print(tokenizer.pad_token)
    
    def test2():
        from transformers import AutoTokenizer, AutoModelForCausalLM
        model_name_or_path = "gpt2"
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path)

        import yaml
        from pathlib import Path
        import os
        # # for creating this file
        # patterns_filename = Path(os.path.expanduser("~/reinforcement/alphageometry/assets/def-patterns.yml"))
        # filename_out = open(os.path.expanduser("~/reinforcement/alphageometry/assets/def-patterns-desc-simple.yml"), "w")
        # filename_out = None
        # print("\n".join(list(f"{x}: {x}" for x in yaml.safe_load(patterns_filename.read_text()).keys())), file=filename_out)
        patterns_desc_filename = Path(os.path.expanduser("~/reinforcement/alphageometry/assets/def-patterns-desc.yml"))

        def_to_desc = yaml.safe_load(patterns_desc_filename.read_text())
        print(f"Tok len: {len(tokenizer)}")    
        add_new_tokens_with_average_init(model, tokenizer, def_to_desc)
        print(f"Tok len: {len(tokenizer)}")
        
    print("Test1")
    test1()
    print("Test2")
    test2()
    
if __name__ == "__main__":
    main()
    
    