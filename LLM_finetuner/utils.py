import logging
import sys
import torch
from transformers.trainer_utils import get_last_checkpoint
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
# from peft import AutoPeftModelForCausalLM

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def subset_dataset(dataset, n_samples=None):
    if n_samples is not None:
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

def load_model_for_inference(model_checkpoints_dir):
    model_name_or_path = infer_checkpoint(model_checkpoints_dir)
    logger.info(f"Loading model from '{model_name_or_path}'")
    # model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    # bnb_config = None
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto", quantization_config=bnb_config)
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

def setup_logging():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO,
    )
    
def main():

    from transformers import AutoTokenizer

    model_name_or_path = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
    print(tokenizer.eos_token)
    print(tokenizer.pad_token)
    
if __name__ == "__main__":
    main()
    
    