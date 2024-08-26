from pathlib import Path
from transformers import PreTrainedModel
from transformers.utils import is_peft_available
from peft import PeftModel

def save_model(model, output_dir, state_dict=None, safe_serialization=True):
    # see HF Trainer._save
    supported_classes = (
        (PreTrainedModel,) if not is_peft_available() else (PreTrainedModel, PeftModel)
    )
    assert isinstance(model, supported_classes)

    model.save_pretrained(
        output_dir, state_dict=state_dict, safe_serialization=safe_serialization
    )


def save_tokenizer(tokenizer, output_dir):
    tokenizer.save_pretrained(output_dir)


def create_dir(dir):
    """create dir and return it"""
    dir = Path(dir)
    dir.mkdir(parents=True, exist_ok=True)
    return dir


def load_pretrained_config_from_scratch(*args, auto_model_class=None, **kwargs):
    """
    Load config of pretrained model, but initialize weights from scratch

    # todo: add test
    from transformers import GPT2LMHeadModel
    assert isinstance(load_pretrained_config_from_scratch("gpt2", auto_model_class=AutoModelForCausalLM), GPT2LMHeadModel)
    """
    from transformers import AutoConfig, AutoModel

    if auto_model_class is None:
        auto_model_class = AutoModel

    config = AutoConfig.from_pretrained(
        *args, **kwargs
    )  # it seems we don't see AutoConfigFor* here
    config._name_or_path = "invalid"  # just to be sure
    model = auto_model_class.from_config(config)
    return model

def freeze_params(model):
    for param in model.parameters():
        param.requires_grad = False
        
def is_frozen(model):
    return all(param.requires_grad == False for param in model.parameters())

def get_comma_separated_strings(lst):
    # get_comma_separated_strings(["aaa", "bbb", "ccc"])
    return ', '.join(f'"{x}"' for x in lst)

def get_hostname():
    import socket
    return socket.gethostname()