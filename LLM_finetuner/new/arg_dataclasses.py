from dataclasses import dataclass, field
from typing import Optional, Union
from transformers import GenerationConfig

# class HfParserWithConfig:
#     """
#     HfArgumentsParser, additionally supports reading a '--config' which can be overwritten by explicitly
#     passing arguments on the CLI
    
#     Similar to TrlParser, but not specific to their arg classes
#     """
#     def parse_args_and_config(self):
#         args, = self.parser.parse_args_into_dataclasses()
#         return args

# @dataclass
# class ConfigArgument:
#     config: Optional[str] = field(default=None, metadata={"help": "path to the config file"})
    
@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": ("The model checkpoint for weights initialization.")},
    )
    use_peft: bool = field(
        default=False,
        metadata={"help": ("Whether to use PEFT or not for training.")},
    )
    
@dataclass
class SFTArguments:
    # model_name: str
    # resume_path
    # dataset_path
    dataset_name: str = field(default="timdettmers/openassistant-guanaco", metadata={"help": "the dataset name"})
    dataset_train_name: str = field(default="train", metadata={"help": "the name of the training set of the dataset"})
    dataset_eval_name: str = field(default="test", metadata={"help": "the name of the evaluation set of the dataset"})
    
    eval_generation_config: Optional[GenerationConfig] = field(
        default=None,
        metadata={"help": "generation config for evaluation"}
    )
    
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