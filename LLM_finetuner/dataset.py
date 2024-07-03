import functools
import textwrap
from typing import List

## format single prompt

def replace_words_with_tokens(fl_statement, geom_tokens: List[str]):
    """
    Replace geometry words with tokens
    geom_tokens need to be sorted by length to avoid replacing subtokens!
    """
    # global _checked_sorted
    # if not _checked_sorted:
    #     assert geom_tokens == sorted(geom_tokens, key=lambda x: -len(x))
    #     _checked_sorted = True

    for geom_token in geom_tokens:
        fl_statement = fl_statement.replace(f"{geom_token} ", f"[{geom_token}] ")
    return fl_statement

SYSTEM_MESSAGE = textwrap.dedent(f"""\
    You are an expert in translating geometry problems from natural language into formal language.
    It is crucial to form syntactically valid statements in the formal language that correspond 
    exactly to its natural language description.
    You should only output the formal language description of the question, not the solution or anything else.
    """).strip().replace("\n", " ")

JSON_SCHEMA = textwrap.dedent("""\
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "array",
  "items": {
    "type": "object",
    "properties": {
      "fun": {
        "type": "string"
      },
      "args": {
        "type": "array",
        "items": {
          "type": "string"
        }
      },
      "defines": {
        "type": "array",
        "items": {
          "type": "string"
        }
      }
    },
    "required": ["fun", "args", "defines"]
  }
}""")
SYSTEM_MESSAGE_JSON = textwrap.dedent(f"""\
You are an expert in translating geometry problems from natural language to JSON.
You should only output the formal language description of the question, not the solution or anything else.
Use the following JSON schema:
{JSON_SCHEMA}""")#.strip().replace("\n", " ")

def create_chat(question, answer, system_message=None):
    chat = []
    if system_message:
        chat.append({
            "role": "system",
            "content": system_message
        })
    # chat += [
    #     {
    #         "role": "user",
    #         "content": question
    #     },
    #     {
    #         "role": "assistant",
    #         "content": answer
    #     }
    # ]
    chat.append({
        "role": "user",
        "content": question
    })
    if answer:
        chat.append({
            "role": "assistant",
            "content": answer
        })
    return chat
    
def is_chat_model(tokenizer):
    # return hasattr(model, "chat")
    return tokenizer.chat_template is not None
    # return getattr(tokenizer, "chat_template", None) is not None # not working since creates default_chat_template automatically

DEFAULT_CHAT_TEMPLATE = textwrap.dedent("""\
{% for message in messages %}
### {{ message['role'] }}: {{ message['content'] }}{% if not loop.last %}{{ '  ' }}{% endif %}{% endfor %}{%if add_generation_prompt %}  ### assistant: {% endif %}""")
def format_question_answer_single(tokenizer, question, answer, system_message=None, end=""):
    chat = create_chat(question, answer, system_message=system_message)
    if answer:
        add_generation_prompt = False
    else:
        add_generation_prompt = True
        end = ""
    return tokenizer.apply_chat_template(
        chat, 
        chat_template=None if is_chat_model(tokenizer) else DEFAULT_CHAT_TEMPLATE, tokenize=False,
        add_generation_prompt=add_generation_prompt
    ) + end
    
import json

def convert_formalalphageom_to_json(expression, pretty=False):
    """
    Splits by semicolon, then by equal sign, then args and outputs by whitespace
    """
    parts = expression.split("; ")
    result = []

    for part in parts:
        defines, func_call = part.split(" = ")
        func_name, *args = func_call.split()
        defines_list = defines.split()

        func_dict = {
            "fun": func_name,
            "args": args,
            "defines": defines_list
        }

        result.append(func_dict)

    # return json.dumps(result)
    return json.dumps(result, indent=2) if pretty else result

# # Example usage
# expression = 'A B C = ieq_triangle A B C; D E F = risos D E F; G = psquare G A E; H = intersection_cc H C G E; I J = tangent I J C B F; K = intersection_tt K J D I E A G'
# def show_transformation(expression):
#     print(expression)
#     print("Translation:")
#     json_output = convert_formalalphageom_to_json(expression, pretty=True)
#     # json_output = convert_formalalphageom_to_json(expression, pretty=False)
#     print(json_output)
# show_transformation(expression)
# # print(convert_to_json(dataset["train"][1]["fl_statement"]))


## Dataset operations
# END_TOKEN = "[END]"
def format_question_answer_batch_for_training(batch, tokenizer, col_name="text", **kwargs):
    """END_TOKEN should be added to the vocabulary"""
    return {col_name: [
        format_question_answer_single(tokenizer, question, answer, **kwargs) 
        for (question, answer) in zip(batch["nl_statement"], batch["fl_statement"])
    ]}

def format_question_answer_batch_for_generation(batch, tokenizer, **kwargs):
    # no end, answer=""
    return {"text": [
        format_question_answer_single(tokenizer, question, answer="", **kwargs) 
        for question in batch["nl_statement"]
    ]}
    
# def tokenize_text_col(batch):
#     return tokenizer(batch["text"])
# def get_label_ids(batch):
#     # ignoring attention masks for labels since they already come from input, no shift necessary
#     return {"labels": tokenizer(batch["gt_text"])["input_ids"]}


def create_inputs_labels_for_generation(dataset, tokenizer, **kwargs):
    # adds "text" column with question, but no answer
    dataset = dataset.map(
        functools.partial(format_question_answer_batch_for_generation, tokenizer=tokenizer, **kwargs), 
        batched=True
    )
    # "labels = input_ids" normally, see gpt2 docs; here labels correspond to full prompt with answer
    # adds "labels" column with question and answer
    dataset = dataset.map(
        functools.partial(format_question_answer_batch_for_training, tokenizer=tokenizer, col_name="labels", **kwargs), batched=True
    )
    # dataset = dataset.map(get_label_ids, batched=True)
    return dataset
    
def tests():
    from transformers import AutoTokenizer
    
    assert len(create_chat("my_question", "my_answer", "system_msg")) == 3
    assert len(create_chat("my_question", "my_answer", None)) == 2
    
    
    from datasets import Dataset, DatasetDict
    dataset = DatasetDict({
    "train": Dataset.from_dict({
            "fl_statement": ["A B C = triangle A B C", "A B D = triangle A B D"], 
            "nl_statement": ["ABC is a triangle", "ABD is a triangle"]
        }),
        "val": Dataset.from_dict({
            "fl_statement": ["A B E = triangle A B E", "A B F = triangle A B F"], 
            "nl_statement": ["ABE is a triangle", "Triangle formed by A, B and F"]
        }),
    })
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    dataset["train"] = dataset["train"].map(
        functools.partial(format_question_answer_batch_for_training, tokenizer=tokenizer), 
        batched=True
    )
    dataset["val"] = create_inputs_labels_for_generation(dataset["val"], tokenizer=tokenizer)
    # dataset = dataset.map(tokenize_text_col, batched=True)

    # def decode_labels(batch):
    #     return {"labels_decoded": tokenizer.batch_decode(batch["labels"])}
    # dataset["val"] = dataset["val"].map(decode_labels, batched=True)
    dataset["train"][:2], dataset["val"][:2]
    
    
    # for model_name in ["gpt2", "meta-llama/Llama-2-7b-chat-hf"]:
    #     # assert is_chat_model(model_name)
    #     tokenizer = AutoTokenizer.from_pretrained(model_name)
    #     print(f"tokenization for model {model_name}:")
    #     print(format_question_answer_single(tokenizer, "my_question", "my_answer", system_message="system_msg"))
    #     print(format_question_answer_single(tokenizer, "my_question", "my_answer", system_message=None))
        
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    assert format_question_answer_single(tokenizer, "my_question", "my_answer") == '### user: my_question  ### assistant: my_answer'
    assert format_question_answer_single(tokenizer, "my_question", "") == '### user: my_question  ### assistant: '

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    assert format_question_answer_single(tokenizer, "my_question", "my_answer") == '<s>[INST] my_question [/INST] my_answer </s>'
    assert format_question_answer_single(tokenizer, "my_question", "") == '<s>[INST] my_question [/INST]'
    
    
    
    assert replace_words_with_tokens("A B C = eq_triangle A B C; C D E = ieq_triangle C D E", ["ieq_triangle", "eq_triangle"]) == \
    "A B C = [eq_triangle] A B C; C D E = [ieq_triangle] C D E"
    
    # extra_tokens_file = Path(os.path.expanduser("~/reinforcement/alphageometry/assets/def-patterns-desc.yml"))
    # def_to_desc = yaml.safe_load(extra_tokens_file.read_text())
    # # eq_triangle, ieq_triangle, so replace by longest first
    # geom_tokens = sorted(list(def_to_desc.keys()), key=lambda x: -len(x))
    # fl_statement = dataset["train"][0]["fl_statement"]

    # replace_words_with_tokens(fl_statement, geom_tokens)
    
if __name__ == "__main__":
    tests()