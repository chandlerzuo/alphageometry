#%%
import itertools
import os
from datasets import load_dataset

dataset_filename = os.path.expanduser("~/reinforcement/alphageometry/LLM_finetuner/runs/verbalization/datasets/alpha_geo_arrow")
dataset = load_dataset(dataset_filename)
# dataset["train"]
# dataset[0]
# num_examples = 100
# fewshot_examples_dataset = dataset["train"].select(range(num_examples))
# train_dataset = dataset["train"].select(range(num_examples, num_examples + 10))

num_train_examples = 100
train_dataset = dataset["train"].take(num_train_examples)
fewshot_examples_dataset = dataset["train"].skip(num_train_examples)

del dataset
#%%
train_dataset[0]
fewshot_examples_dataset["nl_statement"]
#%%
defs_filename = os.path.expanduser("~/reinforcement/alphageometry/defs.txt")

allowed_functions = []
with open(defs_filename, "r") as f:
    for (i, line) in enumerate(f):
        if i % 6 == 0:
            allowed_functions.append(line.strip().split()[0])
allowed_functions_str = ", ".join(allowed_functions)
allowed_functions_str
#%%

# greedy creation of fewshot examples
ex = train_dataset[0]
ex["fl_statement"]
def get_appearing_functions(fl_statement):
    """get all functions that appear in the fl statement through substring search"""
    appearing_fns = []
    for fcn in allowed_functions:
        if (f" {fcn} " in fl_statement):
            appearing_fns.append(fcn)
    return set(appearing_fns)
    
def greedy_get_fewshot_examples(fl_statement, fewshot_examples_dataset, verbose=False):
    """
    Greedily select examples from the dataset that cover all appearing functions in fl_statement
    """
    required_fns = get_appearing_functions(fl_statement)
    idx = 0
    fewshot_indices = [] # list of indices for fewshot
    while len(required_fns) > 0:
        appearing_fns = get_appearing_functions(fewshot_examples_dataset[idx]["fl_statement"])
        new_functions = appearing_fns.intersection(required_fns)
        if len(new_functions) > 0:
            # some new function appeared
            fewshot_indices.append(idx)
            if verbose:
                print(f"Idx {idx} introduces new functions {new_functions}")
        required_fns = required_fns - appearing_fns
        idx += 1
    assert len(required_fns) == 0, "couldn't obtain examples for all functions"

    # return fewshot_indices
    return fewshot_examples_dataset[fewshot_indices]

greedy_get_fewshot_examples(ex["fl_statement"], fewshot_examples_dataset)
#%%
custom_examples_filename = os.path.expanduser("~/reinforcement/alphageometry/LLM_finetuner/custom_geom_examples.txt")
with open(custom_examples_filename, "r") as f:
    lines = f.readlines()
    lines = [line.strip() for line in lines] # replaces '\n' with ''
    lines = [line for line in lines if len(line) > 0 and not line.startswith("#")]
lines
    
#%%
from datasets import Dataset
custom_train_dataset = Dataset.from_dict({
    "nl_statement": [l for (i, l) in enumerate(lines) if i % 2 == 0],
    "fl_statement": [l for (i, l) in enumerate(lines) if i % 2 == 1]
})
custom_train_dataset
#%%

train_dataset[0]

system_message = f"""\
Your task is to translate a geometry problem in natural language to formal language.
When you introduce a point that satisfies several constraints, separate these by commas. 
Saying that point P = Q is not allowed.
You are allowed to use the following functions: {allowed_functions_str}. 
You are not allowed to refer to any other functions.
You should only output the geometric formal statement, nothing more.
Here are some examples:"""

example_template = """\
Natural: {nl_statement}
Formal: {fl_statement}"""
def create_fewshot_examples(fewshot_examples_dataset, question, answer=""):
    """create prompt with few shot examples followed by question and (possibly empty) answer"""
    return "\n".join(
        example_template.format(nl_statement=nl_statement, fl_statement=fl_statement) for 
        (nl_statement, fl_statement) in itertools.chain(
            zip(fewshot_examples_dataset["nl_statement"], fewshot_examples_dataset["fl_statement"]),
            [(question, answer)]
        )
    )
    

def create_prompt(example, fewshot_examples_dataset):
    """create prompt by combining system message with few shot examples"""
    main_prompt = create_fewshot_examples(fewshot_examples_dataset, example["nl_statement"], answer="") #answer=example["fl_statement"])
    # print(examples)
    message = system_message + "\n" + main_prompt
    return message, example["fl_statement"]
message, label = create_prompt(train_dataset[0], fewshot_examples_dataset.take(2))
message, label = create_prompt(train_dataset[1], fewshot_examples_dataset.take(2))

ex = train_dataset[0]
ex = custom_train_dataset[2]
message, label = create_prompt(ex, greedy_get_fewshot_examples(ex["fl_statement"], fewshot_examples_dataset, verbose=True))
print()
print(message)
print()
print("Expected answer:", label)

#%%
from verb.verbalize import IndependentStatementVerbalization
verbalizer = IndependentStatementVerbalization(None)
fl_statement = "A B C = triangle A B C"
nl_prob = verbalizer.problem_fl_2_nl(fl_statement)

# %%
# assert label == "A B C = ieq_triangle A B C; D E F G H = pentagon D E F G H; I = angle_equal_on_line I F H A B G A E; J = on_dia J E D"
# pred_label = "A B C = ieq_triangle A B C; D E F G H = pentagon D E F G H; I = angle_equal_on_line I F H A B G A E; J = on_dia J E D"
# pred_label = "A B C D = eq_trapezoid A B C D; E = free E; F = lc_tangent F C A; G H I J = incenter2 G H I J E C B; K = angle_mirror K F D J" # train idx 0
# pred_label = "A B C = ieq_triangle A B C; D E F = risos D E F; G = psquare G A E; H = intersection_cc H C G E; I J = tangent I J C B F; K = intersection_tt K J D I E A G" # train idx 1
# pred_label = "A B C D = quadrangle A B C D; E = orthocenter E A B C; F = lc_tangent F D C" # train idx 2
pred_label = "A B C = triangle A B C; O = circumcenter O A B C; G = centroid G A B C; O = G; A B C = ieq_triangle A B C"
print(label)
print(pred_label)
print("Correct:", label == pred_label)

#%%
xx = "\n".join(train_dataset["fl_statement"])
xx.find(",")

# todo: check whether gpt can restrict to only known keywords, select few shot examples greedily
# formalize online problems and see how it performs
# have gpt Try again if output is garbage