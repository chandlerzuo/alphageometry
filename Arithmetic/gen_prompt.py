import argparse
import pandas as pd
import inspect
import defs
import random
import sympy as sp


def acquire_symbols():
    functions = {}
    """Acquire symbols from the defs module, instantiating callable classes."""
    for name, obj in inspect.getmembers(defs, inspect.isclass):
        instance = obj()  # Instantiate each class
        args = inspect.signature(instance.__call__).parameters
        arity = len(args)
        functions[name] = (arity, instance)

    return functions


def get_symbolic_func_exp(func, num_args):
    symbol_names = [chr(i) for i in range(ord('A'), ord('Z')+1)]
    variables = sp.symbols(' '.join(symbol_names[:num_args]))
    return func(*variables), symbol_names[:num_args]


def main():
    seed = 43
    random.seed(seed)  # Python's random module seed

    # Setup the argument parser
    parser = argparse.ArgumentParser(description="Process CSV files and prepare a prompt.")
    parser.add_argument('--demo_csv', default='../arith_test.csv', type=str, help='Path to the demo CSV file')
    parser.add_argument('--test_csv', default='../arith_test.csv', type=str, help='Path to the test CSV file')
    parser.add_argument('-n', default=50, type=int, help='Number of random examples to include in the prompt')

    instruction = 'Your job is to convert arithmetic questions from natural language descriptions to their formal ' \
                  'version. Here are some examples. At the end there will be a test description and your job is to ' \
                  'give me the formal version. Nothing else. \n'
    available_funcs = acquire_symbols()
    func_definitions = 'You have the following funcs available to you. You can only use these functions. ' \
                       'Nothing else.\n'
    for func_name, (num_args, func) in available_funcs.items():
        func_exp, args = get_symbolic_func_exp(func, num_args)
        func_definitions += f'{func_name}({", ".join(args)}) = {func_exp}\n'

    func_definitions += 'Therefore, your job is to use this functions to construct the arithmetic expression being ' \
                        'described in the natural language description. Imagine the expression being described is ' \
                        'x^2 -y + 2*Z, then you can express it as A = dissolution(x, y); B = calcination(A, z); B?' \
                        'Similarly, x + 2x^2 - 2y can be expressed as A = dissolution(x, y); B = calcination(x, A); B?.' \
                        'Think step by step algebraically how to get the expression while using the elementary ' \
                        'expressions given'

    # Parse command line arguments
    args = parser.parse_args()

    # Read the CSV files
    demo_df = pd.read_csv(args.demo_csv)
    test_df = pd.read_csv(args.test_csv)

    # Ensure the data contains expected columns
    if not {'formal', 'natural', 'answer'}.issubset(demo_df.columns) or not {'formal', 'natural', 'answer'}.issubset(test_df.columns):
        raise ValueError("CSV files must contain 'formal', 'natural', and 'answer' columns")

    # Select 'n' random samples from the demo dataframe
    if args.n > len(demo_df):
        raise ValueError("The number of samples requested exceeds the number of available entries in the demo CSV.")

    random_samples = demo_df.sample(n=args.n)

    # Add the last description from the test dataframe
    last_test_entry = test_df.sample(n=1, random_state=seed).iloc[0]

    # Build the prompt string
    prompt = ""
    for _, row in random_samples.iterrows():
        if last_test_entry['natural'] != row['natural'] and last_test_entry['formal'] != row['formal']:
            prompt += f"description: \n{row['natural']}\nformal: \n{row['formal']}\n\n"

    prompt += f"description: \n{last_test_entry['natural']}\nformal: \n"

    print(f'Row number = {last_test_entry.name}, answer = {last_test_entry["answer"]}')

    # Print the prompt to standard output
    print(instruction + func_definitions + prompt)


if __name__ == "__main__":
    main()
