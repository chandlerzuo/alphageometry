import ast
import importlib
import inspect
import random
import uuid


class SymArithmeticProbGen:
    def __init__(self, seed=None, depth=2):
        self.depth = depth
        if seed is not None:
            random.seed(seed)  # Set the random seed if provided
        # Dynamically load the module
        self.defs = importlib.import_module('defs')
        self.functions = {}
        self.acquire_symbols()
        self.function_names = list(self.functions.keys())
        self.tree = None
        self.code = None
        self.code_lines = []
        self.variable_counter = 0

    def estimate_return_count(self, func, arity):
        """Estimate the number of parameters returned"""
        ret_obj = func(*[random.randint(1, 10) for _ in range(arity)])
        if not (isinstance(ret_obj, int) or isinstance(ret_obj, float)):
            raise ValueError(f"Function {func.__name__} must return an integer or float")
        else:
            return 1

    def acquire_symbols(self):
        # Automatically create a dictionary of functions with their arities
        for name, func in inspect.getmembers(self.defs, inspect.isfunction):
            # Inspect function arguments, ignoring 'self' for class methods if any
            args = inspect.signature(func).parameters
            arity = len(args)
            ret_count = self.estimate_return_count(func, arity)
            self.functions[name] = (arity, ret_count, func)

    def generate_expression(self, depth=0):
        """Recursively generates a random arithmetic expression using AST nodes."""
        if depth > self.depth:
            return ast.Constant(value=random.randint(1, 10))

        # Randomly select a function
        func_name = random.choice(self.function_names)
        func = self.functions[func_name][2]

        # Recursively generate expressions for arguments
        args = [self.generate_expression(depth + 1) for _ in range(len(inspect.signature(func).parameters))]

        # Create a Call node for the selected function
        call_node = ast.Call(func=ast.Name(id=func_name, ctx=ast.Load()), args=args, keywords=[])

        return call_node

    def to_code(self):
        """Converts an AST node to code"""
        self.tree = self.generate_expression()
        self.code = ast.unparse(self.tree)
        return self.code

    def unique_var(self):
        """Generate a unique variable name in alphabetical order followed by numerical suffixes."""
        # Calculate how many times the alphabet has been fully used
        suffix = self.variable_counter // 26
        # Calculate position in the alphabet
        letter = chr(ord('A') + self.variable_counter % 26)

        # If suffix is 0, we just use letters; otherwise, append the suffix number
        var_name = letter if suffix == 0 else f"{letter}{suffix - 1}"

        # Increment the counter for the next variable name
        self.variable_counter += 1
        return var_name

    def reset_var_generator(self):
        """Reset the variable name generator."""
        self.variable_counter = 0

    def visit(self, node):
        """Recursive function to process each node in the AST."""
        if isinstance(node, ast.Call):
            args = []
            for arg in node.args:
                result = self.visit(arg)
                if isinstance(arg, (ast.Call, ast.BinOp)):
                    var_name = self.unique_var()
                    self.code_lines.append(f"{var_name} = {result}")
                    result = var_name
                args.append(result)
            return f"{node.func.id}({', '.join(args)})"

        elif isinstance(node, ast.BinOp):
            left = self.visit(node.left)
            right = self.visit(node.right)
            op = {ast.Add: '+', ast.Sub: '-', ast.Mult: '*', ast.Div: '/'}[type(node.op)]
            return f"{left} {op} {right}"

        elif isinstance(node, ast.Constant):
            return str(node.value)
        elif isinstance(node, ast.Name):
            return node.id

    def decompose_expression(self):
        """Convert an AST into a sequence of simple statements."""
        if self.tree is None:
            self.to_code()

        final_expr = self.visit(self.tree)
        self.code_lines.append(final_expr)
        problem_txt = '; '.join(self.code_lines)
        problem_txt += f' ? {eval(self.code):0.2f}'
        return problem_txt


# Example usage
if __name__ == "__main__":
    # Example usage
    generator = SymArithmeticProbGen()
    code = generator.to_code()
    print(code)
    from defs import *
    print(f'value = {eval(code)}')
    print(generator.decompose_expression())
