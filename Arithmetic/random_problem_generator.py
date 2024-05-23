import ast
import importlib
import inspect
import random


class SymArithmeticProbGen:
    def __init__(self, seed=None, depth=2):
        self.depth = depth
        if seed is not None:
            random.seed(seed)
        self.defs = importlib.import_module('defs')
        self.functions = {}
        self.acquire_symbols()
        self.function_names = list(self.functions.keys())
        self.categorize_functions()
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
        """Acquire symbols from the defs module, instantiating callable classes."""
        for name, obj in inspect.getmembers(self.defs, inspect.isclass):
            instance = obj()  # Instantiate each class
            args = inspect.signature(instance.__call__).parameters
            arity = len(args)
            ret_count = self.estimate_return_count(instance, arity)
            self.functions[name] = (arity, ret_count, instance)

    def generate_expression(self, depth=0):
        """Recursively generates a random arithmetic expression using AST nodes."""
        if depth > self.depth:
            return ast.Constant(value=random.randint(1, 10))
        func_name = random.choice(self.function_names)
        func = self.functions[func_name][2]
        args = [self.generate_expression(depth + 1) for _ in range(len(inspect.signature(func.__call__).parameters))]
        call_node = ast.Call(func=ast.Name(id=func_name, ctx=ast.Load()), args=args, keywords=[])
        self.tree = call_node
        return call_node

    def to_code(self):
        """Convert an AST node to code"""
        self.code = ast.unparse(self.tree)
        return self.code

    def non_perturbing_intervention(self, node, changed=False):
        """Shuffles arguments for commutative functions and reorders binary operations if they are also commutative.
        Returns True if a change was made to stop further modifications."""
        # Only proceed with interventions if no change has been made yet
        if not changed:
            if isinstance(node, ast.Call) and getattr(self.functions[node.func.id][2], 'commutative', False):
                # This node is a function call to a commutative function
                if len(node.args) > 1:
                    original_order = [ast.unparse(arg) for arg in node.args]
                    random.shuffle(node.args)
                    new_order = [ast.unparse(arg) for arg in node.args]
                    if original_order != new_order:
                        # print(f"Shuffled {node.func.id} arguments from {original_order} to {new_order}")
                        return True  # Signal that a change has been made

            elif isinstance(node, ast.BinOp) and getattr(node.op, 'commutative', False):
                # This node is a binary operation that is commutative
                left_expr = ast.unparse(node.left)
                right_expr = ast.unparse(node.right)
                if left_expr != right_expr:
                    node.left, node.right = node.right, node.left
                    print(f"Swapped operands: {right_expr} and {left_expr}")
                    return True  # Signal that a change has been made

            # Recurse into the tree if it's a binary operation
            if isinstance(node, ast.BinOp):
                if self.non_perturbing_intervention(node.left, changed):
                    return True
                if self.non_perturbing_intervention(node.right, changed):
                    return True

            # Recurse into the function arguments if it's a function call
            elif isinstance(node, ast.Call):
                for arg in node.args:
                    if self.non_perturbing_intervention(arg, changed):
                        return True

        return False  # No change was made during this call

    def categorize_functions(self):
        self.func_by_arity = {}
        for name, (arity, _, func) in self.functions.items():
            if arity not in self.func_by_arity:
                self.func_by_arity[arity] = []
            self.func_by_arity[arity].append(name)

    def perturbing_intervention(self, node):
        """Recursively changes a function to another with the same number of input arguments. Returns True if a change was made."""
        if isinstance(node, ast.Call):
            # Get the current function name and its arity
            current_func_name = node.func.id
            arity = len(node.args)

            # Select a new function from the same arity category, excluding the current function
            possible_functions = [f for f in self.func_by_arity.get(arity, []) if f != current_func_name]
            if possible_functions and random.choice([True, False]):  # Random choice to change the function
                new_func_name = random.choice(possible_functions)
                node.func.id = new_func_name  # Change the function name in the AST node
                # print(f"Changed function from {current_func_name} to {new_func_name}")
                return True  # Return True indicating a change has been made

        # Recurse into the arguments of the function call
        if isinstance(node, ast.Call):
            for arg in node.args:
                if self.perturbing_intervention(arg):  # If a change is made in any argument, stop further changes
                    return True

        if isinstance(node, ast.BinOp):
            # Continue recursion for binary operations
            if self.perturbing_intervention(node.left) or self.perturbing_intervention(node.right):
                return True

        return False  # Return False if no changes have been made

    def unique_var(self):
        """Generate a unique variable name in alphabetical order followed by numerical suffixes."""
        suffix = self.variable_counter // 26
        letter = chr(ord('A') + self.variable_counter % 26)
        var_name = letter if suffix == 0 else f"{letter}{suffix - 1}"
        self.variable_counter += 1
        return var_name

    def reset_var_generator(self):
        """Reset the variable name generator."""
        self.variable_counter = 0

    def visit(self, node):
        """Recursive function to process each node in the AST."""
        if isinstance(node, ast.Call):
            # Recursively process each argument and capture the results
            args = [self.visit(arg) for arg in node.args]
            func_name = node.func.id  # Get the function name from the AST node
            result_expression = f"{func_name}({', '.join(args)})"  # Format the function call

            # Always assign the result of a function call to a new variable
            var_name = self.unique_var()
            self.code_lines.append(f"{var_name} = {result_expression}")  # Append the assignment to code lines
            return var_name  # Return the variable name to be used in higher level expressions

        elif isinstance(node, ast.BinOp):
            left = self.visit(node.left)
            right = self.visit(node.right)
            op = {ast.Add: '+', ast.Sub: '-', ast.Mult: '*', ast.Div: '/'}[type(node.op)]
            return f"{left} {op} {right}"

        elif isinstance(node, ast.Constant):
            return str(node.value)
        elif isinstance(node, ast.Name):
            return node.id

    def evaluate_expression(self, code):
        """Evaluate the generated code using instances of the operations."""
        # Prepare the environment with instances of the operation classes
        env = {name: self.functions[name][2] for name in self.function_names}
        # Evaluate the expression in this prepared environment
        result = eval(code, {}, env)
        return result

    def decompose_expression(self):
        self.reset_var_generator()
        self.code_lines = []
        """Convert an AST into a sequence of simple statements."""
        self.to_code()
        final_expr = self.visit(self.tree)
        self.code_lines.append(final_expr)
        problem_txt = '; '.join(self.code_lines)
        problem_txt += f' ? {self.evaluate_expression(self.code):0.2f}'
        return problem_txt


# Example usage
if __name__ == "__main__":
    generator = SymArithmeticProbGen()
    generator.generate_expression()
    print(f'original: \t\t {generator.decompose_expression()}')
    generator.perturbing_intervention(generator.tree)
    print(f'Perturbed: \t\t {generator.decompose_expression()}')

    if generator.non_perturbing_intervention(generator.tree):
        print(f'Non Perturbed: \t {generator.decompose_expression()}')
    else:
        print("No non-perturbing intervention was possible.")
