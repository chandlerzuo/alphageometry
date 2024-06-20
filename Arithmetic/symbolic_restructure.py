import sympy as sp
import numpy as np
import random
import ast
import inspect
from constant_replacement import CodeConstTransformer


def sym_alter_exp(expr):
    funcs = [sp.simplify, sp.expand, sp.factor]

    # Collect randomly around one of the free symbols if available
    symbols = list(expr.free_symbols)
    if symbols:
        # Choose a random symbol to collect around
        symbol_to_collect = np.random.choice(symbols)
        # Add collect function with this symbol to the function list
        funcs.append(lambda expr: sp.collect(expr, symbol_to_collect))

    # Check if the expression is a rational function for 'together' and 'apart'
    if expr.is_rational_function():
        funcs.append(sp.together)
        if len(symbols) == 1:  # Ensure that the expression is univariate before adding sp.apart
            funcs.append(sp.apart)

    # Randomly select a function that is applicable
    func = np.random.choice(funcs)
    return func(expr)


# Define a class to walk through the AST and generate code
class CodeGenerator(ast.NodeVisitor):
    def __init__(self):
        self.operations = []
        self.var_count = 0

    def visit_BinOp(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)
        op_type = type(node.op)
        result_var = f"A_{self.var_count}"
        self.var_count += 1

        if op_type == ast.Add:
            self.operations.append(f"{result_var} = add({left}, {right})")
        elif op_type == ast.Sub:
            self.operations.append(f"{result_var} = minus({left}, {right})")
        elif op_type == ast.Mult:
            self.operations.append(f"{result_var} = mul({left}, {right})")
        elif op_type == ast.Div:
            self.operations.append(f"{result_var} = div({left}, {right})")
        else:
            raise NotImplementedError(f"Operation {ast.dump(node.op)} not supported")

        return result_var

    def visit_Name(self, node):
        return node.id

    def visit_Num(self, node):
        return str(node.n)

    def visit_Constant(self, node):
        return str(node.value)

    def visit_Expr(self, node):
        return self.visit(node.value)

    def generate_code(self, node):
        result_var = self.visit(node)
        return self.operations, result_var, '; '.join(self.operations)


class GetAlternativeCode:
    def __init__(self, defs_module=None, seed=None):
        if seed is not None:
            random.seed(seed)
        if defs_module is None:
            import defs as defs_module  # Make sure this import is valid based on your project structure
        self.defs = defs_module

        # Building the function map from the defs module
        self.acquire_symbols()
        self.code_gen = CodeGenerator()
        self.const_transformer = CodeConstTransformer()

    def acquire_symbols(self):
        self.function_map = {}
        # Assuming defs contains callable classes that when instantiated, can be used directly as functions
        for name, obj in inspect.getmembers(self.defs, lambda x: inspect.isclass(x) and callable(x)):
            instance = obj()  # Instantiate each class
            self.function_map[name] = instance

    def execute_expression(self, node, local_vars):
        if isinstance(node, ast.Call):
            func_name = node.func.id
            args = [self.execute_expression(arg, local_vars) for arg in node.args]
            if func_name in self.function_map:
                return self.function_map[func_name](*args)
            else:
                raise KeyError(f"Function {func_name} not found in function map.")
        elif isinstance(node, ast.Name):
            if node.id in local_vars:
                return local_vars[node.id]
            return sp.Symbol(node.id)  # Use sympy Symbol for variable names
        elif isinstance(node, ast.Constant):  # For Python 3.8+
            return node.value
        elif isinstance(node, ast.Num):  # For Python < 3.8
            return node.n

    def process_tree(self, tree):
        local_vars = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                target = node.targets[0].id
                value = self.execute_expression(node.value, local_vars)
                local_vars[target] = value
        return local_vars

    def __call__(self, code_string):
        code_string, key_var = code_string.split('?')
        key_var = key_var.strip()  # Cleanup any whitespace around the variable name


        code_with_sym_var = self.const_transformer.replace_constants_with_temp_vars(code_string)

        expr_tree = ast.parse(code_with_sym_var.strip(), mode='exec')  # Parse the code string into an AST
        processed_vars = self.process_tree(expr_tree)  # Process the AST to execute expressions
        final_expr = processed_vars.get(key_var.strip(), f'Variable {key_var} not found')  # Fetch the final variable expression

        altered_expression = sym_alter_exp(final_expr)
        altered_expression_body = ast.parse(str(altered_expression), mode='eval').body
        _, result_var, new_code_sym_var = self.code_gen.generate_code(altered_expression_body)

        new_code_const_var = self.const_transformer.restore_constants_in_expression(new_code_sym_var)

        return new_code_const_var + f' ? {result_var}'


if __name__ == "__main__":
    import defs  # Import your module, ensuring it is accessible and correctly defined
    init_prog = 'A = calcination(1, 2); B = separation(100, 23); C = conjunction(A, B)? C'
    code_changer = GetAlternativeCode(defs)
    result = code_changer(init_prog)
    print("Result:", result)
