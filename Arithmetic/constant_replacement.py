import ast


class CodeConstTransformer:
    def __init__(self):
        self.temp_var_counter = 0
        self.constant_mapping = {}

    def get_temp_var(self):
        self.temp_var_counter += 1
        return f"X{self.temp_var_counter}"

    def replace_constants_with_temp_vars(self, code_string):
        # Parse the input string into an AST
        expr_tree = ast.parse(code_string.strip(), mode='exec')

        # Process the tree and replace constants
        new_code_string = self.process_tree_and_replace_constants(expr_tree)
        return new_code_string

    def process_tree_and_replace_constants(self, tree):
        new_code_lines = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                target = node.targets[0].id
                value_code = self.replace_constants_in_node(node.value)
                new_code_lines.append(f"{target} = {value_code}")

        return "; ".join(new_code_lines)

    def replace_constants_in_node(self, node):
        if isinstance(node, ast.Call):
            func_name = node.func.id
            args = [self.replace_constants_in_node(arg) for arg in node.args]
            args_str = ", ".join(args)
            return f"{func_name}({args_str})"
        elif isinstance(node, ast.Constant):  # For Python 3.8+
            temp_var = self.get_temp_var()
            self.constant_mapping[temp_var] = node.value
            return temp_var
        elif isinstance(node, ast.Num):  # For Python < 3.8
            temp_var = self.get_temp_var()
            self.constant_mapping[temp_var] = node.n
            return temp_var
        elif isinstance(node, ast.Name):
            return node.id

    def restore_constants_in_expression(self, expression):
        for temp_var, value in self.constant_mapping.items():
            expression = expression.replace(temp_var, str(value))
        return expression


if __name__ == '__main__':
    # Example usage
    code_string = 'A = calcination(1, 2); B = separation(100, 23); C = conjunction(A, B)'
    transformer = CodeConstTransformer()
    new_code_string = transformer.replace_constants_with_temp_vars(code_string)
    print("New Code String with Temporary Variables:", new_code_string)
    print("Constant Mapping:", transformer.constant_mapping)

    # Given alternative expression
    alternative_expression = 'A_0 = add(X1, X2); A_1 = mul(X3, X4); A_2 = div(A_0, A_1)'
    restored_expression = transformer.restore_constants_in_expression(alternative_expression)
    print("Restored Expression with Original Values:", restored_expression)
