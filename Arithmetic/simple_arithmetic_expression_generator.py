import ast
import operator
import random


class RandomExpression:
    def __init__(self, depth=2, seed=None):
        if seed is not None:
            random.seed(seed)  # Set the random seed if provided
        self.depth = depth
        self.tree = self.generate_expression()
        self.tree = ast.fix_missing_locations(self.tree)  # Fix missing lineno and col_offset
        self.expression = ""

    def generate_expression(self, depth=0):
        """Recursively generates a random arithmetic expression using AST nodes."""
        if depth > self.depth:
            return ast.Constant(value=random.randint(1, 10))

        # Possible operations
        ops = [ast.Add, ast.Sub, ast.Mult, ast.Div]
        left = self.generate_expression(depth + 1)
        right = self.generate_expression(depth + 1)

        # Randomly decide to wrap nodes in parentheses
        if random.choice([True, False]):
            left = ast.BinOp(left=left, op=random.choice(ops)(), right=self.generate_expression(depth + 2))
        if random.choice([True, False]):
            right = ast.BinOp(left=self.generate_expression(depth + 2), op=random.choice(ops)(), right=right)

        return ast.BinOp(left=left, op=random.choice(ops)(), right=right)

    def non_perturbing_intervention(self, node, changed=False):
        """Reorders operations that are commutative and ensures at least one change."""
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Mult)):
            left_expr = ast.unparse(node.left)
            right_expr = ast.unparse(node.right)
            if left_expr != right_expr:  # Only swap if they are not the same
                node.left, node.right = node.right, node.left
                changed = True  # Mark that a change has occurred
            # Continue recursion and pass whether any change has happened
            changed = self.non_perturbing_intervention(node.left, changed)
            changed = self.non_perturbing_intervention(node.right, changed)
        elif isinstance(node, ast.BinOp):
            # Recursively check left and right branches
            changed = self.non_perturbing_intervention(node.left, changed)
            changed = self.non_perturbing_intervention(node.right, changed)
        return changed

    def perturbing_intervention(self, node):
        """Changes the operation to potentially alter the result."""
        if isinstance(node, ast.BinOp):
            if random.choice([True, False]):  # Random choice to change the operation
                node.op = random.choice([ast.Sub(), ast.Div()]) if isinstance(node.op, (ast.Add, ast.Mult)) else random.choice([ast.Add(), ast.Mult()])
            self.perturbing_intervention(node.left)
            self.perturbing_intervention(node.right)

    def evaluate(self):
        """Evaluates the manipulated expression safely using the ast module."""
        compiled_expr = compile(ast.Expression(self.tree), '<string>', 'eval')
        return eval(compiled_expr, {'__builtins__': None}, {'sqrt': operator.pow})

    def to_text(self):
        """Returns the expression along with its evaluated result in a specific format."""
        self.expression = ast.unparse(self.tree)
        result = self.evaluate()
        return f"{self.expression}?{result:.2f}"


# Example usage:
if __name__ == "__main__":

    random_expr = RandomExpression(depth=2, seed=1)
    print("Original:", random_expr.to_text())
    changed = random_expr.non_perturbing_intervention(random_expr.tree)
    if not changed:  # If no change has happened, force one
        random_expr.non_perturbing_intervention(random_expr.tree, changed=True)
    print("Non-Perturbing Changed:", random_expr.to_text())
    random_expr.perturbing_intervention(random_expr.tree)
    print("Perturbing Changed:", random_expr.to_text())
