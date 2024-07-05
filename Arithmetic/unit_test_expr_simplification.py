import random
import unittest
import sympy as sp
from symbolic_restructure import simplify_explained


class ExprSimplification(unittest.TestCase):

    def test_step_by_step_symplification(self):
        expr = sp.sympify('x*(x + y) + y*(x + y) + (x**2 - y**2)/(x - y)')
        explanation, final_expr = simplify_explained(expr)
        # Printing how to get from a compact to a specific expression that we can represent in our altered formal system
        print(explanation)
        assert sp.simplify(expr - final_expr) == 0, f"Expected {expr} but got {final_expr}"


if __name__ == "__main__":
    import numpy as np
    random.seed(6)
    np.random.seed(6)
    unittest.main()
