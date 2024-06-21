import random
import re
import ast
import unittest
import sympy as sp
from constant_replacement import CodeConstTransformer
from code_evaluator import evaluate_expression
from symbolic_restructure import GetAlternativeCode, sym_alter_exp, CodeGenerator, get_code_last_var


class TestSymRestructure(unittest.TestCase):
    def test_restructured_code(self):
        codes = ['A = conjunction(9, 1); B = conjunction(3, 8); C = fermentation(6, 3, 2); D = fermentation(A, B, C); D ?',
                 'A = calcination(1, 2); B = dissolution(100, 23); C = separation(A, B); C ?',
                 'A = conjunction(6, 7); B = fermentation(9, 7, 4); C = separation(A, B); C ?',
                 'A = calcination(10, 4); B = fermentation(7, 4, 9); C = calcination(A, B); C ?',
                 'A = fermentation(7, 5, 8); B = fermentation(4, 8, 7); C = separation(A, B); C ?',
                 ]
        code_changer = GetAlternativeCode()
        for code in codes:
            changed_code = code_changer(code)
            original_result = evaluate_expression(code_changer.function_map, *get_code_last_var(code))
            changed_result = evaluate_expression(code_changer.function_map, *get_code_last_var(changed_code))
            self.assertAlmostEqual(original_result, changed_result, delta=1e-7,
                                   msg=f"Original: {original_result}, Changed: {changed_result}")

    def test_constant_replacement(self):
        expressions = ['A = calcination(1, 2); B = dissolution(100, 23); C = separation(A, B); C ?',
                       'A = conjunction(6, 7); B = fermentation(9, 7, 4); C = separation(A, B); C ?',
                       'A = calcination(10, 4); B = fermentation(7, 4, 9); C = calcination(A, B); C ?',
                       'A = fermentation(7, 5, 8); B = fermentation(4, 8, 7); C = separation(A, B); C ?',
                       'A = conjunction(9, 1); B = conjunction(3, 8); C = fermentation(6, 3, 2); D = fermentation(A, B, C); D ?',]
        const_transformer = CodeConstTransformer()
        for exp in expressions:
            code, target_var = get_code_last_var(exp)
            code_with_sym_var = const_transformer.replace_constants_with_temp_vars(code)
            code_restored = const_transformer.restore_constants_in_expression(code_with_sym_var)
            const_transformer.reset()
            assert code == code_restored, f"Expected {code}, got {code_restored}"

    def test_code_gen(self):
        expressions = [
            "(3 + 5) * (7 - 2)",
            "4**2 + 2*4 + 1",
            "9/3 - 4*6 + 3*8",
            "2*5 + 7/1 - 3*4",
            "(8 + 6)**2 - (8 - 6)**2",
            "2*7 - 3*5 + 9/4",
            "3*8 + 4/2 - 5*7",
            "1**3 + 3*1**2 + 3*1 + 1",
            "2*3*4 - 6/3 + 5**2",
            "9/(3 + 2) - 2/(9 - 3)"
            "-5*6 + 3*4 - 2*7",
        ]
        code_gen = CodeGenerator()
        for exp in expressions:
            expression_body = ast.parse(str(exp), mode='eval').body
            _, result_var, new_code_sym_var = code_gen.generate_code(expression_body)
            res_new_code = evaluate_expression({}, new_code_sym_var, result_var)
            original_res = eval(exp)
            self.assertAlmostEqual(original_res, res_new_code, delta=1e-7,
                                   msg=f"Original: {original_res}, Changed: {res_new_code}")

    def test_sym_alter_exp(self):
        def extract_variables(expr):
            # Find all single character alphabetic variables
            return sorted(set(re.findall(r'[a-zA-Z]', expr)))

        expressions = [
            "(a + b) * (c - d)",
            "x**2 + 2*x + 1",
            "y/z - 4*y + 3*z",
            "p*q + r/s - t*u",
            "(m + n)**2 - (m - n)**2",
            "2*x - 3*y + z/4",
            "a*b + c/d - e*f",
            "x**3 + 3*x**2 + 3*x + 1",
            "a*b*c - d/e + f**2",
            "x/(y + z) - z/(x - y)"
        ]
        for exp in expressions:
            variables = set(extract_variables(exp))
            # Define the symbols in SymPy
            symbols = {var: sp.symbols(var) for var in variables}
            # Simplify the difference between the two expressions
            # Parse the expressions
            expr1_sympy = sp.sympify(exp, locals=symbols)
            expr2_sympy = sym_alter_exp(expr1_sympy)

            # Simplify the difference between the two expressions
            difference = sp.simplify(expr1_sympy - expr2_sympy)

            assert difference == 0, f"Expected 0, got {difference}"


if __name__ == "__main__":
    import numpy as np
    random.seed(6)
    np.random.seed(6)
    unittest.main()
