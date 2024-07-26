import inspect
import defs
import random
import unittest
import sympy as sp


def acquire_symbols():
    function_map = []
    # Assuming defs contains callable classes that when instantiated, can be used directly as functions
    for name, obj in inspect.getmembers(defs, lambda x: inspect.isclass(x) and callable(x)):
        if name != 'Stringifiable':
            instance = obj()  # Instantiate each class
            function_map.append(instance)
    return function_map


class TestUniquenessOfDefs(unittest.TestCase):

    def test_same_input_different_output(self):
        num_inputs = 10
        functions = acquire_symbols()
        variables = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']

        for fn_n, func1 in enumerate(functions):
            args = inspect.signature(func1.__call__).parameters
            arity1 = len(args)
            for func2 in functions[(fn_n + 1):]:
                args = inspect.signature(func2.__call__).parameters
                arity2 = len(args)
                if arity1 == arity2:
                    input_args1 = [sp.symbols(variables[var_id]) for var_id in range(arity1)]
                    output1 = func1(*input_args1)
                    output2 = func2(*input_args1)
                    assert sp.simplify(output2 - output1) != 0, f"Function {func1} and {func2} are duplicates"


if __name__ == "__main__":
    import numpy as np
    random.seed(6)
    np.random.seed(6)
    unittest.main()