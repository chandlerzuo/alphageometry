def add(a, b):
    return a + b


def minus(a, b):
    return a - b


def mul(a, b):
    return a * b


def div(a, b):
    return a / b


def pow(a, b):
    return a ** b


def evaluate_expression(func_map, code, target_var):
    """Evaluate the generated code using instances of the operations."""
    # Prepare the environment with the operation functions
    local_env = {
        'add': add,
        'minus': minus,
        'mul': mul,
        'div': div,
        'pow': pow,
    }

    # Add function instances from func_map to the local environment
    local_env.update(func_map)

    # Evaluate the expression in this prepared environment
    exec(code, {}, local_env)
    return local_env.get(target_var)


if __name__ == "__main__":
    # Example usage
    from symbolic_restructure import get_code_last_var, GetAlternativeCode
    # code = 'A = calcination(4, 3); B = conjunction(4, 2); C = conjunction(A, B); C ?'
    # code = 'A = dissolution(5, 4); B = conjunction(4, 6); C = calcination(A, B); C ?'
    code = 'A=dissolution(2,0);B=calcination(1,1);C=dissolution(3,0);D=conjunction(A,C);E=dissolution(D,6);F=conjunction(6,8);G=dissolution(E,F);G?'
    code_changer = GetAlternativeCode()
    print(evaluate_expression(code_changer.function_map, *get_code_last_var(code)))

