import re


class Stringifiable:
    num_vars = 0
    vars_mapping = {}

    @classmethod
    def reset(cls):
        cls.num_vars = 0
        cls.vars_mapping = {}

    def __init__(self, desc):
        if isinstance(desc, int):
            # desc = f"x{desc}"
            Stringifiable.num_vars += 1
            value = desc
            desc = f"X{Stringifiable.num_vars}"
            self.vars_mapping[desc] = value
        elif isinstance(desc, Stringifiable):
            desc = desc.desc

        self.desc = desc

    def __repr__(self):
        return self.desc

    def __mul__(self, other):
        return Stringifiable(f"({self} * {other})")

    def __rmul__(self, other):
        return Stringifiable(f"({other} * {self})")

    def __add__(self, other):
        return Stringifiable(f"({self} + {other})")

    def __radd__(self, other):
        return Stringifiable(f"({other} + {self})")

    def __sub__(self, other):
        return Stringifiable(f"({self} - {other})")

    def __rsub__(self, other):
        return Stringifiable(f"({other} - {self})")

    def __truediv__(self, other):
        return Stringifiable(f"({self} / {other})")

    def __rtruediv__(self, other):
        return Stringifiable(f"({other} / {self})")
    # def __minus__(self, other):
    #     return Stringifiable(f"{get_str(self)} - {get_str(other)}")
    # def __add__(self, other):
    #     return Stringifiable(f"{get_str(self)} + {get_str(other)}")
    # def __truediv__(self, other):
    #     return Stringifiable(f"{get_str(self)} / {get_str(other)}")


class calcination:
    def __init__(self):
        self.commutative = True

    def __call__(self, x, y):
        return x + 2 * y


class dissolution:
    def __init__(self):
        self.commutative = False

    def __call__(self, x, y):
        return x * x - y


# Stringifiable(2) * Stringifiable(3)
# calcination()(Stringifiable(2), Stringifiable(3))
# calcination()(2, Stringifiable(3))

# %%
class separation:
    def __init__(self):
        self.commutative = True

    def __call__(self, x, y):
        return x * y - x


class conjunction:
    def __init__(self):
        self.commutative = False

    def __call__(self, x, y):
        if y == -1:
            y += 1e-7
        return x / (y + 1)


class fermentation:
    def __init__(self):
        self.commutative = False

    def __call__(self, x, y, z):
        if y == 0:
            y += 1e-7
        return 6 / y + z


# The following should not be callable classes
# Because these are there only for verbalization not to be used in expression generation
def add(x, y):
    raise NotImplementedError("This function is not meant to be used in expression generation therefore an evaluation "
                              "is not necessary. It is only for verbalization.")


def minus(x, y):
    raise NotImplementedError("This function is not meant to be used in expression generation therefore an evaluation "
                              "is not necessary. It is only for verbalization.")


def mul(x, y):
    raise NotImplementedError("This function is not meant to be used in expression generation therefore an evaluation "
                              "is not necessary. It is only for verbalization.")


def pow(x, y):
    raise NotImplementedError("This function is not meant to be used in expression generation therefore an evaluation "
                              "is not necessary. It is only for verbalization.")


def div(x, y):
    raise NotImplementedError("This function is not meant to be used in expression generation therefore an evaluation "
                              "is not necessary. It is only for verbalization.")


def get_symbolic_expr_code_var_map(expr):
    local_env = {
        "calcination": lambda x, y: calcination()(Stringifiable(x), Stringifiable(y)),
        "dissolution": lambda x, y: dissolution()(Stringifiable(x), Stringifiable(y)),
        "separation": lambda x, y: separation()(Stringifiable(x), Stringifiable(y)),
        "conjunction": lambda x, y: conjunction()(Stringifiable(x), Stringifiable(y)),
        "fermentation": lambda x, y, z: fermentation()(Stringifiable(x), Stringifiable(y), Stringifiable(z)),
        # "add": add,
    }

    expr = expr[:-1]
    var_name = expr.split(";")[-1].strip()
    Stringifiable.reset()
    exec(expr, {}, local_env)

    code = expr
    # Use regex to find all integers and store their positions
    integers = [(m.group(), m.start()) for m in re.finditer(r'\b\d+\b', code)]

    # Sort integers by their position in the string
    integers.sort(key=lambda x: x[1])

    # Replace each integer with x1, x2, ..., in their original positions
    sym_code = code
    offset = 0  # This keeps track of the change in string length due to replacements

    for i, (number, position) in enumerate(integers):
        replacement = f'X{i + 1}'
        # Calculate new position adjusting with the offset
        new_position = position + offset
        # Replace in the string
        sym_code = sym_code[:new_position] + sym_code[new_position:].replace(number, replacement, 1)
        # Update the offset based on the length change from number to replacement
        offset += len(replacement) - len(number)

    return local_env[var_name], sym_code, Stringifiable.vars_mapping


if __name__ == "__main__":

    # %%
    for expr in [
        "A = calcination(1, 2); A ?",
        "A = dissolution(2, 3); A ?",
        "A = separation(1, 2); A ?",
        "A = conjunction(2, 3); A ?",
        "A = fermentation(1, 2, 3); A ?",

        "A = conjunction(7, 10); B = dissolution(1, 3); C = conjunction(A, B); C ?",
        "A = conjunction(4, 4); B = fermentation(7, 10, 7); C = dissolution(A, B); C ?",
        "A = fermentation(4, 1, 1); B = separation(7, 10); C = calcination(A, B); C ?",
        "A = separation(4, 10); B = separation(10, 5); C = dissolution(A, B); C ?",
        "A = fermentation(1, 5, 9); B = fermentation(7, 5, 8); C = dissolution(10, 4); D = fermentation(A, B, C); D ?"]:
        print("Expression:", get_symbolic_expr_code_var_map(expr))  # , Stringifiable.vars_mapping
        print("Formalized:", expr)
