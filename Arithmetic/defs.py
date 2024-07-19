import re

_NUM_FUNCS_TO_USE = 65

assert _NUM_FUNCS_TO_USE in [5, 25, 45, 65]


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
        return (6 + x) / y + z


if _NUM_FUNCS_TO_USE >= 25:
    class sublimation:
        def __init__(self):
            self.commutative = False

        def __call__(self, x, y):
            return x - 3 * y


    class coagulation:
        def __init__(self):
            self.commutative = True

        def __call__(self, x, y):
            return x + y + 1


    class fixation:
        def __init__(self):
            self.commutative = True

        def __call__(self, x, y):
            return x * (y + 2)


    class calcification:
        def __init__(self):
            self.commutative = False

        def __call__(self, x, y):
            if (y + 2) == 0:
                y += 1e-7
            return x / (y + 2)


    class distillation:
        def __init__(self):
            self.commutative = False

        def __call__(self, x, y):
            return (x ** 2) + (y ** 2)


    class coction:
        def __init__(self):
            self.commutative = True

        def __call__(self, x, y):
            return (x * 3) - y


    class ceration:
        def __init__(self):
            self.commutative = False

        def __call__(self, x, y):
            return (x + y) * 5


    class amalgamation:
        def __init__(self):
            self.commutative = True

        def __call__(self, x, y):
            return x * (y - 0.1) + 2


    class levigation:
        def __init__(self):
            self.commutative = True

        def __call__(self, x, y):
            return (x + 5) - (y * 2)


    class solution:
        def __init__(self):
            self.commutative = False

        def __call__(self, x, y):
            return x / 3 + y / 2


    class filtration:
        def __init__(self):
            self.commutative = False

        def __call__(self, x, y):
            return (x + y ** 2) / 2


    class extraction:
        def __init__(self):
            self.commutative = True

        def __call__(self, x, y):
            return (x - y) * (y - 2)


    class digestion:
        def __init__(self):
            self.commutative = False

        def __call__(self, x, y):
            return (x + 2 * y) / 3


    class gennesis:
        def __init__(self):
            self.commutative = True

        def __call__(self, x, y):
            if x == y:
                x += 1e-7
            return x + x * y * 2 / (x - y)


    class projection:
        def __init__(self):
            self.commutative = False

        def __call__(self, x, y):
            return (x * y + 2) ** 2


    class elixiration:
        def __init__(self):
            self.commutative = True

        def __call__(self, x, y):
            return (x + y) * (x - y)


    class cementation:
        def __init__(self):
            self.commutative = False

        def __call__(self, x, y):
            return 3 * (x + y)


    class imbibition:
        def __init__(self):
            self.commutative = True

        def __call__(self, x, y):
            if (y + 1) == 0:
                y += 1e-7
            return (x + 10) / (y + 1)


    class putrefaction:
        def __init__(self):
            self.commutative = False

        def __call__(self, x, y):
            if (x + y) == 0:
                x += 1e-7
            return y - x / (x + y)


    class tincturation:
        def __init__(self):
            self.commutative = True

        def __call__(self, x, y):
            return x * 2 + y * 2


if _NUM_FUNCS_TO_USE >= 45:
    class pulverization:
        def __init__(self):
            self.commutative = True

        def __call__(self, x, y):
            return (x ** 2 - y ** 2) + x * y - 2


    class crystallization:
        def __init__(self):
            self.commutative = False

        def __call__(self, x, y):
            return x % y if y != 0 else 0


    class reverberation:
        def __init__(self):
            self.commutative = False

        def __call__(self, x, y):
            if (x + 1) == 0:
                x += 1e-7
            return (x * y) / (x + 1)


    class cerementation:
        def __init__(self):
            self.commutative = True

        def __call__(self, x, y):
            return (x + y) / 10


    class precipitation:
        def __init__(self):
            self.commutative = True

        def __call__(self, x, y):
            return (x * 10) - (y * 5)


    class incineration:
        def __init__(self):
            self.commutative = False

        def __call__(self, x, y):
            return x - y * 3 + x * y


    class maceration:
        def __init__(self):
            self.commutative = True

        def __call__(self, x, y):
            return (x * 2 - x * y) + (y * 3 + x)


    class evaporation:
        def __init__(self):
            self.commutative = False

        def __call__(self, x, y):
            return x / 2 - y


    class infusion:
        def __init__(self):
            self.commutative = True

        def __call__(self, x, y):
            return (x + y * 2) * x


    class impletion:
        def __init__(self):
            self.commutative = False

        def __call__(self, x, y):
            return x * (y + 5)


    class congelation:
        def __init__(self):
            self.commutative = True

        def __call__(self, x, y):
            return x * x + y * y - x * y


    class rectification:
        def __init__(self):
            self.commutative = False

        def __call__(self, x, y):
            return x - y * 2 + 3


    class subduction:
        def __init__(self):
            self.commutative = True

        def __call__(self, x, y, z):
            if y == z:
                y += 1e-7
            return (x - y) / (y - z)


    class ascension:
        def __init__(self):
            self.commutative = False

        def __call__(self, x, y):
            return (x + y) ** 2


    class mortification:
        def __init__(self):
            self.commutative = True

        def __call__(self, x, y):
            return x * y - x ** 2


    class reduction:
        def __init__(self):
            self.commutative = False

        def __call__(self, x, y, z):
            if z == 0:
                z += 1e-7
            return x / z + y


    class percolation:
        def __init__(self):
            self.commutative = True

        def __call__(self, x, y, z):
            if z == 0:
                z += 1e-7
            return (x + y) * 3 / z


    class liquefaction:
        def __init__(self):
            self.commutative = False

        def __call__(self, x, y):
            if x == y:
                x += 1e-7
            return (x * 2 + y * 3) / (x - y)


    class amalgation:
        def __init__(self):
            self.commutative = True

        def __call__(self, x, y):
            return x ** 2 - y ** 3


    class calcinization:
        def __init__(self):
            self.commutative = False

        def __call__(self, x, y):
            if (x + 1) == 0:
                x += 1e-7
            return (x + y) / (x + 1)


if _NUM_FUNCS_TO_USE >= 65:
    class vinculation:
        def __init__(self):
            self.commutative = True

        def __call__(self, x, y):
            return (x * y) + (x - y)


    class effervescence:
        def __init__(self):
            self.commutative = False

        def __call__(self, x, y):
            return (x ** 2) - 3 * y


    class amalgam:
        def __init__(self):
            self.commutative = True

        def __call__(self, x, y):
            if x == y:
                x += 1e-7
            return (x * y)/(x - y) * 2


    class albedo:
        def __init__(self):
            self.commutative = True

        def __call__(self, x, y):
            return (x + y) ** 3


    class nigredo:
        def __init__(self):
            self.commutative = False

        def __call__(self, x, y):
            if (x + 1) == 0:
                x += 1e-7
            return (x * y) / (x + 1 + x * y)


    class rubedo:
        def __init__(self):
            self.commutative = False

        def __call__(self, x, y):
            return x ** 2 + y ** 2 + x


    class deliquescence:
        def __init__(self):
            self.commutative = True

        def __call__(self, x, y, z):
            if (x + y - z) == 0:
                x += 1e-7

            return x * y / (x + y - z)


    class incrustation:
        def __init__(self):
            self.commutative = True

        def __call__(self, x, y):
            return (x * 3 + y * 2) / 5


    class quintessence:
        def __init__(self):
            self.commutative = False

        def __call__(self, x, y):
            return (x + y * x) / 2


    class alkahest:
        def __init__(self):
            self.commutative = True

        def __call__(self, x, y):
            return x - y + 10 + x * y


    class philosophers_stone:
        def __init__(self):
            self.commutative = False

        def __call__(self, x, y):
            return x * y + 1


    class athanor:
        def __init__(self):
            self.commutative = True

        def __call__(self, x, y):
            return (x + y) * (x - y) + 2


    class aurification:
        def __init__(self):
            self.commutative = False

        def __call__(self, x, y):
            return x + 5 * y


    class solification:
        def __init__(self):
            self.commutative = True

        def __call__(self, x, y, z):
            if z == 0:
                z += 1e-7
            return (x * x / z) - (y * y)


    class lunation:
        def __init__(self):
            self.commutative = False

        def __call__(self, x, y):
            return x / 2 + y * 2


    class solarization:
        def __init__(self):
            self.commutative = True

        def __call__(self, x, y):
            return x * 2 + y * 3


    class hermetic_seal:
        def __init__(self):
            self.commutative = False

        def __call__(self, x, y, z):
            return x * y - z


    class spiritization:
        def __init__(self):
            self.commutative = True

        def __call__(self, x, y, z):
            return (x + z) - (y + 10)


    class transmutation:
        def __init__(self):
            self.commutative = False

        def __call__(self, x, y):
            return x * 3 - y * 2


    class volatilization:
        def __init__(self):
            self.commutative = True

        def __call__(self, x, y):
            if (y + 1) == 0:
                y += 1e-7
            return (x * 10) / (y + 1)



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
    from Arithmetic.convert_to_string import Stringifiable
    local_env = {
        "calcination": lambda x, y: calcination()(Stringifiable(x), Stringifiable(y)),
        "dissolution": lambda x, y: dissolution()(Stringifiable(x), Stringifiable(y)),
        "separation": lambda x, y: separation()(Stringifiable(x), Stringifiable(y)),
        "conjunction": lambda x, y: conjunction()(Stringifiable(x), Stringifiable(y)),
        "fermentation": lambda x, y, z: fermentation()(Stringifiable(x), Stringifiable(y), Stringifiable(z)),
        # ""
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
