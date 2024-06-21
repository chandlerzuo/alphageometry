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
        return 3 * 2 / y + z


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
