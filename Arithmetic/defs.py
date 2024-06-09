class add:
    def __init__(self):
        self.commutative = True

    def __call__(self, x, y):
        return x + y

class sub:
    def __init__(self):
        self.commutative = False

    def __call__(self, x, y):
        return x - y

class mul:
    def __init__(self):
        self.commutative = True

    def __call__(self, x, y):
        return x * y

class div:
    def __init__(self):
        self.commutative = False

    def __call__(self, x, y):
        if y == 0:
            y += 1e-7
        return x / y

class risky_trick:
    def __init__(self):
        self.commutative = False

    def __call__(self, x, y):
        if y == 0:
            y += 1e-7
        return 3 * 2 / y - (8 + 8 + x / y) + (10 + (x + 2) - (6 - y / (9 - 4)))

class p_exp:
    def __init__(self):
        self.commutative = False

    def __call__(self, x, y, z):
        if y == 0:
            y += 1e-7
        return z * 2 / y - (z + 8 + x / y) + (z + (x + 2) - (6 - y / (9 - 4))) + z

class leap_of_faith:
    def __init__(self):
        self.commutative = False

    def __call__(self, x, y, z, w):
        if y == 0:
            y += 1e-7
        return w * 2 / y - (w + 8 + z / y) + (z + (x + 2) - (6 - y / (9 - 4))) + w
