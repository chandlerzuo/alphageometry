from math import sqrt

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
        return x / y

class risky_trick:
    def __init__(self):
        self.commutative = False

    def __call__(self, x, y):
        z = x^2 + 3*y - 4*x*y + 5
        if z < 0:
            z = 0
        return sqrt(z)

class p_exp:
    # explodes when Z = 4 so potentially explosive
    def __init__(self):
        self.commutative = False

    def __call__(self, x, y, z):
        if z == 4:
            y += 1e-7
        return (x + 3*y) / (z - 4)

class leap_of_faith:
    def __init__(self):
        self.commutative = False

    def __call__(self, x, y, z, w):
        if y == 0:
            y += 1e-7
        return w * 2 / y - (w + 8 + z / y) + (z + (x + 2) - (6 - y / (9 - 4))) + w

