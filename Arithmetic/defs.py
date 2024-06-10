class calcination:
    def __init__(self):
        self.commutative = True

    def __call__(self, x, y):
        return x + y

class dissolution:
    def __init__(self):
        self.commutative = False

    def __call__(self, x, y):
        return x - y

class separation:
    def __init__(self):
        self.commutative = True

    def __call__(self, x, y):
        return x * y

class conjunction:
    def __init__(self):
        self.commutative = False

    def __call__(self, x, y):
        if y == 0:
            y += 1e-7
        return x / y

class fermentation:
    def __init__(self):
        self.commutative = False

    def __call__(self, x, y, z):
        if y == 0:
            y += 1e-7
        return 3 * 2 / y - (z + 8 + x / y) + (10 + (x + z) - (6 - y / (9 - 4)))
