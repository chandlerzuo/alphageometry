def add(x, y):
    return x + y


def sub(x, y):
    return x - y


def mul(x, y):
    return x * y


def div(x, y):
    return x / y


def bin_op1(x, y):
    if y == 0:
        y += 1e-7
    return 3 * 2 / y - (8 + 8 + x / y) + (10 + (x + 2) - (6 - y / (9 - 4)))


def bin_op2(x, y):
    if y == 0:
        y += 1e-7

    if x == 0:
        x += 1e-7
    return 2 / y + 10 + (6 - x * (9 - 4)) + (2 / y / x - 7 / x / (x / y))


def trin_op1(x, y, z):
    if y == 0:
        y += 1e-7
    return z * 2 / y - (z + 8 + x / y) + (z + (x + 2) - (6 - y / (9 - 4))) + z


def quat_op1(x, y, z, w):
    if y == 0:
        y += 1e-7
    return w * 2 / y - (w + 8 + z / y) + (z + (x + 2) - (6 - y / (9 - 4))) + w
