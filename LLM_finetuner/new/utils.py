import math


def handle_fraction(fraction_or_int, total_len):
    if fraction_or_int < 1:
        return math.ceil(fraction_or_int * total_len)
    return fraction_or_int