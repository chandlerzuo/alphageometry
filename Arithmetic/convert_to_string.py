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
