import random

import problem as pr
import graph as gh


def load_definitions_and_rules(defs_path, rules_path):
    """Load definitions and rules from text files."""
    definitions = pr.Definition.from_txt_file(defs_path, to_dict=True)
    rules = pr.Theorem.from_txt_file(rules_path, to_dict=True)
    return definitions, rules

def main():
    random.seed(0)
    # Example entities and conditions for illustration purposes

    defs_path = 'defs.txt'
    rules_path = 'rules.txt'

    # Load definitions and rules
    definitions, rules = load_definitions_and_rules(defs_path, rules_path)

    txt = 'a b c = triangle a b c; d = midpoint d a b; e = midpoint e b c; f = midpoint f c a; g = on_line g d c, on_line g e a ? coll f b g'
    p = pr.Problem.from_txt(txt)
    g, _ = gh.Graph.build_problem(p, definitions)


    print(g)


if __name__ == '__main__':
    main()

