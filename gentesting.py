from absl import logging
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
    
    clause1 = "a b c = triangle"
    clause2 = "d = on_tline b a c, on_tline c a b"
    clause3 = "perp a d b c" # goal
    pr.Clause.from_txt(clause1)
    constr1 = "on_tline c a b"
    pr.Construction.from_txt(constr1)
    # point a is located at (0.3, 0.2)
    pr.Clause(points=["a@0.3_0.2", "b", "c"], constructions=[pr.Construction("on_tline", ["c", "a", "b"])])
    
    from pretty import pretty_nl

    def_text1 = """\
angle_bisector x a b c
x : a b c x
a b c = ncoll a b c
x : eqangle b a b x b x b c
bisect a b c
"""
    def_text2 = """\
eq_triangle x b c
x : b c
b c = diff b c
x : cong x b b c, cong b c c x; eqangle b x b c c b c x, eqangle x c x b b x b c
circle b b c, circle c b c
"""

    pr.Definition.from_string(def_text2)[0]
    
    
    

    defs_path = 'defs.txt'
    rules_path = 'rules.txt'

    # Load definitions and rules
    definitions, rules = load_definitions_and_rules(defs_path, rules_path)

    # show that the 3 lines from the corner to the midpoint of the opposite edge intersect in a single point
    # txt = 'a b c = triangle a b c; d = midpoint d a b; e = midpoint e b c; f = midpoint f c a; g = on_line g d c, on_line g e a ? coll f b g'
    # txt = "a b c = triangle; h = on_tline b a c, on_tline c a b ? perp a h b c"
    txt = "a b c = triangle; d = on_tline b a c, on_tline c a b ? perp a d b c"

    # imo problem    
    # txt = "a b = segment a b; g1 = on_tline g1 a a b; g2 = on_tline g2 b b a; m = on_circle m g1 a, on_circle m g2 b; n = on_circle n g1 a, on_circle n g2 b; c = on_pline c m a b, on_circle c g1 a; d = on_pline d m a b, on_circle d g2 b; e = on_line e a c, on_line e b d; p = on_line p a n, on_line p c d; q = on_line q b n, on_line q c d ? cong e p e q"
    
    logging.set_verbosity(logging.DEBUG)
    
    p = pr.Problem.from_txt(txt)
    print(f"Parsed problem: {p}")
    print("Res: " + p.setup_str_from_problem(definitions))
    g, _ = gh.Graph.build_problem(p, definitions)


    print(g)


if __name__ == '__main__':
    main()

