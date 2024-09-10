import sys

from utils.loading_utils import load_definitions_and_rules

sys.path.append('..')
import random
import ddar
from alphageometry import write_solution
import graph as gh
import problem as pr
from clause_generation import CompoundClauseGen
import signal


class TimeoutException(Exception):
    """Custom exception to indicate a timeout."""
    pass


def signal_handler(signum, frame):
    """Signal handler that raises a TimeoutException."""
    raise TimeoutException("Operation timed out due to signal 14 (SIGALRM)")


# Register the signal handler for SIGALRM
signal.signal(signal.SIGALRM, signal_handler)


def main():
    random.seed(7)
    # Example entities and conditions for illustration purposes

    defs_path = '../defs.txt'
    rules_path = '../rules.txt'

    # Load definitions and rules
    definitions, rules = load_definitions_and_rules(defs_path, rules_path)
    definitions, rules = load_definitions_and_rules(defs_path, rules_path)
    cc_gen = CompoundClauseGen(definitions, 2, 3, 2)
    txt = cc_gen.generate_clauses()
    # Let P be an interior point of triangle ABC and AP, BP, CP meet the sides BC, CA, AB in D, E, F respectively. Show that AP/PD = AF/FB + AE/EC
    # txt = 'A B C = triangle A B C; D = on_line B C; E = on_line C A; P = on_line A D, on_line B E; F = on_line C P, on_line A B'
    txt = 'A B C = iso_triangle A B C; D = angle_mirror D B C A; E = midpoint E A K ? eqangle P R K P Q L L Q'
    # txt = 'A B C D = quadrangle A B C D; E F G H = incenter2 E F G H B C D; I = on_tline I B A D; J = angle_mirror J G C A, on_opline E G; K L M N = excenter2 K L M N A J G; O P Q R = r_trapezoid O P Q R; S T = on_pline S A C D, angle_bisector T R B G'

    txt = txt.split('?')[0].strip()
    print(txt)

    p = pr.Problem.from_txt(txt)

    print(f'Problem created, Building graph ...')
    try:
        # Set an alarm for 10 seconds
        signal.alarm(10)

        # Code block to execute with timeout
        g, _ = gh.Graph.build_problem(p, definitions)

        # Disable the alarm
        signal.alarm(0)
    except TimeoutException as e:
        print("Graph couldn't bre create in reasonable time. Perhaps problem with the premises. Exiting ...")
        raise e

    # Additionaly draw this generated problem
    gh.nm.draw(
        g.type2nodes[gh.Point],
        g.type2nodes[gh.Line],
        g.type2nodes[gh.Circle],
        g.type2nodes[gh.Segment])

    print(f'Solving ...')

    ddar.solve(g, rules, p, max_level=1000)

    # Randomly select a cache node to be the goal. #TODO: Is this right can we do better? Consider coverage!
    # random.seed(4)
    cache_node = random.choice(list((g.cache.keys())))
    goal = pr.Construction(cache_node[0], list(cache_node[1:]))
    write_solution(g, p, goal=goal, out_file='')


if __name__ == "__main__":
    main()
