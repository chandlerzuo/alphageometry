import sys
sys.path.append('..')
import random
import ddar
from alphageometry import write_solution
import graph as gh
import problem as pr
from clause_generation import ClauseGenerator
import signal


class TimeoutException(Exception):
    """Custom exception to indicate a timeout."""
    pass


def signal_handler(signum, frame):
    """Signal handler that raises a TimeoutException."""
    raise TimeoutException("Operation timed out due to signal 14 (SIGALRM)")


# Register the signal handler for SIGALRM
signal.signal(signal.SIGALRM, signal_handler)


def load_definitions_and_rules(defs_path, rules_path):
    """Load definitions and rules from text files."""
    definitions = pr.Definition.from_txt_file(defs_path, to_dict=True)
    rules = pr.Theorem.from_txt_file(rules_path, to_dict=True)
    return definitions, rules


def main():
    random.seed(37)
    # Example entities and conditions for illustration purposes

    defs_path = '../defs.txt'
    rules_path = '../rules.txt'

    # Load definitions and rules
    definitions, rules = load_definitions_and_rules(defs_path, rules_path)
    cg = ClauseGenerator(definitions)
    txt = cg.generate_clauses(5)
    # txt = 'A B = segment A B; C, D = square A B C D, on_bline C A B'
    # txt = 'A B = segment A B; C = free C; D = circle C A B, on_bline D A B'
    # txt = 'A B C = triangle A B C; D = parallelogram B C A D; E = on_pline E A D C; F = shift F D C E; G = psquare G F D'

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

    # Additionally draw this graph
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
