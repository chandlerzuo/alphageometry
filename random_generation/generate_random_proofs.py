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
    seed = 22
    import numpy as np
    np.random.seed(seed)
    random.seed(seed)
    # import tensorflow as tf
    # tf.random.set_seed(seed)
    import jax
    key = jax.random.PRNGKey(seed)

    # Example entities and conditions for illustration purposes

    defs_path = '../defs.txt'
    rules_path = '../rules.txt'

    # Load definitions and rules
    definitions, rules = load_definitions_and_rules(defs_path, rules_path)
    cg = ClauseGenerator(definitions)
    txt = cg.generate_clauses(2)
    # txt = 'A B C D = rectangle A B C D'
    txt = 'A B C = triangle A B C; o = circle o A B C' # #; A C D = triangle A C D, perp A D D C, para A B C D'

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

    print(f'Solving ...')

    ddar.solve(g, rules, p, max_level=1000)

    # Randomly select a cache node to be the goal. #TODO: Is this right can we do better? Consider coverage!
    # random.seed(4)
    cache_node = random.choice(list((g.cache.keys())))
    goal = pr.Construction(cache_node[0], list(cache_node[1:]))
    write_solution(g, p, goal=goal, out_file='', return_nl_also=True)


if __name__ == "__main__":
    main()
