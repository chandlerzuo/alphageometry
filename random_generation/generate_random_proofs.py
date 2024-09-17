import sys

sys.path.append("..")
import logging
import os
import random
import shutil
import signal
from datetime import datetime

import ddar
import graph as gh
import jax
import numpy as np
import problem as pr
from alphageometry import write_solution
from clause_generation import ClauseGenerator


logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    stream=sys.stdout,
    level=logging.DEBUG,
)


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

    root_dir = "/home/chandlerzuo/alphageo_syn/experiment_20240917"
    try:
        os.mkdir(root_dir)
    except FileExistsError:
        pass
    N_PROBLEMS = 10
    seed = int(round(datetime.now().timestamp()))
    np.random.seed(seed)
    random.seed(seed)
    # import tensorflow as tf
    # tf.random.set_seed(seed)

    # key = jax.random.PRNGKey(seed)

    # Example entities and conditions for illustration purposes

    defs_path = "/home/chandlerzuo/alphageo_syn/alphageometry/defs.txt"
    rules_path = "/home/chandlerzuo/alphageo_syn/alphageometry/rules.txt"

    # Load definitions and rules
    definitions, rules = load_definitions_and_rules(defs_path, rules_path)
    i_problem = 0
    while i_problem < N_PROBLEMS:
        while True:
            output_dir = os.path.join(root_dir, f"problem_{i_problem}")
            try:
                os.mkdir(output_dir)
                break
            except FileExistsError:
                shutil.rmtree(output_dir)
                # i_problem += 1
        logging.info(f"Try to generate problem {i_problem} ...")
        cg = ClauseGenerator(definitions)
        txt = cg.generate_clauses(5)
        # txt = 'A B C = triangle A B C; D = circle A B C'
        # txt = 'a b c d = free_4pt a b c d, equal_seg d a d b, equal_seg d b d c'
        # txt = 'a b c d = free_4pt a b c d'
        # txt = 'a b c d = rectangle a b c d'
        # txt = 'B = free B; C = free C; A = free A; D = free D; A = equal_seg A B B C'
        # txt = 'B = free B; C = free C; A = free A; D = free D; A = equal_seg A B B C; D = equal_seg D A D B; D = equal_seg D B D C'
        # txt = 'B = free B; C = free C; A = free A; D = free D; E = free E; F = free F; A = eqangle3 A C B C A B'
        # txt = 'b = free b; c = free c; a = eq_triangle a b c'
        print("Clauses: ", txt)

        try:
            signal.alarm(10)
            # Set an alarm for 10 seconds
            p = pr.Problem.from_txt(txt)
        except Exception as e:
            logging.error(f"Error creating the problem: {e}")
            signal.alarm(0)
            continue
        logging.info("Problem created, Building graph ...")
        try:
            signal.alarm(10)
            # Code block to execute with timeout
            g, _ = gh.Graph.build_problem(p, definitions)
        except TimeoutException as e:
            logging.error(
                "Graph couldn't bre create in reasonable time. "
                "Perhaps problem with the premises. Skipping ..."
                f"{e}"
            )
            signal.alarm(0)
            continue
        except Exception as e:
            logging.error(f"Error creating the graph: {e}")
            signal.alarm(0)
            continue

        logging.info(f"Solving problem {i_problem}")
        try:
            signal.alarm(20)
            ddar.solve(g, rules, p, max_level=1000)
        except TimeoutException:
            logging.error("Time out solving the problem.")
            signal.alarm(0)
            continue
        except Exception as e:
            logging.error(f"Error solving the problem: {e}")
            signal.alarm(0)
            continue
        signal.alarm(0)
        # Randomly select a cache node to be the goal. #TODO: Is this right can we do better? Consider coverage!
        i_proof = 0
        for cache_node in g.cache.keys():
            # cache_node[0]: name, cache_node[1:]: args
            # key contains the clause to be proved
            goal = pr.Construction(cache_node[0], list(cache_node[1:]))
            print("Goal from problem: ", p.goal.txt() if p.goal else "None")
            print(f"Cache: {cache_node}")
            solution_nl = write_solution(
                g,
                p,
                goal=goal,
                out_file=os.path.join(output_dir, f"proof_{i_proof}.txt"),
                return_nl_also=True,
            )
            if solution_nl:
                with open(os.path.join(output_dir, f"proof_{i_proof}.txt"), "w") as f:
                    f.write(solution_nl)
                i_proof += 1
        logging.info(f"Proof written for problem {i_problem}")
        i_problem += 1


if __name__ == "__main__":
    main()
