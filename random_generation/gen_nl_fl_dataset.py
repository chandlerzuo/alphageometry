import sys
sys.path.append('..')
import random
import ddar
import graph as gh
import problem as pr
from clause_generation import ClauseGenerator
import signal
from generate_random_proofs import load_definitions_and_rules, TimeoutException
from prettier_print.pretty_problem_statement import get_nl_problem_statement
from pretty import pretty_nl
from prettier_print.prettier_proof_statements import translate_step
from utils.get_rand_gen_states import get_random_states

import csv

if __name__ == "__main__":

    dataset_length = 20
    filename = '../../datasets/nl_fl_dataset.csv'
    # filename = '../data/nl_fl_dataset_2.csv'
    random.seed(17)
    defs_path = '../defs.txt'
    rules_path = '../rules.txt'

    # Load definitions and rules
    definitions, rules = load_definitions_and_rules(defs_path, rules_path)

    field_names = ['sl_n', 'num_clauses', 'nl_statement', 'fl_statement', 'goal_nl', 'goal_fl', 'rnd_states']

    # Write data to the CSV file
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        # writer = csv.DictWriter(csvfile, fieldnames=field_names,)# delimiter='#')
        # this is necessary for the inspect to work
        writer = csv.DictWriter(csvfile, fieldnames=field_names, delimiter='#')
        writer.writeheader()
        serial_num = 0
        for i in range(dataset_length):
            num_clauses = random.randint(3, 10)
            cg = ClauseGenerator(definitions)
            fl_statement = cg.generate_clauses(5)

            print(fl_statement)

            try:
                p = pr.Problem.from_txt(fl_statement)
            except KeyError as e:
                print(e)
                continue

            print(f'Problem created, Building graph ...')
            try:
                # Set an alarm for 10 seconds
                signal.alarm(10)

                # Code block to execute with timeout
                g, _ = gh.Graph.build_problem(p, definitions)

                # Disable the alarm
                signal.alarm(0)
            except TimeoutException as e:
                print("Graph couldn't bre create in reasonable time. Perhaps problem with the premises. Continuing ...")
                continue
            except KeyError:
                print("Key error while building graph. Continuing ...")
                continue
            except ValueError:
                print("Value error while building graph. Continuing ...")
                continue

            print(f'Solving ...')

            ddar.solve(g, rules, p, max_level=1)

            # Randomly select a cache node to be the goal. #TODO: Is this right can we do better? Consider coverage!
            possible_goals = list(g.cache.keys())
            if len(possible_goals) > 0:
                goal_fl = random.choice(possible_goals)  # comment this line
                # goal_fl = random.choice(possible_goals + [''])  # uncomment this line to get goal less problems
                if goal_fl == '':
                    goal_nl = ''
                else:
                    pretty_goal = pretty_nl(goal_fl[0], goal_fl[1:])
                    if pretty_goal is None:
                        raise ValueError(f'Could not pretty print goal: {goal_fl}')
                    goal_nl = translate_step(pretty_goal)
                    goal_fl = ' '.join(goal_fl)
                # Now we know that the generated premises are not contradictory
                nl_prob = get_nl_problem_statement(fl_statement)
                # dump this row
                row = {'sl_n': serial_num, 'num_clauses': num_clauses, 'nl_statement': nl_prob,
                       'fl_statement': fl_statement, 'goal_nl': goal_nl, 'goal_fl': goal_fl,
                       'rnd_states': get_random_states()}
                writer.writerow(row)
                serial_num += 1
