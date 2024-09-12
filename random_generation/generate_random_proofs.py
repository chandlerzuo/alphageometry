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
import copy


def convert_var_names_from_alpha_geo_names(var_map, goal_statement_as_list):
    new_var_statement = copy.deepcopy(goal_statement_as_list)
    for var_id in range(1, len(goal_statement_as_list)):
        new_var_statement[var_id] = var_map[goal_statement_as_list[var_id]]

    return new_var_statement


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
    cc_gen = CompoundClauseGen(definitions, 2, 3, 2, 42)
    txt = cc_gen.generate_clauses()
    # Let P be an interior point of triangle ABC and AP, BP, CP meet the sides BC, CA, AB in D, E, F respectively. Show that AP/PD = AF/FB + AE/EC
    txt = 'W P8 T4 Q3 = r_trapezoid W P8 T4 Q3; V7 W6 D3 W8 E0 = pentagon V7 W6 D3 W8 E0; U2 Y = square E0 P8 U2 Y; X5 = eqangle3 X5 Y Q3 D3 T4 E0; A6 P1 U1 Q7 = cc_tangent A6 P1 U1 Q7 Y P8 V7 W; V = nsquare V U1 E0; A0 = nsquare A0 V E0; Z2 = angle_mirror Z2 D3 Y X5; Y7 Z T7 = risos Y7 Z T7; P7 = intersection_lp P7 Q7 U1 U2 V7 Z; I7 = midpoint I7 E0 Y; Z7 = intersection_ll Z7 Q3 W8 Q7 D3 ? eqratio V7 P1 V7 Q7 Y A6 Y U1'
    # txt = 'A B C = iso_triangle A B C; D = angle_mirror D B C A; E = midpoint E A K ? eqangle P R K P Q L L Q'
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
    cache_node = list(random.choice(list((g.cache.keys()))))
    capitalized_pt_names = [point_name.capitalize() for point_name in cache_node[1:]]
    cache_node[1:] = capitalized_pt_names
    var_map = cc_gen.get_varname_2_alpha_geo_var_map()
    new_cache_node = convert_var_names_from_alpha_geo_names(var_map, cache_node)
    goal = pr.Construction(cache_node[0], list(cache_node[1:]))
    write_solution(g, p, goal=goal, out_file='')


if __name__ == "__main__":
    main()
