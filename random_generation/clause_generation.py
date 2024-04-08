import random
import string


#TODO(Priya): IMPORTANT! 's_angle' will not be sampled correctly as the 4th argument must be a numerical FIX this!
# 'triangle12'  also will not be generated right
class ClauseGenerator:
    def __init__(self, defs):
        self.defs = defs
        self.defined_points = []
        self.clause_relations = list(defs.keys()) # this is the full set we can't deal with it yet
        # To limit to a a few concepts uncomment the following line
        # self.clause_relations = ['angle_bisector', 'angle_mirror', 'circle', 'circumcenter', 'midpoint', 'triangle']
        self.point_counter = 0  # Start from 0
        self.max_points = 26 * 10  # 26 letters, 10 cycles (0 to 9, inclusive)

    def generate_point(self):
        """
        Generate the next point in sequence: A, B, ..., Z, A1, B1, ..., Z9.
        After Z9, raise an error.
        """
        if self.point_counter >= self.max_points:
            # If we've exhausted all possible names, raise an error
            raise ValueError("All point names have been exhausted.")

        # Calculate the letter and number parts of the name
        letter_part = string.ascii_uppercase[self.point_counter % 26]
        number_part = self.point_counter // 26

        # Prepare the point name
        if number_part == 0:
            # For the first cycle (A-Z), we don't add a number part
            point_name = letter_part
        else:
            # For subsequent cycles, add the number part (reduced by 1 to start from 0)
            point_name = f"{letter_part}{number_part - 1}"

        # Increment the counter for the next call
        self.point_counter += 1

        return point_name

    def generate_new_point(self):
        while True:
            point = self.generate_point()
            if point not in self.defined_points:
                return point

    def choose_random_n_defined_points(self, n):
        """
        Choose n random defined points
        """
        return random.sample(self.defined_points, n)

    def get_text_clause(self, clause_relation, arg_vars, result_vars):
        """
        Make a canonical clause for a given relation
        """
        if result_vars:
            clause_txt = f'{" ".join(result_vars)} = {clause_relation} {" ".join(result_vars + arg_vars)}'
        else:
            clause_txt = f'{clause_relation} {" ".join(arg_vars)}'
        return clause_txt

    def choose_random_defined_points(self, minimum_pts, max_pts):
        if not self.defined_points or minimum_pts < 1:  # Check if the list is empty
            return []  # Return an empty list or handle the scenario as needed

        # Choose a random number of points to select, from 1 up to the length of the list
        n = random.randint(minimum_pts, min(max_pts, len(self.defined_points)))

        # Randomly select 'n' points from the list
        chosen_defined_pts = random.sample(self.defined_points, n)

        return chosen_defined_pts

    def generate_clauses(self, n):
        """
        Generate n random clauses with all points defined
        """
        clauses = []
        for i in range(n):
            # choose a random definition key as the first clause
            suitable_clause = False
            while not suitable_clause:
                clause_relation = random.choice(self.clause_relations)
                needs_defined_points = len(self.defs[clause_relation].args)
                defines_points = len(self.defs[clause_relation].points)
                if needs_defined_points <= len(self.defined_points):
                    suitable_clause = True

            chosen_defined_pts = random.sample(self.defined_points, needs_defined_points)
            # Generate names of points that are needed for the clause
            will_be_defined_pts = []
            while defines_points > 0:
                will_be_defined_pts.append(self.generate_new_point())
                self.defined_points.append(will_be_defined_pts[-1])
                defines_points -= 1

            clause = self.get_text_clause(clause_relation, chosen_defined_pts, will_be_defined_pts)
            clauses.append(clause)

        return '; '.join(clauses)


if __name__ == "__main__":
    # random.seed(2)
    from generate_random_proofs import load_definitions_and_rules
    defs_path = '../defs.txt'
    rules_path = '../rules.txt'

    # Load definitions and rules
    definitions, rules = load_definitions_and_rules(defs_path, rules_path)
    cg = ClauseGenerator(definitions)
    # print(cg.get_text_clause('angle_bisector', ['x', 'y'], ['b', 'c']))

    print(cg.generate_clauses(5))
