import re
import random


def verbalize_clause(element_name, arguments):
    if element_name == 'angle_bisector':
        return random.choice([
            f"{arguments[0]} is a point such that ∠{arguments[1]}{arguments[2]}{arguments[0]} = ∠{arguments[0]}{arguments[2]}{arguments[2]}. ",
            f"{arguments[0]} is a point such that ∠{arguments[1]}{arguments[2]}{arguments[0]} is equal to ∠{arguments[0]}{arguments[2]}{arguments[2]}. "])
    elif element_name == 'angle_mirror':
        return random.choice([
            f"{arguments[0]} is a point such that ∠{arguments[1]}{arguments[2]}{arguments[3]} = ∠{arguments[3]}{arguments[2]}{arguments[0]}. ",
            f"{arguments[0]} is a point such that ∠{arguments[1]}{arguments[2]}{arguments[3]} is equal to ∠{arguments[3]}{arguments[2]}{arguments[0]}. "])
    elif element_name == 'circle':
        return f"{arguments[0]} is the centre of the circle that passes through {arguments[1]}, {arguments[2]}, {arguments[3]}. "
    elif element_name == 'circumcenter':
        return f"{arguments[0]} is the centre of the circumcenter of the triangle {arguments[1]}{arguments[2]}{arguments[3]}. "
    elif element_name == 'eq_quadrangle':
        return f"{arguments[0]}{arguments[1]}{arguments[2]}{arguments[3]} is a quadrilateral with {arguments[0]}{arguments[3]} = {arguments[1]}{arguments[2]}. "
    elif element_name == 'eq_trapezoid':
        return f"{arguments[0]}{arguments[1]}{arguments[2]}{arguments[3]} is an isosceles trapezoid with {arguments[3]}{arguments[0]} = {arguments[1]}{arguments[2]}. "
    elif element_name == 'eq_triangle':
        return f"{arguments[0]}{arguments[1]}{arguments[2]} is an equilateral triangle. "
    elif element_name == 'eqangle2':
        return f"{arguments[0]} is a point such that ∠{arguments[2]}{arguments[1]}{arguments[0]} = ∠{arguments[0]}{arguments[3]}{arguments[2]}. "
    elif element_name == 'eqdia_quadrangle':
        return f"{arguments[0]}{arguments[1]}{arguments[2]}{arguments[3]} is a quadrilateral whose diagonals are equal. "
    elif element_name == 'eqdistance':
        return f"{arguments[0]} is a point such that {arguments[0]}{arguments[1]} = {arguments[2]}{arguments[3]}. "
    elif element_name == 'foot':
        return f"{arguments[0]} is the foot of the perpendicular dropped on to {arguments[2]}{arguments[3]} from {arguments[1]}. "
    elif element_name == 'free':
        return f"{arguments[0]} is a point. "
    elif element_name == 'incenter':
        return f"{arguments[0]} is is the incentre of triangle {arguments[1]}{arguments[2]}{arguments[3]}. "
    elif element_name == 'incenter2': # incentre and the tangent points on the triangle arms
        # TODO(Partha): opportunity to augment! Symantic!  Also use respectively!
        return f"{arguments[3]} is is the incentre of triangle {arguments[4]}{arguments[5]}{arguments[6]}. The incircle touches the arm {arguments[4]}{arguments[5]} at {arguments[2]}, the arm {arguments[5]}{arguments[6]} at {arguments[0]} and the arm {arguments[6]}{arguments[4]} at {arguments[1]}. "
    elif element_name == 'excenter':
        # TODO(Partha): opportunity to augment! Symantic!
        return f"{arguments[0]} is the excentre of triangle triangle {arguments[1]}{arguments[2]}{arguments[3]} opposite to the angle {arguments[2]}{arguments[1]}{arguments[3]}. "
    elif element_name == 'excenter2':
        # TODO(Partha): opportunity to augment! Symantic! Also use respectively!
        return f"{arguments[3]} is the excentre of triangle {arguments[4]}{arguments[5]}{arguments[6]} opposite to the angle {arguments[5]}{arguments[4]}{arguments[6]}. The excircle touches the arm {arguments[4]}{arguments[5]} at {arguments[2]}, the arm {arguments[5]}{arguments[6]} at {arguments[0]} and the arm {arguments[6]}{arguments[4]} at {arguments[1]}. "
    elif element_name == 'centroid':
        # TODO(Partha): opportunity to augment! Symantic! permuting the points of the arms.
        return f"{arguments[4]}{arguments[5]}{arguments[6]} is a triangle. {arguments[3]} is the centroid of the triangle and {arguments[0]}, {arguments[1]}, {arguments[2]} are the midpoints of the sides {arguments[5]}{arguments[6]}, {arguments[4]}{arguments[6]}, {arguments[5]}{arguments[4]} respectively. "
    elif element_name == 'ninepoints':
        # TODO(Partha): opportunity to augment! Symantic! permuting the points of the arms.
        return f"{arguments[4]}{arguments[5]}{arguments[6]} is a triangle. {arguments[0]}, {arguments[1]}, {arguments[2]} are the midpoints of the sides {arguments[5]}{arguments[6]}, {arguments[4]}{arguments[6]}, {arguments[5]}{arguments[4]} respectively. {arguments[3]} is the centre of the circumcircle of the triangle {arguments[0]}{arguments[1]}{arguments[2]}."
    elif element_name == 'intersection_cc':
        # TODO(Partha): opportunity to augment! Symantic! just describe in text
        return f"{arguments[0]}{arguments[1]}{arguments[2]}{arguments[3]} is a quadrilateral where {arguments[1]}{arguments[3]} = {arguments[1]}{arguments[0]} and {arguments[2]}{arguments[3]} = {arguments[2]}{arguments[0]}. "
    elif element_name == 'intersection_lc':  # intersection line circle I think
        # TODO(Partha): Understood better?
        return f"{arguments[1]}{arguments[2]}{arguments[3]} is a triangle. The circle drawn with the centre at {arguments[2]} and passing through {arguments[3]} intersects the (possibly extended) line {arguments[1]}{arguments[3]} again at point {arguments[0]}. "
    elif element_name == 'intersection_ll':  # intersection between two lines
        # TODO(Partha, Felix): Mention they are not parallel?
        return f"Line {arguments[1]}{arguments[2]} and line {arguments[3]}{arguments[4]} intersect each other at {arguments[0]}. "
    elif element_name == 'intersection_lp':  # intersection between line and a pair of parallel lines
        return f"{arguments[4]}{arguments[5]} and {arguments[1]}{arguments[2]}  are two non parallel lines. {arguments[0]} lies on {arguments[1]}{arguments[2]} but not on {arguments[4]}{arguments[5]} such that {arguments[3]}{arguments[0]} is parallel to {arguments[4]}{arguments[5]}. "
    elif element_name == 'intersection_ltk':
        return f"{arguments[0]}{arguments[1]}{arguments[2]} is a triangle. "
    elif element_name == 'triangle':
        return f"{arguments[0]}{arguments[1]}{arguments[2]} is a triangle. "
    elif element_name == 'triangle':
        return f"{arguments[0]}{arguments[1]}{arguments[2]} is a triangle. "
    elif element_name == 'triangle':
        return f"{arguments[0]}{arguments[1]}{arguments[2]} is a triangle. "
    elif element_name == 'triangle':
        return f"{arguments[0]}{arguments[1]}{arguments[2]} is a triangle. "
    elif element_name == 'triangle':
        return f"{arguments[0]}{arguments[1]}{arguments[2]} is a triangle. "
    elif element_name == 'triangle':
        return f"{arguments[0]}{arguments[1]}{arguments[2]} is a triangle. "
    elif element_name == 'triangle':
        return f"{arguments[0]}{arguments[1]}{arguments[2]} is a triangle. "
    elif element_name == 'triangle':
        return f"{arguments[0]}{arguments[1]}{arguments[2]} is a triangle. "
    elif element_name == 'triangle':
        return f"{arguments[0]}{arguments[1]}{arguments[2]} is a triangle. "
    elif element_name == 'triangle':
        return f"{arguments[0]}{arguments[1]}{arguments[2]} is a triangle. "
    elif element_name == 'triangle':
        return f"{arguments[0]}{arguments[1]}{arguments[2]} is a triangle. "
    elif element_name == 'triangle':
        return f"{arguments[0]}{arguments[1]}{arguments[2]} is a triangle. "
    elif element_name == 'triangle':
        return f"{arguments[0]}{arguments[1]}{arguments[2]} is a triangle. "
    elif element_name == 'triangle':
        return f"{arguments[0]}{arguments[1]}{arguments[2]} is a triangle. "
    elif element_name == 'triangle':
        return f"{arguments[0]}{arguments[1]}{arguments[2]} is a triangle. "
    elif element_name == 'triangle':
        return f"{arguments[0]}{arguments[1]}{arguments[2]} is a triangle. "
    elif element_name == 'triangle':
        return f"{arguments[0]}{arguments[1]}{arguments[2]} is a triangle. "
    elif element_name == 'triangle':
        return f"{arguments[0]}{arguments[1]}{arguments[2]} is a triangle. "
    elif element_name == 'triangle':
        return f"{arguments[0]}{arguments[1]}{arguments[2]} is a triangle. "
    elif element_name == 'triangle':
        return f"{arguments[0]}{arguments[1]}{arguments[2]} is a triangle. "
    elif element_name == 'triangle':
        return f"{arguments[0]}{arguments[1]}{arguments[2]} is a triangle. "
    elif element_name == 'triangle':
        return f"{arguments[0]}{arguments[1]}{arguments[2]} is a triangle. "
    elif element_name == 'triangle':
        return f"{arguments[0]}{arguments[1]}{arguments[2]} is a triangle. "
    elif element_name == 'triangle':
        return f"{arguments[0]}{arguments[1]}{arguments[2]} is a triangle. "
    elif element_name == 'triangle':
        return f"{arguments[0]}{arguments[1]}{arguments[2]} is a triangle. "
    elif element_name == 'triangle':
        return f"{arguments[0]}{arguments[1]}{arguments[2]} is a triangle. "
    elif element_name == 'triangle':
        return f"{arguments[0]}{arguments[1]}{arguments[2]} is a triangle. "
    elif element_name == 'triangle':
        return f"{arguments[0]}{arguments[1]}{arguments[2]} is a triangle. "
    elif element_name == 'triangle':
        return f"{arguments[0]}{arguments[1]}{arguments[2]} is a triangle. "
    elif element_name == 'triangle':
        return f"{arguments[0]}{arguments[1]}{arguments[2]} is a triangle. "
    elif element_name == 'triangle':
        return f"{arguments[0]}{arguments[1]}{arguments[2]} is a triangle. "
    elif element_name == 'triangle':
        return f"{arguments[0]}{arguments[1]}{arguments[2]} is a triangle. "
    elif element_name == 'triangle':
        return f"{arguments[0]}{arguments[1]}{arguments[2]} is a triangle. "
    elif element_name == 'triangle':
        return f"{arguments[0]}{arguments[1]}{arguments[2]} is a triangle. "
    elif element_name == 'triangle':
        return f"{arguments[0]}{arguments[1]}{arguments[2]} is a triangle. "
    elif element_name == 'triangle':
        return f"{arguments[0]}{arguments[1]}{arguments[2]} is a triangle. "
    elif element_name == 'triangle':
        return f"{arguments[0]}{arguments[1]}{arguments[2]} is a triangle. "
    elif element_name == 'triangle':
        return f"{arguments[0]}{arguments[1]}{arguments[2]} is a triangle. "
    elif element_name == 'triangle':
        return f"{arguments[0]}{arguments[1]}{arguments[2]} is a triangle. "
    elif element_name == 'triangle':
        return f"{arguments[0]}{arguments[1]}{arguments[2]} is a triangle. "
    elif element_name == 'triangle':
        return f"{arguments[0]}{arguments[1]}{arguments[2]} is a triangle. "
    elif element_name == 'triangle':
        return f"{arguments[0]}{arguments[1]}{arguments[2]} is a triangle. "
    # Add more templates for other geometric elements as needed
    else:
        return '' #TODO(Partha): Remove this later!
        raise Exception(f"Element name {element_name} not recognized.")


def get_nl_problem_statement(fl_problem):
    # for now assuming that the problem statement doesn't have a goal
    clauses_fl = fl_problem.split(';')
    nl_problem = ''
    for clause_fl in clauses_fl:
        nl_problem += get_nl_clause(clause_fl)

    return nl_problem


def get_nl_clause(fl_clause):
    # Split the string by the equal sign
    parts = fl_clause.split('=')

    if len(parts) != 2:
        raise Exception(f"Unable to parse clause: {fl_clause}.")

    name_and_args = parts[1].split()

    # Extract geometric element name and its arguments
    element_name = name_and_args[0]
    arguments = name_and_args[1:]
    nl_clause = verbalize_clause(element_name, arguments)

    return nl_clause


if __name__ == '__main__':
    txt = 'x = angle_bisector x a b c'
    # txt = 'Zz x l = triangle Zz x l; y = angle_bisector y Z x l; a b c = angle_mirror x a b c; c = circle x a b c; c = circumcenter x a b c'
    # txt = 'Zz x l = eq_quadrangle a b c d; a c = eq_trapezoid a b c d; a = eq_triangle a b c; a = eqangle2 x a b c; a = eqdia_quadrangle a b c d'
    # txt = 'a = eqdistance x a b c; a = foot x a b c; a = incenter x a b c; c = incenter2 x y z i a b c'
    # txt = 'a = excenter2 x y z i a b c'
    # txt = 'a = intersection_cc x o w a'
    txt = 'a = intersection_lp x a b c m n'
    # txt = 'Z x l = triangle Z x l; V = excenter V Z l x; R = on_bline R l V; ' \
    #       'Yj Tv I4 Tt = ninepoints Yj Tv I4 Tt R Z V; f OR G b = trapezoid f OR G b'
    print(get_nl_problem_statement(txt))