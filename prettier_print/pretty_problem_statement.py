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
    elif element_name == 'intersection_lt':
        # TODO(Partha, Felix): Mention XC is not parallel to ab for this to make sense and therefore ab can't be
        #  perpendicular to de. Augmentations?
        return random.choice([f"{arguments[1]}{arguments[2]} and {arguments[4]}{arguments[5]} are two non " \
                             f"perpendicular lines. Line {arguments[0]}{arguments[3]} is perpendicular to " \
                             f"{arguments[4]}{arguments[5]} it intersects line" \
                             f" {arguments[1]}{arguments[2]} at {arguments[0]}. ",
                             f"The line perpendicular to {arguments[4]}{arguments[5]} intersects the " \
                             f"line {arguments[1]}{arguments[2]} at {arguments[0]}. ", ])
    elif element_name == 'intersection_pp':  # intersection of two pairs of parallel lines
        #TODO(Partha, Felix): The following is what I read from the defs! But what the hell is it trying to describe?
        # mustFix!
        return f"Line {arguments[0]}{arguments[1]} and line {arguments[2]}{arguments[3]} are parallel. " \
               f"Line {arguments[4]}{arguments[0]} and line {arguments[5]}{arguments[6]} are parallel. " \
               f"{arguments[0]} is the intersection point. "
    elif element_name == 'intersection_tt':  # intersection between two perpendicular lines
        # TODO(Partha, Felix): This can be augmented nicely
        return f"{arguments[2]}{arguments[3]} and {arguments[5]}{arguments[6]} are two non parallel lines. " \
               f"{arguments[0]}{arguments[1]} and {arguments[0]}{arguments[4]} are perpendicular lines to the " \
               f"aforementioned lines respectively. The Perpendicular lines intersect at {arguments[0]}. "
    elif element_name == 'iso_triangle':  # isosceles triangle
        # TODO(Partha): Also add the angle equal description
        return random.choice[f"{arguments[0]}{arguments[1]}{arguments[2]} is an isosceles triangle with base" \
                             f" {arguments[1]}{arguments[2]}. ",
                             f"{arguments[0]}{arguments[1]}{arguments[2]} is an isosceles triangle with" \
                             f" {arguments[0]}{arguments[1]} = {arguments[0]}{arguments[2]}. "]
    elif element_name == 'lc_tangent':  # line circle tangent
        # TODO(Partha, Felix): Write this better
        return random.choice[f"{arguments[0]}{arguments[1]} is the tangent to the circle centered on " \
                             f"{arguments[1]}{arguments[2]} (possibly extended) and passing through {arguments[1]}. ",
                             f"{arguments[0]}{arguments[1]} is perpendicular to {arguments[1]}{arguments[2]}"]
    elif element_name == 'midpoint':
        return f"{arguments[0]} is the midpoint of {arguments[1]}{arguments[2]}. "
    elif element_name == 'mirror':
        # TODO(Partha, Felix): This is a bit confusing! Fix it! i mean how would it appear in geometry problems?
        return f"{arguments[0]} is the mirror image of {arguments[1]} in {arguments[2]}. "
    elif element_name == 'nsquare':  # from the definitions it is same as psquare This troubling!
        #TODO(Felix): Do you see a better way of saying it? Lets keep all we can think of! Look at p square
        return f"{arguments[0]}{arguments[1]}{arguments[2]} is an isosceles right angle triangle with " \
               f"∠{arguments[0]}{arguments[1]}{arguments[2]} being the right angle. "
    elif element_name == 'on_aline':  # on same angle line
        # Perhaps construction element and we don't need to template it nicely?
        return f"{arguments[0]} is a point such that " \
               f"∠{arguments[0]}{arguments[1]}{arguments[2]} = ∠{arguments[3]}{arguments[4]}{arguments[5]}. "
    elif element_name == 'on_aline2':
        # Perhaps construction element and we don't need to template it nicely?
        return f"{arguments[0]} is a point such that " \
               f"∠{arguments[1]}{arguments[0]}{arguments[2]} = ∠{arguments[3]}{arguments[4]}{arguments[5]}. "
    elif element_name == 'on_bline':
        return f"{arguments[0]} lies on the perpendicular bisector of {arguments[1]}{arguments[2]}. "
    elif element_name == 'on_circle':
        return f"{arguments[0]} lies on the circle centered are {arguments[1]} with radios " \
               f"{arguments[1]}{arguments[2]}. "
    elif element_name == 'on_line':
        return f"{arguments[0]} lies on line {arguments[1]}{arguments[2]}. "
    elif element_name == 'on_pline':  # on parallel line
        #TODO(Partha): Shuffle line name points
        return f"{arguments[0]} is on the line passing through {arguments[1]} and parallel to" \
               f" {arguments[2]}{arguments[3]}. "
    elif element_name == 'on_tline':  # on perpendicular line
        return random.choice([f"{arguments[0]} is a point such that {arguments[0]}{arguments[1]} is perpendicular to " \
                             f"{arguments[1]}{arguments[2]}. ",
                             f"{arguments[0]} lies on the perpendicular line to {arguments[2]}{arguments[3]} " \
                             f"at {arguments[1]}. ", ])
    elif element_name == 'orthocenter':
        return f"{arguments[0]} is the orthocenter of triangle {arguments[1]}{arguments[2]}{arguments[3]}. "
    elif element_name == 'parallelogram':
        return f"{arguments[0]}{arguments[1]}{arguments[2]}{arguments[4]} is a parallelogram. "
    elif element_name == 'pentagon':
        return f"{arguments[0]}{arguments[1]}{arguments[2]}{arguments[3]}{arguments[4]} is a pentagon. "
    elif element_name == 'psquare':  # from the definitions it is same as nsquare This troubling!
        # TODO(Felix): Do you see a better way of saying it? Lets keep all we can think of! same as nsquare This troubling!
        return f"{arguments[0]}{arguments[1]}{arguments[2]} is an isosceles right angle triangle with " \
               f"∠{arguments[0]}{arguments[1]}{arguments[2]} being the right angle. "
    elif element_name == 'quadrangle':
        return random.choice[f"{arguments[0]}{arguments[1]}{arguments[2]}{arguments[3]} is a quadrangle. ",
                             f"{arguments[0]}{arguments[1]}{arguments[2]}{arguments[3]} is a quadrilateral."]
    elif element_name == 'r_trapezoid':
        #TODO(Partha): Augment with different def of perp
        return f"{arguments[0]}{arguments[1]}{arguments[2]}{arguments[3]} is a right angle trapezoid. " \
               f"With {arguments[0]}{arguments[1]} ⊥ {arguments[0]}{arguments[3]}."
    elif element_name == 'r_triangle':
        # TODO(Partha): Augment with different def of perp
        return f"{arguments[0]}{arguments[1]}{arguments[2]} is a triangle. " \
               f"With ∠{arguments[1]}{arguments[0]}{arguments[2]} being a right angle."
    elif element_name == 'rectangle':
        return f"{arguments[0]}{arguments[1]}{arguments[2]}{arguments[3]} is a rectangle. "
    elif element_name == 'reflect':
        # perhaps we don't have to augment it a lot as this seems to be a construction element
        return f"{arguments[0]} is the reflection of {arguments[1]} on {arguments[2]}{arguments[3]}. "
    elif element_name == 'risos':
        #TODO(Partha): Augment with different def of perp
        return f"{arguments[0]}{arguments[1]}{arguments[2]} is a right angle isosceles triangle. " \
               f"With ∠{arguments[1]}{arguments[0]}{arguments[2]} as the right angle."
    elif element_name == 's_angle':  # Assign that an angle is a constant degree
        # TODO(Partha): This won't be sampled correctly in the clause creator. FIX this there!
        return f"∠{arguments[0]}{arguments[1]}{arguments[2]} is {arguments[2]} degree. "
    elif element_name == 'segment':
        return f"{arguments[0]}{arguments[1]} is a line segment. "
    elif element_name == 'shift':
        #TODO(Partha, Felix): This is not understood yet! Fix this!
        raise Exception("Shift is not Understood yet!.")
        return f"{arguments[0]}{arguments[1]}{arguments[2]} is a triangle. "
    elif element_name == 'square':  # strange redefinition in isquare  # Attention!
        return f"{arguments[0]}{arguments[1]}{arguments[2]}{arguments[3]} is a square. "
    elif element_name == 'isquare':  # strange redefinition in square  # Attention!
        return f"{arguments[0]}{arguments[1]}{arguments[2]}{arguments[3]} is a square. "
    elif element_name == 'trapezoid':
        return f"{arguments[0]}{arguments[1]}{arguments[2]}{arguments[3]} is a trapezoid. "
    elif element_name == 'triangle':
        return f"{arguments[0]}{arguments[1]}{arguments[2]} is a triangle. "
    elif element_name == 'triangle12':  # Strange cache! # Attention!
        # TODO(Partha): Write more naturally! And this happens so often that there is aseparate definition fo this?!
        return f"{arguments[0]}{arguments[1]}{arguments[2]} is a triangle. " \
               f"With {arguments[0]}{arguments[1]} : {arguments[0]}{arguments[2]} = 1 : 2. "
    elif element_name == '2l1c':  # Acircle touching two lines and another circle
        # TODO(Partha, Felix): Verify and Augment.
        return f"{arguments[4]}{arguments[6]} is a line {arguments[5]}{arguments[6]} is another line. " \
               f"A circle centered at {arguments[7]} passes through {arguments[4]} and touches {arguments[5]}. " \
               f"A circle centered at {arguments[3]} touches {arguments[4]}{arguments[6]} at " \
               f"{arguments[0]}, {arguments[5]}{arguments[7]} at {arguments[1]} and the circle centered " \
               f"at {arguments[7]} at {arguments[2]} respectively. "
    elif element_name == 'e5128':
        # TODO(Partha, Felix): Verify and Augment. Attention! Might be WRONG!
        # Did not write the equal angles!
        return f"{arguments[3]}{arguments[0]}{arguments[5]} are points on a circle centered at {arguments[4]}. " \
               f"{arguments[2]}{arguments[3]} is tangent to the circle at {arguments[3]}. " \
               f"The line {arguments[0]}{arguments[5]} intersects {arguments[2]}{arguments[3]} at {arguments[1]}. "
    elif element_name == '3peq':
        raise Exception(f"Element name {element_name} not implemented.")
    elif element_name == 'trisect':
        raise Exception(f"Element name {element_name} not implemented.")
    elif element_name == 'trisegment':
        #TODO(Partha): Augment
        return f"Line segment {arguments[2]}{arguments[3]} is a divided into three equal segments by the points " \
               f"{arguments[0]} and {arguments[1]}. "
    elif element_name == 'on_dia':
        #TODO(Partha): Write better
        return f"{arguments[1]}{arguments[2]} is the diameter of a circle. " \
               f"{arguments[0]} is a point on the circumferance of the circle. "
    elif element_name == 'ieq_triangle':
        return f"{arguments[0]}{arguments[1]}{arguments[2]} is an equilateral triangle. "
    elif element_name == 'on_opline':  # on the extended line or on the linesegment of ab?
        #TODO(partha): Disambiguate and make better
        return random.choice([f"{arguments[0]}, {arguments[1]}, {arguments[2]} are colinear. ",
                              f"{arguments[0]} lies on the line {arguments[1]}{arguments[2]}. "])
    elif element_name == 'cc_tangent0':
        raise Exception(f"Element name {element_name} not implemented.")
    elif element_name == 'cc_tangent': # tangent to two circlles
        raise Exception(f"Element name {element_name} not implemented.")
    elif element_name == 'eqangle3': # tangent to two circlles
        return random.choice([f"angle {arguments[1]}{arguments[0]}{arguments[2]} is equal to "
                             f"angle {arguments[4]}{arguments[3]}{arguments[5]}",
                             f"∠{arguments[1]}{arguments[0]}{arguments[2]} = "
                             f"∠{arguments[4]}{arguments[3]}{arguments[5]}", ])
    elif element_name == 'tangent':  # tangent to a circlle
        #TODO(Partha, Felixm Max): Why did they define two tangents?
        return f"{arguments[2]}{arguments[0]} is the tangent to the circle centered at {arguments[3]} and passing" \
               f" through {arguments[4]} at {arguments[0]} and {arguments[2]}{arguments[1]} is tangent to the " \
               f"same circle at {arguments[1]}"
    elif element_name == 'on_circum':
        return f"{arguments[0]} is a point on te circumcircle of the triangle {arguments[1]}{arguments[2]}{arguments[3]}"

    # Add more templates for other geometric elements as needed
    else:
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
    # txt = 'a = intersection_lp x a b c m n; I = intersection_pp I w k mq e F d; a = excenter2 x y z i a b c'
    txt = 'a = on_circle x o a'
    txt = 'a = intersection_lt x a b c d e'
    # txt = 'Z x l = triangle Z x l; V = excenter V Z l x; R = on_bline R l V; ' \
    #       'Yj Tv I4 Tt = ninepoints Yj Tv I4 Tt R Z V; f OR G b = trapezoid f OR G b'
    print(get_nl_problem_statement(txt))