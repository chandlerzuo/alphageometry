
from .imports import *

from .verbalize import Verbalization

import ast



def test_ast_parsing():
	prob = 'A = add(3, 2); B = quat_op1(10, 5, 9, 1); C = mul(A, B); D = add(8, 10); E = div(8, 8); F = bin_op2(D, E); G = div(C, F); G ?'
	formal = 'A = add(3, 2)'

	tree = ast.parse(formal)

	assert 'A' == tree.body[0].targets[0].id
	assert 'add' == tree.body[0].value.func.id
	assert [3, 2] == [arg.n for arg in tree.body[0].value.args]



def test_entity():

	verb = Verbalization()

	ent = verb.create_constant_entity(None, 'ingredient', 6)

	print(list(ent.gizmos()))
	print()
	# print(ent)
	for _ in range(100):
		print(Controller(ent)['nat'])




def test_relation():
	formal = 'A = add(3, 2)'

	verb = Verbalization()


	print()

	for _ in range(10):

		ctx = verb.parse_fl(formal)
		print(ctx['statement'])

		cert = ctx.certificate()

		ctx = verb.parse_fl(formal)
		ctx['is_solution'] = True
		ctx.update(cert)
		print(ctx['statement'])
		print()



def test_problem():
	prob = 'A = add(3, 2); B = leap_of_faith(10, 5, 9, 1); C = mul(A, B); D = add(8, 10); E = div(8, 8); F = risky_trick(D, E); G = div(C, F); G ?'

	verb = Verbalization()

	for _ in range(10):

		ctx = verb.parse_problem(prob)

		print()
		print(ctx['nl'])

		print()
		print(ctx['answer'])



def test_problem2():
	prob = 'A = add(3, 2); B = leap_of_faith(10, 5, 9, 1); C = mul(A, B); C ?'

	verb = Verbalization()

	for _ in range(10):

		ctx = verb.parse_problem(prob)

		print()
		print(ctx['nl'])

		print()
		print(ctx['answer'])




