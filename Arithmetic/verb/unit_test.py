
from .imports import *

from .verbalize import IndependentStatementVerbalization

import ast



def test_single_statement():
	prob = 'A = add(3, 2); B = quat_op1(10, 5, 9, 1); C = mul(A, B); D = add(8, 10); E = div(8, 8); F = bin_op2(D, E); G = div(C, F); G ?'
	formal = 'A = add(3, 2)'

	tree = ast.parse(formal)

	assert 'A' == tree.body[0].targets[0].id
	assert 'add' == tree.body[0].value.func.id
	assert [3, 2] == [arg.n for arg in tree.body[0].value.args]



def test_verb():

	verb = IndependentStatementVerbalization()

	print(verb)






