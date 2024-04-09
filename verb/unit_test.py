from .imports import *

from .common import ArgumentGenerator
from .atoms import Point, Line, Angle



def test_arg_sampling():

	random.seed(11)

	ctx = Context(ArgumentGenerator(5))

	assert ctx['arguments'] == ('I', 'W', 'P', 'Y', 'Q')

	assert ctx['arg0'] == 'I'
	assert ctx['arg4'] == 'Q'

	assert ctx['selection'] == ('I', 'P', 'Q', 'W', 'Y') # (unshuffled)
	assert ctx['selection_id'] == 59294

	assert ctx['order'] == (0, 3, 1, 4, 2)
	assert ctx['order_id'] == 13



def test_atoms():

	formal = 'test-obj A B C D E'
	args = formal.split()[1:]

	ctx = Context() # treat like a dict which lazily evaluates values that haven't been cached

	# populate the context with points (should be done by default)
	ctx.extend([Point(f'p{i}', i) for i in range(len(args))])

	# optionally include any relevant lines or angles
	ctx.include(Line('line1', 0, 2), Line('line2', 1, 3))
	ctx.include(Angle('angle1', 0, 1, 2), Angle('angle2', 1, 2, 3))

	# set the original arguments
	ctx.update({f'arg{i}': arg for i, arg in enumerate(args)})

	assert ctx['line1'] == 'line AC' # 'line1' is evaluated using the corresponding Line object
	assert ctx['line2'] == 'line BD'

	assert ctx['angle1'] == '∠ABC'
	assert ctx['angle2'] == '∠BCD'

	assert ctx['p3'] == 'D'


from .atoms import Conjunction


def test_conjunction():

	random.seed(19)

	ctx = Controller(ArgumentGenerator(5), Conjunction('conj', ['arg1', 'arg2', 'arg0']))

	ctx['arg1']

	print()

	cases = []
	for case in ctx.consider():
		print(case['conj'])
		cases.append(case)

	print(len(cases))







