from .imports import *



class Atom(Template):
	_num_args = None
	_index_prefix = 'p'
	_base_template = None
	def __init__(self, name: str, *args: str | int):
		if self._num_args is not None:
			assert len(args) == self._num_args, (f'{self.__class__.__name__} expects {self._num_args} arguments, '
												 f'but got {len(args)}: {args}')
		super().__init__(template=None, gizmo=name)
		self._raw_args = args


	@property
	def template(self):
		if self._template is None:
			self._template = self._build_template(self._base_template, *self._raw_args,
												  prefix=self._index_prefix)
		return self._template


	@staticmethod
	def _build_template(base_template: str, *args: str | int, prefix: str = '') -> str:
		'''
		Returns a template trivially filled in with the given arguments.

		Args:
			base_template (str): The template to fill in with indexed arguments (eg. '{0}{1}').
			args (str): The arguments to fill in, generally should be gizmos (eg. 'p1', 'p2').

		Returns:
			str: A template that can be used to generate a gizmo.
		'''
		if base_template is None:
			base_template = ''.join(f'{{{i}}}' for i in range(len(args)))
		fixed = [f'{prefix}{arg}' if isinstance(arg, int) else arg for arg in args]
		fixed = [arg if arg.endswith('}') and arg.startswith('{') else f'{{{arg}}}' for arg in fixed]
		return base_template.format(*fixed)



class Point(Atom):
	_num_args = 1
	_index_prefix = 'arg'


class Line(Atom):
	_num_args = 2
	_base_template = 'line {0}{1}'


class Angle(Atom):
	_num_args = 3
	_base_template = '∠{0}{1}{2}'


class Triangle(Atom):
	_num_args = 3
	_base_template = 'triangle {0}{1}{2}'


class Circle(Atom):
	_num_args = 4
	_base_template = 'circle centered at {0} going through {1}, {2}, and {3}'


class Quadrilateral(Atom):
	_num_args = 4
	_base_template = 'quadrilateral {0}{1}{2}{3}'


class Trapezoid(Atom):
	_num_args = 4
	_base_template = 'trapezoid {0}{1}{2}{3}'


class Parallelogram(Atom):
	_num_args = 4
	_base_template = 'parallelogram {0}{1}{2}{3}'


class Rhombus(Atom):
	_num_args = 4
	_base_template = 'rhombus {0}{1}{2}{3}'


class Rectangle(Atom):
	_num_args = 4
	_base_template = 'rectangle {0}{1}{2}{3}'


class Square(Atom):
	_num_args = 4
	_base_template = 'square {0}{1}{2}{3}'



# Special (complex) atoms

from .common import Enumeration



class Conjunction(ToolKit):
	_term_options = [' and ', ' & ', None] # TODO: maybe split up to properly manage the oxford comma

	def __init__(self, name: str, elements: list[str], **kwargs):
		super().__init__(**kwargs)
		self.include(SimpleDecision(f'{name}_term', self._term_options))
		self.include(Enumeration(elements, gizmo=name, aggregator_gizmo=f'{name}_term', choice_gizmo=f'{name}_order'))



class Equality(ToolKit):
	def __init__(self, name: str, elements: list[str], **kwargs):
		super().__init__(**kwargs)

		options = []

		eqsign = Enumeration(elements, delimiter=' = ', gizmo=name, choice_gizmo=f'{name}_order')
		options.append(eqsign)

		if len(elements) > 2:
			terms = SimpleDecision(f'{name}_term', [' all equal ', ' are all the same as '])
			order = Enumeration(elements, aggregator_gizmo=f'{name}_term', choice_gizmo=f'{name}_order')
			options.append(ToolKit().include(terms, order))
		elif len(elements) == 2:
			terms = SimpleDecision(f'{name}_term', [' equals ', ' is equal to ', ' is the same as '])
			order = Enumeration(elements, aggregator_gizmo=f'{name}_term', choice_gizmo=f'{name}_order')
			options.append(ToolKit().include(terms, order))

		self.include(GadgetDecision(options, choice_gizmo=f'{name}_fmt') if len(options) > 1 else options[0])



class Disjunction(Enumeration):
	_aggregator = ' or '

	def __init__(self, name: str, elements: list[str], **kwargs):
		super().__init__(elements, gizmo=name, choice_gizmo=f'{name}_order', **kwargs)
		self.include(SimpleDecision(f'{name}_aggregator', [' or ', ' | ', '']))







