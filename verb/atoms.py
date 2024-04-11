from .imports import *


class Atom(MultiGadgetBase):
	_kind_registry = Class_Registry()
	@classmethod
	def find(cls, kind: str):
		return cls._kind_registry.get_class(kind)


	def __init_subclass__(cls, kind: str = None, **kwargs):
		super().__init_subclass__(**kwargs)
		if kind is not None:
			cls._kind_registry.new(kind, cls)


	_fill_in_abstract = False
	@classmethod
	def set_abstraction(cls, fill_in_abstract: bool = None):
		if fill_in_abstract is None:
			fill_in_abstract = not cls._fill_in_abstract
		cls._fill_in_abstract = fill_in_abstract
		return cls._fill_in_abstract


	_part_key_fmt = 'p{}'
	def _process_parts(self, raw_parts: Iterable[str | int]) -> list[str]:
		parts = [self._part_key_fmt.format(part) if isinstance(part, int) else part for part in raw_parts]
		return parts


	_fixed_num_parts = None
	def __init__(self, name: str, parts: list[str | int] | tuple = (), **kwargs):
		if len(parts):
			parts = self._process_parts(parts)
		if self._fixed_num_parts is not None:
			assert len(parts) == self._fixed_num_parts, (f'{self.__class__.__name__} expects '
														 f'{self._fixed_num_parts} parts, '
														   f'but got {len(parts)}: {parts}')
		super().__init__(**kwargs)
		self._name = name
		self._parts = tuple(parts)


	@property
	def name(self):
		return self._name


	@property
	def parts(self):
		return self._parts


	@property
	def num_parts(self):
		return len(self.parts)


	@property
	def is_abstract(self):
		return self._fill_in_abstract


	def _as_abstract(self, ctx: 'AbstractGame'):
		raise NotImplementedError


	def grab_from(self, ctx: 'AbstractGame', gizmo: str) -> Any:
		if gizmo == self.name and self.is_abstract:
			out = self._as_abstract(ctx)
			if out is not None:
				return out
		return super().grab_from(ctx, gizmo)



class AtomTemplate(Atom, Template):
	def __init__(self, name: str, parts: list[str | int] = (), gizmo=None, **kwargs):
		if gizmo is None:
			gizmo = name
		super().__init__(name, parts, template=None, gizmo=gizmo, **kwargs)


	@property
	def template(self):
		if self._template is None:
			self._template = self._build_template(self._base_template, self.parts)
		return self._template


	_base_template = None
	@staticmethod
	def _build_template(base_template: str, parts: Iterable[str]) -> str:
		'''
		Returns a template trivially filled in with the given arguments.

		Args:
			base_template (str): The template to fill in with indexed arguments (eg. '{0}{1}').
			args (str): The arguments to fill in, generally should be gizmos (eg. 'p1', 'p2').

		Returns:
			str: A template that can be used to generate a gizmo.
		'''
		if base_template is None:
			base_template = ''.join(f'{{{i}}}' for i in range(len(parts)))
		fixed = [part if part.endswith('}') and part.startswith('{') else f'{{{part}}}' for part in parts]
		return base_template.format(*fixed)


	def _as_abstract(self, ctx: 'AbstractGame'):
		reqs = {key: f'{{{key}}}' for key in self.parts}
		return self.fill_in(reqs)



class Point(AtomTemplate, kind='point'):
	_fixed_num_parts = 1
	_part_key_fmt = 'arg{}'

	def _as_abstract(self, ctx: 'AbstractGame'):
		reqs = {key: f'{{{key}}}'.replace('arg', 'p') for key in self.parts}
		return self.fill_in(reqs)



class Line(AtomTemplate, kind='line'):
	_fixed_num_parts = 2
	_base_template = 'line {0}{1}'


class Angle(AtomTemplate, kind='angle'):
	_fixed_num_parts = 3
	_base_template = '∠{0}{1}{2}'


class Triangle(AtomTemplate, kind='triangle'):
	_fixed_num_parts = 3
	_base_template = 'triangle {0}{1}{2}'


class Circle(AtomTemplate, kind='circle'):
	_fixed_num_parts = 4
	_base_template = 'circle {1}{2}{3} centered at {0}'


class Quadrilateral(AtomTemplate, kind='quadrilateral'):
	_fixed_num_parts = 4
	_base_template = 'quadrilateral {0}{1}{2}{3}'


class Trapezoid(AtomTemplate, kind='trapezoid'):
	_fixed_num_parts = 4
	_base_template = 'trapezoid {0}{1}{2}{3}'


class Parallelogram(AtomTemplate, kind='parallelogram'):
	_fixed_num_parts = 4
	_base_template = 'parallelogram {0}{1}{2}{3}'


class Rhombus(AtomTemplate, kind='rhombus'):
	_fixed_num_parts = 4
	_base_template = 'rhombus {0}{1}{2}{3}'


class Rectangle(AtomTemplate, kind='rectangle'):
	_fixed_num_parts = 4
	_base_template = 'rectangle {0}{1}{2}{3}'


class Square(AtomTemplate, kind='square'):
	_fixed_num_parts = 4
	_base_template = 'square {0}{1}{2}{3}'



# Special (complex) atoms

from .common import Enumeration


class ComplexAtom(Atom):
	def _as_abstract(self, ctx: 'AbstractGame'):
		ctx[f'{self.name}_order'] = 0
		for key in self.parts:
			ctx[key] = f'{{{key}}}'
		return # none to default to previous behavior



class Conjunction(ComplexAtom, ToolKit, kind='conjunction'):
	_term_options = [' and ', ' & ', None] # TODO: maybe split up to properly manage the oxford comma

	def __init__(self, name: str, elements: list[str], **kwargs):
		super().__init__(name, elements, **kwargs)
		self.include(SimpleDecision(f'{name}_term', self._term_options))
		self.include(Enumeration(elements, gizmo=name, oxford=True,
								 aggregator_gizmo=f'{name}_term', choice_gizmo=f'{name}_order'))



class Equality(ComplexAtom, ToolKit, kind='equality'):
	def __init__(self, name: str, elements: list[str], **kwargs):
		super().__init__(name, elements, **kwargs)

		options = []

		eqsign = Enumeration(elements, delimiter=' = ', gizmo=name, choice_gizmo=f'{name}_order')
		options.append(eqsign)

		if len(elements) > 2:
			terms = SimpleDecision(f'{name}_term', [' all equal ', ' are all the same as ', ' are all congruent to '])
			order = Enumeration(elements, gizmo=name, aggregator_gizmo=f'{name}_term', choice_gizmo=f'{name}_order')
			options.append(ToolKit().include(terms, order))
		elif len(elements) == 2:
			terms = SimpleDecision(f'{name}_term', [' equals ', ' is equal to ', ' is the same as ', ' is congruent to '])
			order = Enumeration(elements, gizmo=name, aggregator_gizmo=f'{name}_term', choice_gizmo=f'{name}_order')
			options.append(ToolKit().include(terms, order))

		self.include(GadgetDecision(options, choice_gizmo=f'{name}_fmt') if len(options) > 1 else options[0])



class Disjunction(ComplexAtom, Enumeration, kind='disjunction'):
	_aggregator = ' or '

	def __init__(self, name: str, elements: list[str], **kwargs):
		super().__init__(name, elements, oxford=True, element_gizmos=elements, gizmo=name,
						 choice_gizmo=f'{name}_order', **kwargs)







