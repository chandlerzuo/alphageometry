from .imports import *

from omnibelt import Class_Registry
from omniply.core.genetics import GeneticGadget
# from omniply.apps import Template, GadgetDecision, SimpleDecision, Controller, Combination, Permutation
# from omniply.apps.decisions.abstract import CHOICE

from .rules import Rule, Point, Line, Angle, Triangle, Circle, Quadrilateral, Trapezoid


def from_formal(formal_statement: str):
	if ',' in formal_statement or ';' in formal_statement or '=' in formal_statement:
		raise NotImplementedError('for now')

	terms = formal_statement.split()

	name, *args = terms

	literal = DictGadget({f'arg{i}': arg for i, arg in enumerate(args)}, arguments=args)
	return Context(literal, Definition.find(name))  # TODO: move to a ToolKit subclass to clean up



class StatementGenerator(GadgetDecision):
	def __init__(self, definitions: list['Definition'] | list[str] | str = None, *,
				 choice_gizmo: str = 'def', **kwargs):
		if definitions is None:
			definitions = list(Definition._registry.values())
		elif isinstance(definitions, str):
			definitions = [definitions]
		definitions = [Definition.find(name) if isinstance(name, str) else name for name in definitions]
		choices = {definition.name: definition for definition in definitions}
		super().__init__(choices=choices, choice_gizmo=choice_gizmo, **kwargs)


from .common import ArgumentGenerator


class Definition(ToolKit):
	_registry = {}
	@classmethod
	def find(cls, name: Union[str, 'Definition']):
		if isinstance(name, Definition):
			return name
		if name not in cls._registry:
			raise KeyError(f'No definition found with name {name!r}')
		return cls._registry[name]


	_Multi_Template = GadgetDecision
	_Single_Template = Template


	@classmethod
	def from_data(cls, key: str, data: dict[str, Any]):
		name = data.pop('name', key)
		return cls(name=name, **data)

	def _populate_arguments(self, num_args: int, shuffle=True, **kwargs):
		points = [Point(f'p{i}', [i]) for i in range(num_args)]
		self.extend(points)
		self.include(ArgumentGenerator(num_args, shuffle=shuffle, **kwargs))


	def _populate_clause_template(self, templates: str | Iterable[str] | Mapping[str, str], **kwargs):
		gizmo = 'clause'

		if isinstance(templates, str):
			templates = {0: templates}
		elif not isinstance(templates, Mapping) and isinstance(templates, Iterable):
			templates = {i: template for i, template in enumerate(templates)}
		assert templates is not None and len(templates), (f'No templates were specified for {self.__name__} '
														  f'(set {self.__name__}._templates '
														  f'or pass `templates` as an argument).')

		templates = {key: self._Single_Template(template=template, gizmo=gizmo) for key, template in templates.items()}
		if len(templates) == 1:
			template = next(iter(templates.values()))
			clause_gadget = template
		else:
			clause_gadget = self._Multi_Template(templates, choice_gizmo='def_template', **kwargs)

		self.include(clause_gadget)


	_rule_prefixes = {
		'point': Point,
		'line': Line,
		'angle': Angle,
		'triangle': Triangle,
		'circle': Circle,
		'quad': Quadrilateral,
		'trapezoid': Trapezoid,
	}
	def _populate_rules(self, rules: dict[str, Union[dict[str, Any], list[int]]] = None):

		tools = []

		for name, data in rules.items():
			if isinstance(data, (int, str, list)):
				options = [key for key in self._rule_prefixes if name.startswith(key)]
				assert len(options) == 1, f'unknown/ambiguous rule type inference for {name}: {options}'
				rule_type = self._rule_prefixes[options[0]]
				tools.append(rule_type(name, data))
			else:
				assert isinstance(data, dict) and 'type' in data and 'args' in data, \
					f'invalid rule data for {name}: {data}'
				rule_type = Rule.find(data['type'])
				tools.append(rule_type(name, data['args']))

		self.extend(tools)


	def __init__(self, name: str, num_args: int, *, templates: str | Iterable[str] | Mapping[str, str] = None,
				rules: dict[str, Union[dict[str, Any], list[int]]] = None, **kwargs):
		super().__init__(**kwargs)
		self._name = name
		self._num_args = num_args
		self._populate_arguments(num_args)
		if rules is not None:
			self._populate_rules(rules)
		self._populate_clause_template(templates)
		self._registry[name] = self


	@property
	def name(self):
		return self._name


	@tool.from_context('formal')
	def get_formal(self, ctx):
		return f'{self._name} {" ".join(ctx[parent] for parent in self.formal_argument_names())}'
	@get_formal.parents
	def formal_argument_names(self):
		return [f'arg{i}' for i in range(self._num_args)]


























