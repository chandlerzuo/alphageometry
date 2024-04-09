from .imports import *

from omnibelt import Class_Registry
from omniply.core.genetics import GeneticGadget
# from omniply.apps import Template, GadgetDecision, SimpleDecision, Controller, Combination, Permutation
# from omniply.apps.decisions.abstract import CHOICE

from .atoms import Point, Line, Angle


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



class Definition(ToolKit):
	_registry = {}
	@classmethod
	def find(cls, name: str | 'Definition'):
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


	@classmethod
	def _populate_clause_template(cls, templates: str | Iterable[str] | Mapping[str, str], gizmo: str, **kwargs):
		if isinstance(templates, str):
			templates = {0: templates}
		elif not isinstance(templates, Mapping) and isinstance(templates, Iterable):
			templates = {i: template for i, template in templates}
		assert templates is not None and len(templates), (f'No templates were specified for {cls.__name__} '
														  f'(set {cls.__name__}._templates '
														  f'or pass `templates` as an argument).')

		templates = {key: cls._Single_Template(template=template, gizmo=gizmo) for key, template in templates.items()}
		if len(templates) == 1:
			template = next(iter(templates.values()))
			return template
		return cls._Multi_Template(templates=templates, gizmo=gizmo, **kwargs)


	_atom_prefixes = {
		'point': Point,
		'line': Line,
		'angle': Angle,
	}
	def _populate_atoms(self, atoms: dict[str, Union[dict[str, Any], list[int]]] = None):
		raise NotImplementedError


	_argument_prefix = 'arg'
	def _populate_arguments(self, num_args: int):

		assert num_args <= 26, f'Cannot have more than 26 arguments for a definition: {num_args}'




		raise NotImplementedError

	def __init__(self, name: str, num_args: int, *, templates: str | Iterable[str] | Mapping[str, str] = None,
				atoms: dict[str, Union[dict[str, Any], list[int]]] = None,
				clause_gadget: AbstractGadget = None, gizmo: str = 'clause', **kwargs):
		if clause_gadget is None:
			clause_gadget = self._create_default_clause_template(templates, gizmo=gizmo)
		super().__init__(**kwargs)
		self._name = name
		self._num_args = num_args
		self.include(clause_gadget)
		# self.extend()
		self._registry[name] = self


	@property
	def name(self):
		return self._name


	@tool.from_context('formal')
	def get_formal(self, ctx):
		return f'{self._name} {" ".join(ctx[parent] for parent in self.formal_argument_names())}'
	@get_formal.parents
	def formal_argument_names(self):
		return [f'{self._argument_prefix}{i}' for i in range(self._num_args)]



class AngleBisector(Definition):
	_num_args = 4

	_atoms = {
		'angle1': [1, 2, 0],
		'angle2': [0, 2, 3],
	}

	_templates = [
		"{p0} is a point such that ∠{p1}{p2}{p0} = ∠{p0}{p2}{p3}",
		'{p0} bisects {angle1} and {angle2}'
	]

























