from .imports import *

import yaml
from pathlib import Path
from omnibelt import Class_Registry, load_csv
from omniply.core.genetics import GeneticGadget
# from omniply.apps import Template, GadgetDecision, SimpleDecision, Controller, Combination, Permutation
# from omniply.apps.decisions.abstract import CHOICE

# from .rules import Rule, Point, Line, Angle, Triangle, Circle, Quadrilateral, Trapezoid, Conjunction


def load_patterns(path: Path) -> list['Definition']:
	return ClauseGenerator.load_patterns(path)


class ClauseGenerator(GadgetDecision):
	@classmethod
	def load_patterns(cls, path: Path) -> list['Definition']:
		patterns = yaml.safe_load(path.read_text())
		defs = []
		for name, data in patterns.items():
			defs.append(Definition.from_data(name, data))
		return defs


	def __init__(self, definitions: list['Definition'] | list[str] | str = None, *,
				 choice_gizmo: str = 'definition', **kwargs):
		if definitions is None:
			definitions = list(Definition._registry.values())
		elif isinstance(definitions, str):
			definitions = [definitions]
		definitions = [Definition.find(name) if isinstance(name, str) else name for name in definitions]
		choices = {definition.name: definition for definition in definitions}
		super().__init__(choices=choices, choice_gizmo=choice_gizmo, **kwargs)



class AssignmentVerbalizer(ToolKit):
	_assignment_templates = [
		'define point{"" if len(variables) == 1 else "s"} {varlist}',
		'let {varlist} be {"a " if len(variables) == 1 else ""}point{"" if len(variables) == 1 else "s"}',
		'point{"" if len(variables) == 1 else "s"} {varlist} {"is" if len(variables) == 1 else "are"} defined',
		'{varlist} {"is" if len(variables) == 1 else "are"} defined',
		'{varlist} {"is a" if len(variables) == 1 else "are"} point{"" if len(variables) != 1 else "s"}',
	]
	def __init__(self, variables: list[str], *args, **kwargs):
		super().__init__(*args, **kwargs)
		self._populate(variables)


	def _populate(self, variables: list[str]):
		assert len(variables), f'{variables}'

		# assignment prefix
		options = [
			tool('assignments')(lambda: None), # ignores the assignments
			*[Template(tmpl, 'assignments') for tmpl in self._assignment_templates],
		]
		if len(variables) > 1:
			self.include(Conjunction('varlist', [f'var{i}' for i in range(len(variables))]))
		else:
			self.include(tool('varlist')(lambda var0: var0))
		if len(options) > 1:
			self.include(GadgetDecision(options, choice_gizmo='assignment_choice'))
		elif len(options) == 1:
			self.include(options[0])



class StatementVerbalization(ToolKit):
	'''meant for the verbalization of an existing formal statement (including assignments and subclauses)'''
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self._populate()


	def _populate(self):
		self.include(SimpleDecision('clause_joiner', ['and', 'where', 'such that']))


	@staticmethod
	def parse_formal_clause(clause: str):
		assert ';' not in clause and ',' not in clause, (f'Invalid clause: {clause!r} '
														 f'(could it be a statement or problem?)')
		prior = {}

		terms = clause.split('=')
		assert len(terms) <= 2, f'Invalid clause: {clause!r}'
		if len(terms) > 1:
			prior['variables'] = terms[0].strip().split()
			prior.update({f'var{i}': var for i, var in enumerate(prior['variables'])})

		name, *args = terms[-1].strip().split()
		prior['definition'] = name
		prior['arguments'] = args
		prior.update({f'arg{i}': arg for i, arg in enumerate(args)})
		return prior


	@tool('clause_contexts')
	def from_formal(self, formal: str, seed: int = None) -> list[Controller]:
		clauses = formal.split(',')
		if len(clauses) > 1:
			seeds = [None]*len(clauses)
			if seed is not None:
				rng = random.Random(seed)
				seeds = [rng.randint(0, 2**32) for _ in range(len(clauses))]
			return [ctx for clause, clause_seed in zip(clauses, seeds) for ctx in self.from_formal(clause, clause_seed)]

		prior = self.parse_formal_clause(clauses[0])

		tools = [DictGadget(prior), Definition.find(prior['definition'])]
		if 'variables' in prior:
			assert len(prior['variables'])
			tools.append(AssignmentVerbalizer(prior['variables']))
		return [Controller(*tools)]


	@tool('statement')
	def verbalize_clauses(self, clause_contexts: list[Controller], *, clause_joiner: str = 'and'):
		statement = clause_joiner.join(ctx['clause'] for ctx in clause_contexts)
		if not len(statement):
			return ''
		return f'{statement[0].upper()}{statement[1:]}.'



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
		gizmo = 'construction'

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


	_rule_prefixes = Rule._kind_registry
	def _populate_rules(self, rules: dict[str, Union[dict[str, Any], list[int]]] = None):

		tools = []

		for name, data in rules.items():
			if isinstance(data, (int, str, list)):
				options = [key for key in self._rule_prefixes if name.startswith(key)]
				assert len(options) == 1, f'unknown/ambiguous rule type inference for {name!r}: {options}'
				rule_type = self._rule_prefixes[options[0]].cls
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


	@tool('clause')
	def get_clause(self, construction: str, assignments: str = None):
		return construction if assignments is None else f'{assignments} such that {construction}'


	@tool('statement')
	def as_statement(self, clause: str):
		if not len(clause):
			return ''
		return f'{clause[0].upper()}{clause[1:]}.'
























