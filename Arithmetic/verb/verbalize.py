from .imports import *

from .common import repo_root
from .concepts import Question, Statement
from .relations import Relation
from .entities import Entity, ConstantEntity
from .planners import IndependentStatements

import ast


class AbstractVerbalization:
	def generate_fl_problem(self, seed: int = None) -> str:
		raise NotImplementedError('coming soon')


	def problem_fl_2_nl(self, fl_problem: str, *, seed: int = None) -> str:
		raise NotImplementedError


	def problem_nl_2_fl(self, nl_problem: str) -> str:
		raise NotImplementedError(f'this is too hard - is it even possible?')


	def create_relation(self, name: str, out_name: str, arg_names: Iterable[str]):
		raise NotImplementedError


	def create_entity(self, kind: str, name: str):
		raise NotImplementedError


	def create_constant_entity(self, kind: str, ident: str, value: int):
		raise NotImplementedError



class Verbalization(AbstractVerbalization):
	'''
	verbalizes each statement in the problem independently, which is reasonable, but not ideal as it
	significantly simplifies translation, and limits the expressiveness of the NL
	'''
	def __init__(self, planner: ToolKit = None, *, relation_path: Path = None, entity_path: Path = None, **kwargs):
		'''
		- planner: the planner to use to connect the statements, and verbalize entire problems (defaults to `IndependentStatements`)
		- relation_path: the path to the yaml file containing the relation patterns (defaults to `Arithmetic/arithmetic-def-patterns.yml`)
		- entity_path: the path to the yaml file containing the entity patterns (defaults to `Arithmetic/arithmetic-entities.yml`)
		'''
		if planner is None:
			planner = IndependentStatements()
		if relation_path is None:
			relation_path = repo_root() / 'Arithmetic' / 'arithmetic-def-patterns.yml'
		if entity_path is None:
			entity_path = repo_root() / 'Arithmetic' / 'arithmetic-entities.yml'
		super().__init__(**kwargs)
		self.planner = planner
		self._relation_data = self._load_relation_data(relation_path)
		self._entity_data = self._load_entity_data(entity_path)

	_known_entities: dict[str, Entity] = None


	@staticmethod
	def _load_relation_data(patterns_path: Path, funcs: list[Callable] = None):
		'''
		For each relation the key must be the name of the function, and there should be:
		- args: list of what the arguments are called (for templating purposes)
		- rules: dict where the key is the product, and values are the details to create the rule (see _process_rules)
		- templates: list of templates where any rules and args will be filled in (note that output is referred to as `out`)

		In addition to loading the yaml with the relation patterns, this also connects and verifies that the
		corresponding functions ar consistent (e.g. number of args)
		'''
		assert patterns_path.exists(), f'Pattern file not found: {patterns_path}'
		assert patterns_path.suffix in ['.yml', '.yaml'], f'Invalid pattern file: {patterns_path}'

		if funcs is None:
			print(f'Using default relation functions in {repo_root() / "Arithmetic/defs.py"}')
			from .. import defs
			fn_keys = [name for name in dir(defs) if not name.startswith('_')]
			funcs = [getattr(defs, name) for name in fn_keys]
			funcs = [fn() if isinstance(fn, type) else fn for fn in funcs]

		patterns = yaml.safe_load(patterns_path.read_text())

		fn_names = [fn.__name__ if hasattr(fn, '__name__') else fn.__class__.__name__ for fn in funcs]

		missing_patterns = [name for name in fn_names if name not in patterns]
		if missing_patterns:
			print(f'No pattern found for symbols: {missing_patterns}')

		missing_symbols = [name for name in patterns if name not in fn_names]
		if missing_symbols:
			print(f'No symbol found for patterns: {missing_symbols}')

		data = {name: {'fn': fn, 'name': name, **patterns.get(name, {})} for name, fn in zip(fn_names, funcs)}
		return data
	@staticmethod
	def _load_entity_data(entity_path: Path) -> dict[str, Any]:
		'''
		For each entity, the key must be the kind, and the values are the details to create the entity:
		- rules: same as for relations (see above)
		- templates: filled in to produce a verbalization of that variable
		'''
		assert entity_path.exists(), f'Entity file not found: {entity_path}'
		assert entity_path.suffix in ['.yml', '.yaml'], f'Invalid entity file: {entity_path}'

		kinds = yaml.safe_load(entity_path.read_text())
		return kinds


	# get some basic info from the relation data
	_default_kind = 'ingredient' # this must match an entity kind in the entity patterns yaml
	def get_relation_output_kind(self, name: str) -> str:
		return self._relation_data[name].get('out', self._default_kind)
	def get_relation_input_kinds(self, name: str) -> list[str]:
		return self._relation_data[name].get('kinds', [self._default_kind]*self.get_relation_num_args(name))
	def get_relation_num_args(self, name: str) -> int:
		sig = inspect.signature(self._relation_data[name]['fn'])
		return len(list(sig.parameters.keys()))


	_Relation = Relation
	def create_relation(self, name: str, out_name: str, arg_names: Iterable[str]) -> Relation:
		data = self._relation_data.get(name, {})
		return self._Relation(name, args=arg_names, out=out_name, data=data)


	_Entity = Entity
	def create_entity(self, ident: str, kind: str) -> Entity:
		return self._Entity(ident, kind=kind, data=self._entity_data.get(kind, {}))


	_ConstantEntity = ConstantEntity
	def create_constant_entity(self, ident: str, kind: str, value: Any) -> ConstantEntity:
		return self._ConstantEntity(value, ident, kind=kind, data=self._entity_data.get(kind, {}))


	def _collect_vars(self, fl_problem: str):
		vars = []
		statements = fl_problem.split(';')
		for statement in statements:
			tree = ast.parse(statement)
			assert len(tree.body) == 1 and isinstance(tree.body[0], ast.Assign) and len(tree.body[0].targets) == 1, \
				f'Invalid tree: {tree!r}'
			vars.append(tree.body[0].targets[0].id)
		return vars


	def parse_question(self, fl_statement: str) -> Question:
		query = fl_statement.replace('?', '').strip()
		q = Question(query)
		return q


	def parse_fl(self, fl_statement: str, *, vocab: dict[str, 'Entity'] = None) -> Controller:
		'''
		Top-level method to parse a single statement.

		- vocab: if the statement is part of a whole problem, the vocab keeps track of which entities have already
		been created to avoid multiplicity
		'''
		if vocab is None:
			vocab = {}

		if '?' in fl_statement: # questions are parsed separately
			q = self.parse_question(fl_statement)
			assert q.query in vocab, f'Unknown variable: {q.query}'
			return Controller(q)

		tree = ast.parse(fl_statement) # valid python syntax

		assert (len(tree.body) == 1 and isinstance(tree.body[0], ast.Assign)
				and isinstance(tree.body[0].value, ast.Call)), f'Invalid tree: {tree!r}'
		assert (len(tree.body) == 1 and isinstance(tree.body[0], ast.Assign)
				and isinstance(tree.body[0].value, ast.Call)), f'Invalid tree: {tree!r}'
		rel_name = tree.body[0].value.func.id

		assert len(tree.body) == 1 and isinstance(tree.body[0], ast.Assign) and len(tree.body[0].targets) == 1, \
			f'Invalid tree: {tree!r}'
		out_name = tree.body[0].targets[0].id

		arg_kinds = self.get_relation_input_kinds(rel_name)

		args = []
		entities = []

		# first all new entities are created
		for arg, kind in zip(tree.body[0].value.args, arg_kinds):
			if isinstance(arg, ast.Num):
				entity = self.create_constant_entity(self._generate_identifier(vocab), kind, arg.n)
			elif isinstance(arg, ast.Name) and arg.id in vocab:
				entity = vocab[arg.id]
			else:
				entity = self.create_entity(self._generate_identifier(vocab), kind)
			if entity.ident not in vocab: # for existing entities, don't add them to this statement's context
				vocab[entity.ident] = entity
				entities.append(entity)
			args.append(entity.ident)

		# then the relation is created (using just the identifiers)
		rel = self.create_relation(rel_name, out_name, args)

		if out_name in vocab:
			raise ValueError(f'Out variables should be unique: {out_name}')
			out = vocab[out_name]
		else:
			out = self.create_entity(out_name, self.get_relation_output_kind(rel_name))
			vocab[out_name] = out

		ctx = Controller(rel, out, *entities)
		return ctx


	def fl_2_nl(self, clause_fl: str) -> str:
		'''convienence method to veralize a single statement'''
		ctx = self.parse_fl(clause_fl)
		return ctx['statement']


	def parse_problem(self, fl_problem: str, *, seed: int = None) -> Controller:
		'''
		Top level method to verbalize a full problem (using the planner)
		'''
		if seed is not None:
			raise NotImplementedError
		statements = fl_problem.split(';')

		vocab = {}
		ctxs = [self.parse_fl(statement.strip(), vocab=vocab) for i, statement in enumerate(statements)]

		return self.planner.connect(ctxs)


	def problem_fl_2_nl(self, fl_problem: str, *, seed: int = None) -> str:
		'''convenience method'''
		ctx = self.parse_problem(fl_problem, seed=seed)
		return ctx['nl']


	def _generate_identifier(self, existing: list[str] = None) -> str:
		if existing is None:
			existing = []

		idx = len(existing)
		candidate = f'v{idx}'
		while candidate in existing:
			idx += 1
			candidate = f'v{idx}'
		return candidate
























