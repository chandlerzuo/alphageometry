from .imports import *

from .common import repo_root
from .relations import RelationManager, Relation
from .entities import EntityManager, Entity, ConstantEntity

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



class IndependentStatementVerbalization(AbstractVerbalization):
	'''
	verbalizes each statement in the problem independently, which is reasonable, but not ideal as it
	significantly simplifies translation, and limits the expressiveness of the NL
	'''
	def __init__(self, *, relation_path: Path = None, entity_path: Path = None, **kwargs):
		if relation_path is None:
			relation_path = repo_root() / 'Arithmetic' / 'arithmetic-def-patterns.yml'
		if entity_path is None:
			entity_path = repo_root() / 'Arithmetic' / 'arithmetic-entities.yml'
		super().__init__(**kwargs)
		self._relation_data = self._load_relation_data(relation_path)
		self._entity_data = self._load_entity_data(entity_path)

	_known_entities: dict[str, Entity] = None

	@staticmethod
	def _load_entity_data(entity_path: Path) -> dict[str, Any]:
		assert entity_path.exists(), f'Entity file not found: {entity_path}'
		assert entity_path.suffix in ['.yml', '.yaml'], f'Invalid entity file: {entity_path}'

		kinds = yaml.safe_load(entity_path.read_text())
		return kinds
	@staticmethod
	def _load_relation_data(patterns_path: Path, funcs: list[Callable] = None):
		assert patterns_path.exists(), f'Pattern file not found: {patterns_path}'
		assert patterns_path.suffix in ['.yml', '.yaml'], f'Invalid pattern file: {patterns_path}'

		if funcs is None:
			print(f'Using default relation functions in {repo_root() / "Arithmetic/defs.py"}')
			from . import defs
			fn_keys = [name for name in dir(defs) if not name.startswith('_')]
			funcs = [getattr(defs, name) for name in fn_keys]

		patterns = yaml.safe_load(patterns_path.read_text())

		fn_names = [fn.__name__ for fn in funcs]

		assert all(name in patterns for name in fn_names), (f'No pattern found for symbols: '
															f'{[name for name in fn_names if name not in patterns]}')

		assert all(pattern in fn_names for pattern in patterns), (f'No symbol found for patterns: '
																  f'{[pattern for pattern in patterns if pattern not in fn_names]}')

		# relations = [Relation(name, patterns[name], fn) for name, fn in zip(fn_names, funcs)]
		# return relations
		data = {name: {'fn': fn, 'name': name, **patterns.get(name, {})} for name, fn in zip(fn_names, funcs)}
		return data

	# def get_entity(self, name: str, kind: str) -> Entity:
	# 	if self._known_entities is None:
	# 		return self.create_entity(name, kind)
	# 	if name not in self._known_entities:
	# 		self._known_entities[name] = self.create_entity(name, kind)
	# 	assert self._known_entities[name].kind == kind, f'Kind mismatch: {self.known_entities[name].kind} != {kind}'
	# 	return self._known_entities[name]

	_default_kind = 'variable'
	def get_relation_output_kind(self, name: str) -> str:
		return self._relation_data[name].get('out', self._default_kind)
	def get_relation_input_kinds(self, name: str) -> list[str]:
		return self._relation_data[name].get('args', [self._default_kind]*self.get_relation_num_args(name))
	def get_relation_num_args(self, name: str) -> int:
		sig = inspect.signature(self._relation_data[name]['fn'])
		return len(list(sig.parameters.keys()))

	_Relation = Relation
	def create_relation(self, name: str, out_name: str, arg_names: Iterable[str]) -> Relation:
		entities = []
		args = []
		data = self._relation_data.get(name, {})
		return self._Relation(name, args=arg_names, out=out_name, data=data)


	_Entity = Entity
	def create_entity(self, ident: str, kind: str) -> Entity:
		return self._Entity(ident, kind=kind, data=self._entity_data.get(kind, {}))


	_ConstantEntity = ConstantEntity
	def create_constant_entity(self, ident: str, kind: str, value: Any) -> ConstantEntity:
		return self._ConstantEntity(ident, value, kind=kind, data=self._entity_data.get(kind, {}))


	def _collect_vars(self, fl_problem: str):
		vars = []
		statements = fl_problem.split(';')
		for statement in statements:
			tree = ast.parse(statement)
			assert len(tree.body) == 1 and isinstance(tree.body[0], ast.Assign) and len(tree.body[0].targets) == 1, \
				f'Invalid tree: {tree!r}'
			vars.append(tree.body[0].targets[0].id)
		return vars


	def parse_fl(self, fl_statement: str, *, vocab: dict[str, 'Entity'] = None) -> Controller:
		vocab = vocab or {}
		tree = ast.parse(fl_statement) # valid python syntax

		assert (len(tree.body) == 1 and isinstance(tree.body[0], ast.Assign)
				and isinstance(tree.body[0].value, ast.Call)), f'Invalid tree: {tree!r}'
		# arguments could be literals (ints) or variables
		# args = [arg.n if isinstance(arg, ast.Num) else arg.id for arg in tree.body[0].value.args]
		assert (len(tree.body) == 1 and isinstance(tree.body[0], ast.Assign)
				and isinstance(tree.body[0].value, ast.Call)), f'Invalid tree: {tree!r}'
		rel_name = tree.body[0].value.func.id

		assert len(tree.body) == 1 and isinstance(tree.body[0], ast.Assign) and len(tree.body[0].targets) == 1, \
			f'Invalid tree: {tree!r}'
		out_name = tree.body[0].targets[0].id

		arg_kinds = self.get_relation_input_kinds(rel_name)

		args = []
		entities = []

		for arg, kind in zip(tree.body[0].value.args, arg_kinds):
			if isinstance(arg, ast.Num):
				entity = self.create_constant_entity(self._generate_identifier(vocab), kind, arg.n)
			elif isinstance(arg, ast.Name) and arg.id in vocab:
				entity = vocab[arg.id]
			else:
				entity = self.create_entity(self._generate_identifier(vocab), kind)
			vocab[entity.ident] = entity
			entities.append(entity)
			args.append(entity.ident)

		rel = self.create_relation(rel_name, out_name, args)

		if out_name in vocab:
			out = vocab[out_name]
		else:
			out = self.create_entity(out_name, self.get_relation_output_kind(rel_name))
			vocab[out_name] = out

		return Controller(rel, out, *entities, DictGadget({'formal': fl_statement}))


	def fl_2_nl(self, clause_fl: str) -> str:
		ctx = self.parse_fl(clause_fl)
		return ctx['statement']


	def problem_fl_2_nl(self, fl_problem: str, *, seed: int = None) -> str:
		if seed is not None:
			raise NotImplementedError
		statements = fl_problem.split(';')

		vocab = {}
		ctxs = [self.parse_fl(statement, vocab=vocab) for statement in statements]

		# connect statements as needed
		for i, ctx in enumerate(ctxs):
			ctx.include(DictGadget({'past_ctxs': ctxs[:i],
									'future_ctxs': ctxs[i + 1:]}))

		lines = [ctx['statement'] for ctx in ctxs]
		return ' '.join(lines)


	def _generate_identifier(self, existing: list[str] = None) -> str:
		if existing is None:
			existing = []

		idx = len(existing)
		candidate = f'v{idx}'
		while candidate in existing:
			idx += 1
			candidate = f'v{idx}'
		return candidate



# class StatementVerbalization(Scope):
# 	def __init__(self, relation: Relation, out: Entity, args: list[Entity], **kwargs):
# 		super().__init__(**kwargs)




class ProblemVerbalization(ToolKit):
	@staticmethod
	def parse_formal_clause(clause: str):
		assert ';' not in clause and ',' not in clause, (f'Invalid clause: {clause!r} '
														 f'(could it be a statement or problem?)')
		prior = {'tree': ast.parse(clause)}
		return prior

		# terms = clause.split('=')
		# assert len(terms) <= 2, f'Invalid clause: {clause!r}'
		# if len(terms) > 1:
		# 	prior['variables'] = terms[0].strip().split()
		# 	prior.update({f'var{i}': var for i, var in enumerate(prior['variables'])})
		#
		# name, *args = terms[-1].strip().split()
		# prior['definition'] = name
		# prior['arguments'] = args
		# prior.update({f'arg{i}': arg for i, arg in enumerate(args)})
		# return prior


	@tool('return')
	def get_variable(self, tree: ast.Module):
		assert len(tree.body) == 1 and isinstance(tree.body[0], ast.Assign) and len(tree.body[0].targets) == 1, \
			f'Invalid tree: {tree!r}'
		return tree.body[0].targets[0].id


	@tool('symbol')
	def get_function(self, tree: ast.Module):
		assert (len(tree.body) == 1 and isinstance(tree.body[0], ast.Assign)
				and isinstance(tree.body[0].value, ast.Call)), f'Invalid tree: {tree!r}'
		return tree.body[0].value.func.id


	@tool('arguments')
	def get_arguments(self, tree: ast.Module):
		assert (len(tree.body) == 1 and isinstance(tree.body[0], ast.Assign)
				and isinstance(tree.body[0].value, ast.Call)), f'Invalid tree: {tree!r}'
		return [arg.n for arg in tree.body[0].value.args]
































