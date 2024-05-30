from .imports import *

from .common import repo_root
from .relations import RelationManager, Relation
from .entities import EntityManager, Entity

import ast


class AbstractVerbalization:
	def generate_fl_problem(self, seed: int = None) -> str:
		raise NotImplementedError('coming soon')


	def problem_fl_2_nl(self, fl_problem: str, *, seed: int = None) -> str:
		raise NotImplementedError


	def problem_nl_2_fl(self, nl_problem: str) -> str:
		raise NotImplementedError(f'this is too hard - is it even possible?')



class IndependentStatementVerbalization(AbstractVerbalization):
	'''
	verbalizes each statement in the problem independently, which is reasonable, but not ideal as it
	significantly simplifies translation, and limits the expressiveness of the NL
	'''
	def __init__(self, relations: RelationManager = None, entities = None, **kwargs):
		if relations is None:
			relations = self._RelationManager()#.load()
		if entities is None:
			entities = self._EntityManager()
		super().__init__(**kwargs)
		self.relations = relations
		self.entities = entities

	_RelationManager = RelationManager
	_EntityManager = EntityManager


	def _generate_identifier(self, existing: list[str] = None) -> str:
		if existing is None:
			existing = []

		idx = len(existing)
		candidate = f'v{idx}'
		while candidate in existing:
			idx += 1
			candidate = f'v{idx}'
		return candidate


	def _collect_vars(self, fl_problem: str):
		vars = []
		statements = fl_problem.split(';')
		for statement in statements:
			tree = ast.parse(statement)
			assert len(tree.body) == 1 and isinstance(tree.body[0], ast.Assign) and len(tree.body[0].targets) == 1, \
				f'Invalid tree: {tree!r}'
			vars.append(tree.body[0].targets[0].id)
		return vars


	def parse_fl(self, fl_statement: str, vocab: dict[str, 'Entity'] = None, vars: list[str] = None) -> Controller:
		vocab = vocab or {}
		tree = ast.parse(fl_statement) # valid python syntax

		assert (len(tree.body) == 1 and isinstance(tree.body[0], ast.Assign)
				and isinstance(tree.body[0].value, ast.Call)), f'Invalid tree: {tree!r}'
		rel_name = tree.body[0].value.func.id
		rel = self.relations.from_formal(rel_name)

		assert (len(tree.body) == 1 and isinstance(tree.body[0], ast.Assign)
				and isinstance(tree.body[0].value, ast.Call)), f'Invalid tree: {tree!r}'
		# arguments could be literals (ints) or variables
		args = [arg.n if isinstance(arg, ast.Num) else arg.id for arg in tree.body[0].value.args]
		entities = []
		for arg in tree.body[0].value.args:
			if isinstance(arg, ast.Num):
				entities.append(self.entities.make(str(arg.n), arg.n))
			else:
				if arg.id not in vocab:
					vocab[arg.id] = self.entities.make(arg.id, arg.id)
				entities.append(vocab[arg.id])



		return Controller(StatementVerbalization(), DictGadget({'formal': fl_statement}))


	def fl_2_nl(self, clause_fl: str) -> str:
		ctx = self.parse_fl(clause_fl)
		return ctx['statement']


	def problem_fl_2_nl(self, fl_problem: str, *, seed: int = None) -> str:
		if seed is not None:
			raise NotImplementedError
		statements = fl_problem.split(';')
		verbs = []
		for fl_statement in statements:
			nl_statement = self.fl_2_nl(fl_statement)
			verbs.append(nl_statement)
		return ' '.join(verbs)



class StatementVerbalization(Scope):
	def __init__(self, relation: Relation, out: Entity, args: list[Entity], **kwargs):
		super().__init__(**kwargs)


	def






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
































