from .imports import *

from .common import repo_root
from .concepts import Concept
from .entities import Entity



class AbstractRelation:
	@property
	def output(self) -> Entity:
		raise NotImplementedError

	@property
	def inputs(self) -> tuple[Entity]:
		raise NotImplementedError



class RelationEvaluator(ToolKit):
	def __init__(self, fn: Callable, args: Iterable[str], **kwargs):
		super().__init__(**kwargs)
		self.args = args
		self.fn = fn


	@tool.from_context('value')
	def get_value(self, ctx):
		values = [ctx[self.gap(f'{name}_value')] for name in self.args]
		return self.fn(*values)



class Relation(Concept):
	def _process_patterns(self, data: dict[str, Any]):
		if 'rules' in data:
			rules = list(self._process_rules(data['rules']))
			self.extend(rules)

		if 'templates' in data:
			templates = data['templates']
			templater = self._process_templates('clause', templates)
			self.include(templater)

		self.include(RelationEvaluator(self.fn, self.inputs))

		# process templates and rules


	def __init__(self, name: str, args: tuple[str], out: str, fn: Callable = None,
				 gap: dict[str, str] = None,
				 data: dict[str, Any] = None, **kwargs):
		if fn is None:
			assert data is not None and 'fn' in data, f'No function provided for relation {name}'
			fn = data['fn']
		if gap is None:
			gap = {}
		if 'value' not in gap:
			gap['value'] = f'{out}_value'
		super().__init__(gap=gap, **kwargs)
		self._name = name
		self._args = args
		self._out = out
		self.fn = fn
		self.num_args = len(self._get_args(fn))
		self._process_patterns(data)
		self.gauge_apply({'value': f'{out}_value', 'return': f'{out}'})

	@property
	def name(self):
		return self._name
	@property
	def output(self) -> str:
		return self._out
	@property
	def inputs(self) -> tuple[str]:
		return tuple(self._args)

	def __eq__(self, other: 'Relation'):
		return other.name == self.name and other.inputs == self.inputs and other.output == self.output
	def __hash__(self):
		return hash((self.name, self.inputs, self.output))


	# def gizmo_from(self, gizmo: str) -> str:
	# 	'''Converts an external gizmo to its internal representation.'''
	# 	gizmo = super().gizmo_from(gizmo)
	# 	if gizmo.startswith(f'{self.out}_'):
	# 		return gizmo[len(self.out) + 1:]
	# 	return gizmo


	# def gizmo_to(self, gizmo: str) -> str:
	# 	'''Converts an internal gizmo to its external representation.'''
	# 	gizmo = super().gizmo_to(gizmo)
	#
	# 	if gizmo == 'return':
	# 		return self.out
	#
	# 	if gizmo.startswith(f'arg'):
	# 		if '_' in gizmo:
	# 			arg, key = gizmo.split('_', 1)
	# 			idx = int(arg[3:])
	# 			return f'{self.args[idx]}_{key}'
	# 		idx = int(gizmo[3:])
	# 		return self.args[idx]
	#
	# 	return gizmo


	@staticmethod
	def _get_args(fn):
		sig = inspect.signature(fn)
		return list(sig.parameters.keys())


	def __str__(self):
		return f'{self.name}({", ".join(self.inputs)}) -> {self.output}'




class TemplateRelation(Relation):
	pass



class RelationManager(UserDict):

	def __init__(self, relation_data: dict[str, Any] = None, *, funcs: list[Callable] = None,
				 patterns_path: Path = None, root: Path = None):
		if relation_data is None:
			if root is None:
				root = repo_root()
			if patterns_path is None:
				patterns_path = root / 'Arithmetic' / 'arithmetic-def-patterns.yml'
			if funcs is None:
				from . import defs
				fn_keys = [name for name in dir(defs) if not name.startswith('_')]
				funcs = [getattr(defs, name) for name in fn_keys]
			relation_data = self._load_default(funcs, patterns_path)
		else:
			assert funcs is None and patterns_path is None, f'Cannot specify both relation_data and funcs/patterns_path'
		super().__init__()
		self.data.update(relation_data)


	def from_formal(self, name: str, args: list[str], out: str) -> Relation:



		return self[identifier]









