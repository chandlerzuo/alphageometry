from .imports import *

from .common import repo_root



class RelationEvaluator(ToolKit):
	def __init__(self, fn: Callable, args: list[str], **kwargs):
		super().__init__(**kwargs)
		self.args = args
		self.fn = fn


	@tool.from_context('value')
	def get_value(self, ctx):
		values = [ctx[f'arg{i}_value'] for i in range(len(self.args))]
		return self.fn(*values)



class Relation(Scope):
	def _populate_defaults(self):
		self.include(RelationEvaluator(self.fn, self.args))

		# process templates and rules



	def __init__(self, name: str, fn: Callable, args: list[str], out: str, gap: dict[str, str] = None,
				 rules: dict[str, Any] = None, templates: list[str] = None, **kwargs):
		if gap is None:
			gap = {}
		if 'value' not in gap:
			gap['value'] = f'{out}_value'
		super().__init__(**kwargs)
		self.name = name
		self.rules = rules
		self.templates = templates
		self.args = args
		self.out = out
		self.fn = fn
		self.num_args = len(self._get_args(fn))
		self._populate_defaults()


	def gizmo_from(self, gizmo: str) -> str:
		'''Converts an external gizmo to its internal representation.'''
		gizmo = super().gizmo_from(gizmo)
		if gizmo.startswith(f'{self.out}_'):
			return gizmo[len(self.out) + 1:]
		return gizmo


	def gizmo_to(self, gizmo: str) -> str:
		'''Converts an internal gizmo to its external representation.'''
		gizmo = super().gizmo_to(gizmo)

		if gizmo == 'return':
			return self.out

		if gizmo.startswith(f'arg'):
			if '_' in gizmo:
				arg, key = gizmo.split('_', 1)
				idx = int(arg[3:])
				return f'{self.args[idx]}_{key}'
			idx = int(gizmo[3:])
			return self.args[idx]

		return gizmo


	@staticmethod
	def _get_args(fn):
		sig = inspect.signature(fn)
		return list(sig.parameters.keys())


	def __str__(self):
		return self.name


	@property
	def formal(self):
		return self.name




class TemplateRelation(Relation):
	pass



class RelationManager(UserDict):
	@staticmethod
	def _load_default(funcs: list[Callable], patterns_path: Path):
		assert patterns_path.exists(), f'Pattern file not found: {patterns_path}'
		assert patterns_path.suffix in ['.yml', '.yaml'], f'Invalid pattern file: {patterns_path}'

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









