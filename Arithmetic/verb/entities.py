from .imports import *

from .common import repo_root
from .concepts import Concept



class Entity(Concept):
	def _process_patterns(self, data: dict[str, Any]):
		if 'rules' in data:
			rules = list(self._process_rules(data['rules']))
			self.extend(rules)

		if 'templates' in data:
			templates = data['templates']
			templater = self._process_templates('instance', templates)
			self.include(templater)


	def __init__(self, ident: str = None, kind: str = None, data: dict[str, Any] = None,
				 gap: dict[str, str] = None, **kwargs):
		gap = gap or {}
		super().__init__(**kwargs)

		if ident is not None:
			if 'nat' not in gap:
				gap['nat'] = ident
			for gizmo in self.gizmos():
				if gizmo not in gap:
					gap[gizmo] = f'{ident}_{gizmo}'

		self._ident = ident
		self._kind = kind
		# self.data = data
		self._process_patterns(data) # adds rules and templates
		self.gauge_apply(gap) # relabels all products


	@property
	def kind(self):
		return self._kind
	@tool('kind')
	def get_kind(self):
		return self.kind


	def __str__(self):
		return self._kind


	@tool('singular')
	def get_singular(self, instance):
		return instance

	_irregular_plurals = {
		'child': 'children', 'person': 'people', 'foot': 'feet', 'tooth': 'teeth',
		'goose': 'geese', 'mouse': 'mice', 'man': 'men', 'woman': 'women',
		'ox': 'oxen', 'louse': 'lice', 'die': 'dice',
	}
	@tool('plural')
	@classmethod
	def pluralize(cls, singular: str):
		if ' of ' in singular:
			subject, prep = singular.split(' of ', 1)
			return f'{cls.pluralize(subject)} of {prep}'
		if singular in cls._irregular_plurals:
			return cls._irregular_plurals[singular]
		if singular.endswith('y'):
			return singular[:-1] + 'ies'
		if singular.endswith('s') or singular.endswith('ch') or singular.endswith('sh'):
			return singular + 'es'
		if singular.endswith('f') or singular.endswith('fe'):
			return singular[:-1] + 'ves'
		if singular.endswith('us'):
			return singular[:-2] + 'i'
		if singular.endswith('on') or singular.endswith('um'):
			return singular[:-2] + 'a'
		if singular.endswith('ex') or singular.endswith('ix'):
			return singular[:-2] + 'ices'
		return singular + 's'


	@tool('nat') # top level -> referred to by the entity identifier!
	def get_word(self, singular, plural, quantity=None):
		if quantity is not None and quantity == 1:
			return singular
		return plural


	@tool('formal')
	def get_ident(self):
		return self._ident



class ConstantEntity(Entity):
	def __init__(self, value: Any, ident: str = None, kind: str = None, *, gap: dict[str, str] = None, **kwargs):
		if ident is not None:
			gap = gap or {}
			if 'value' not in gap:
				gap['value'] = f'{ident}_value'
		super().__init__(ident=ident, kind=kind, gap=gap, **kwargs)
		self.value = value


	@tool('value')
	@tool('quantity')
	def get_value(self):
		return self.value


	@tool('formal')
	def get_formal(self):
		return str(self.value)






# class EntityManager(UserDict):
#
#
# 	def __init__(self, items: dict[str, Any] = None, constant_entity: ConstantEntity = None, *,
# 				 entity_path: Path = None, root: Path = None):
# 		if items is None:
# 			if root is None:
# 				root = repo_root()
# 			if entity_path is None:
# 				entity_path = root / 'Arithmetic' / 'arithmetic-entities.yml'
# 			items = self._load_default(entity_path)
# 		if constant_entity is None:
# 			if any(item.kind == 'constant' for item in items):
# 				constant_entity = next(item for item in items if item.kind == 'constant')
# 			else:
# 				constant_entity = self._ConstantEntity
# 		super().__init__()
# 		self.const = constant_entity
# 		self.update(items)
#
# 	_Entity = Entity
# 	_ConstantEntity = ConstantEntity
#
#
# 	def from_kind(self, identifier: str, kind: str) -> Entity:
# 		return self._Entity(identifier, self[kind], kind=kind)
#
#
# 	def from_literal(self, identifier: str, value: Any) -> Entity:
# 		return self.const.make(identifier, value)




























