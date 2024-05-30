from .imports import *

from .common import repo_root



class Entity(ToolKit):
	def __init__(self, ident: str, data: list[dict[str, str]] | dict[str, str] = None, kind: str = None, **kwargs):
		super().__init__(**kwargs)
		if data is not None:
			if isinstance(data, dict):
				if isinstance(next(iter(data.values())), str):
					self.include(DictGadget(data))
				else:
					assert isinstance(next(iter(data.values())), dict), f'Invalid data for entity {kind}: {data}'

					self.include(GadgetDecision(data, choice_gizmo=f'{name}_id'))

		self._kind = kind
		self.data = data


	@property
	def kind(self):
		return self._kind
	@tool('kind')
	def get_kind(self):
		return self.kind


	@tool('main')
	def get_main(self):
		raise NotImplementedError


	def __str__(self):
		return self._kind


	@classmethod
	def make(cls, name: str, data: Any, kind: str = None):
		return cls(name, data, kind=kind)



class ConstantEntity(Entity):
	@classmethod
	def make(cls, name: str, value: int):
		return cls(name, value)


	def __init__(self, name: str, value: int, **kwargs):
		super().__init__(name=name, kind='constant', **kwargs)
		self.value = value


	@tool('value')
	def get_value(self):
		return self.value


	@tool('quantity')
	def get_quantity(self):
		return self.value


	@tool('singular')
	def get_singular(self):
		return str(self.value)


	@tool('plural')
	def get_plural(self, singular: str):
		return singular + 's'


	@tool('main')
	def verbalize(self, singular, plural, quantity=None):
		if quantity is not None and quantity == 1:
			return singular
		return plural


	@tool('formal')
	def get_formal(self):
		return str(self.value)


	@tool('nat')
	def get_nat(self):
		raise NotImplementedError



class EntityManager(UserDict):
	@staticmethod
	def _load_default(entity_path: Path) -> dict[str, Any]:
		assert entity_path.exists(), f'Entity file not found: {entity_path}'
		assert entity_path.suffix in ['.yml', '.yaml'], f'Invalid entity file: {entity_path}'

		kinds = yaml.safe_load(entity_path.read_text())
		return kinds


	def __init__(self, items: dict[str, Any] = None, constant_entity: ConstantEntity = None, *,
				 entity_path: Path = None, root: Path = None):
		if items is None:
			if root is None:
				root = repo_root()
			if entity_path is None:
				entity_path = root / 'Arithmetic' / 'arithmetic-entities.yml'
			items = self._load_default(entity_path)
		if constant_entity is None:
			if any(item.kind == 'constant' for item in items):
				constant_entity = next(item for item in items if item.kind == 'constant')
			else:
				constant_entity = self._ConstantEntity
		super().__init__()
		self.const = constant_entity
		self.update(items)

	_Entity = Entity
	_ConstantEntity = ConstantEntity


	def from_kind(self, identifier: str, kind: str) -> Entity:
		return self._Entity(identifier, self[kind], kind=kind)


	def from_literal(self, identifier: str, value: Any) -> Entity:
		return self.const.make(identifier, value)




























