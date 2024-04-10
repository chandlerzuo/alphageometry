from pathlib import Path
from .imports import *


from omniply.core.genetics import GeneticGadget
from omniply.apps import Template, GadgetDecision, SimpleDecision, Controller, Combination, Permutation


def repo_root():
	return Path(__file__).parent.parent


class Selection(Combination): # always returns the selection in order
	def __init__(self, pool: Iterable[str], num: int, **kwargs):
		pool = list(pool)
		assert num <= len(pool), f'Cannot select {num} from a pool of {len(pool)}'
		super().__init__(N=len(pool), K=num, **kwargs)
		self._pool = pool


	def _commit(self, ctx, choice: int, gizmo: str) -> tuple[Any, ...]:
		return tuple(self._pool[i] for i in super()._commit(ctx, choice, gizmo))



class Unwrapper(GeneticGadget):
	def __init__(self, root: str, num: int, *, source: str = None, **kwargs):
		if source is None:
			source = f'{root}s'
		super().__init__(**kwargs)
		self._root = root
		self._num = num
		self._source = source


	def grab_from(self, ctx, gizmo: str) -> Any:
		idx = int(gizmo[len(self._root):])
		return ctx[self._source][idx]


	def gizmos(self) -> Iterator[str]:
		yield from (f'{self._root}{i}' for i in range(self._num))


	def _genetic_information(self, gizmo: str):
		return {**super()._genetic_information(gizmo), 'parents': (self._source,)}



class ArgumentGenerator(ToolKit):
	_argument_name_pool = 'abcdefghijklmnopqrstuvwxyz'.upper()

	def _populate_selector(self, num_args: int, pool: Iterable[str] = None, shuffle: bool = True):
		if pool is None:
			pool = self._argument_name_pool

		tools = []

		selection = Selection(pool, num_args, gizmo='selection', choice_gizmo='selection_id')
		tools.append(selection)

		unwrapper = Unwrapper('arg', num_args, source='arguments')
		tools.append(unwrapper)

		if shuffle:
			order = Permutation(N=num_args, gizmo='order', choice_gizmo='order_id')
			tools.append(order)

		return tools


	def __init__(self, N: int, *, shuffle: bool = True, pool: Iterable[str] = None, **kwargs):
		super().__init__(**kwargs)
		self.extend(self._populate_selector(N, pool=pool, shuffle=shuffle))


	@tool('arguments')
	def _get_arguments(self, selection, order=None):
		if order is None:
			return selection
		return tuple(selection[i] for i in order)



class Enumeration(Permutation): # GeneticGadget
	_aggregator: Optional[str] = None
	_delimiter: Optional[str] = None
	_prefix: Optional[str] = None
	_suffix: Optional[str] = None

	def __init__(self, element_gizmos: Iterable[str], oxford: bool = False,
				 aggregator_gizmo: str = None, aggregator: str = None,
				 prefix_gizmo: str = None, prefix: str = None,
				 suffix_gizmo: str = None, suffix: str = None,
				 delimiter_gizmo: str = None, delimiter: str = ', ',
				 **kwargs):
		element_gizmos = tuple(element_gizmos)
		# assert len(element_gizmos) > 1, f'Cannot enumerate {len(element_gizmos)} elements: {element_gizmos}'
		super().__init__(N=len(element_gizmos), **kwargs)
		if aggregator is not None:
			self._aggregator = aggregator
		self._aggregator_gizmo = aggregator_gizmo
		if delimiter is not None:
			self._delimiter = delimiter
		self._delimiter_gizmo = delimiter_gizmo
		if suffix is not None:
			self._suffix = suffix
		self._suffix_gizmo = suffix_gizmo
		if prefix is not None:
			self._prefix = prefix
		self._prefix_gizmo = prefix_gizmo
		self._element_gizmos = element_gizmos
		self._oxford = oxford


	def _genetic_information(self, gizmo: str):
		parents = list(self._element_gizmos)
		if self._delimiter_gizmo is not None:
			parents.append(self._delimiter_gizmo)
		if self._aggregator_gizmo is not None:
			parents.append(self._aggregator_gizmo)
		if self._prefix_gizmo is not None:
			parents.append(self._prefix_gizmo)
		if self._suffix_gizmo is not None:
			parents.append(self._suffix_gizmo)
		return {**super()._genetic_information(gizmo), 'parents': tuple(parents)}


	def _commit(self, ctx, choice: int, gizmo: str) -> str:
		elements = [ctx[self._element_gizmos[i]] for i in super()._commit(ctx, choice, gizmo)]

		delimiter = self._delimiter if self._delimiter_gizmo is None else ctx[self._delimiter_gizmo]
		assert delimiter is not None, f'Delimiter {delimiter} not specified (eg. set attribute `_delimiter`)'
		aggregator = self._aggregator if self._aggregator_gizmo is None else ctx[self._aggregator_gizmo]

		txt = delimiter.join(elements) if aggregator is None \
			else (f'{delimiter.join(elements[:-1])}{delimiter.strip() if self._oxford and len(elements) > 2 else ""}'
				  f'{aggregator}{elements[-1]}')

		if self._prefix_gizmo is not None:
			txt = f'{ctx[self._prefix_gizmo]}{txt}'
		elif self._prefix is not None:
			txt = f'{self._prefix}{txt}'
		if self._suffix_gizmo is not None:
			txt = f'{txt}{ctx[self._suffix_gizmo]}'
		elif self._suffix is not None:
			txt = f'{txt}{self._suffix}'

		return txt




