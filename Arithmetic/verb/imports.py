from __future__ import annotations
from typing import Any, Optional, Iterable, Iterator, Mapping, Union, Callable, Self
from pathlib import Path
import random, yaml, inspect
from collections import UserDict

from omnibelt import pformat, Class_Registry
from omniply.core.genetics import GeneticGadget
from omniply.core.gadgets import SingleGadgetBase
from omniply.core.gaggles import MultiGadgetBase
from omniply import AbstractGadget

from omniply.apps.gaps import ToolKit, tool, Gapped, Gauged, GaugedGaggle, GaugedGame, DictGadget, Table, GAUGE

from omniply.apps import (Template as _Template, GadgetDecision as _GadgetDecision,
						  SimpleDecision as _SimpleDecision, Controller as _Controller)
from omniply.apps.decisions.decisions import DecisionBase
from omniply.apps import Combination as _Combination, Permutation as _Permutation



class Controller(_Controller, GaugedGame):
	'''A Controller that is also a GaugedGame.'''
	pass



class _SingleGapped(SingleGadgetBase, Gapped):
	def gauge_apply(self, gauge: GAUGE) -> Self:
		'''Applies the gauge to the Gauged.'''
		super().gauge_apply(gauge)
		if self._gizmo in gauge:
			self._gizmo = gauge[self._gizmo]
		return self



class _DecisionGapped(DecisionBase, Gapped):
	def gauge_apply(self, gauge: GAUGE) -> Self:
		'''Applies the gauge to the Gauged.'''
		super().gauge_apply(gauge)
		if self._choice_gizmo in gauge:
			self._choice_gizmo = gauge[self._choice_gizmo]
		return self



class Template(_Template, _SingleGapped):
	'''A Template that is also Gauged.'''
	def _grab_from(self, ctx: Optional['AbstractGame']) -> Any:
		reqs = {key: ctx.grab(self.gap(key)) for key in self.keys}
		return self.fill_in(reqs)



class SimpleDecision(_SimpleDecision, _DecisionGapped, _SingleGapped):
	'''A SimpleDecision that is also Gauged.'''



class Combination(_Combination, _SingleGapped):
	'''A Combination that is also Gauged.'''



class Permutation(_Permutation, _SingleGapped):
	'''A Permutation that is also Gauged.'''



class GadgetDecision(_GadgetDecision, _DecisionGapped):
	'''A GadgetDecision that is also Gauged.'''
	def gauge_apply(self, gauge: GAUGE) -> Self:
		'''Applies the gauge to the Gauged.'''
		super().gauge_apply(gauge)
		for choice in self._choices.values():
			choice.gauge_apply(gauge)

		options = {gauge.get(gizmo, gizmo): choice for gizmo, choice in self._option_table.items()}
		self._option_table.clear()
		self._option_table.update(options)

		return self

