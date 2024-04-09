from __future__ import annotations
from typing import Any, Optional, Iterable, Iterator, Mapping, Union

import random
from omnibelt import pformat
from omniply import Context, ToolKit, tool
from omniply.core.genetics import GeneticGadget
from omniply.core.gadgets import SingleGadgetBase
from omniply import AbstractGadget, ToolKit, tool
from omniply.apps import Template as _Template, GadgetDecision, DictGadget, SimpleDecision, Controller



class Template(_Template):
	_use_name: bool = False
	@classmethod
	def toggle_use_name(cls):
		cls._use_name = not cls._use_name
		return cls._use_name


	def fill_in(self, reqs: dict[str, str] = None, **vals: str):
		if self._use_name:
			return f'[{self._gizmo}]'
		return super().fill_in(reqs, **vals)



