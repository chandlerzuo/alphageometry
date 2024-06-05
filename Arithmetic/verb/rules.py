
from omniply.apps.decisions.abstract import CHOICE
from .imports import *

from .common import Enumeration


def build_rule(target: str, data: dict[str, Any]) -> 'Rule':
	rule_type = data['type']

	rule_cls = _rule_registry.get_class(rule_type)

	return rule_cls.from_data(target, data)

_rule_registry = Class_Registry()


class Rule(ToolKit):
	@classmethod
	def from_data(cls, target: str, data: dict[str, Any]):
		raise NotImplementedError


	def __init_subclass__(cls, name: str = None, **kwargs):
		super().__init_subclass__(**kwargs)
		if name is not None:
			_rule_registry.new(name, cls)


	def __init__(self, target: str, **kwargs):
		super().__init__(**kwargs)
		self._target = target


	@property
	def target(self):
		return self._target



class Conjunction(Rule, name='conjunction'):
	_term_options = [' and ', ' & ', None] # TODO: maybe split up to properly manage the oxford comma


	@classmethod
	def from_data(cls, target: str, data: dict[str, Any]):
		elements = data['elements']
		ordered = data.get('ordered', False)
		return cls(target, elements, ordered=ordered)


	def __init__(self, target: str, elements: list[str], ordered=False, **kwargs):
		super().__init__(target, elements, **kwargs)
		self.include(SimpleDecision(f'{target}_term', self._term_options))
		self.include(Enumeration(elements, gizmo=target, oxford=True, ordered=ordered,
								 aggregator_gizmo=f'{target}_term', choice_gizmo=f'{target}_order'))



class OrderedConjunction(Conjunction, name='ord-conjunction'):
	@classmethod
	def from_data(cls, target: str, data: dict[str, Any]):
		elements = data['elements']
		return cls(target, elements, ordered=True)



class Equality(Rule, name='equality'):
	@classmethod
	def from_data(cls, target: str, data: dict[str, Any]):
		elements = data['elements']
		return cls(target, elements)


	def __init__(self, target: str, elements: list[str], **kwargs):
		super().__init__(target, **kwargs)

		options = []

		eqsign = Enumeration(elements, delimiter=' = ', gizmo=target, choice_gizmo=f'{target}_order')
		options.append(eqsign)

		if len(elements) > 2:
			terms = SimpleDecision(f'{target}_term', [' all equal ', ' are all the same as ', ' are all congruent to '])
			order = Enumeration(elements, gizmo=target, aggregator_gizmo=f'{target}_term', choice_gizmo=f'{target}_order')
			options.append(ToolKit().include(terms, order))
		elif len(elements) == 2:
			terms = SimpleDecision(f'{target}_term', [' equals ', ' is equal to ', ' is the same as ', ' is congruent to '])
			order = Enumeration(elements, gizmo=target, aggregator_gizmo=f'{target}_term', choice_gizmo=f'{target}_order')
			options.append(ToolKit().include(terms, order))

		self.include(GadgetDecision(options, choice_gizmo=f'{target}_fmt') if len(options) > 1 else options[0])




# class Disjunction(Rule, Enumeration, kind='disjunction'):
# 	_aggregator = ' or '
#
# 	def __init__(self, name: str, elements: list[str], **kwargs):
# 		super().__init__(name, elements, oxford=True, element_gizmos=elements, gizmo=name,
# 						 choice_gizmo=f'{name}_order', **kwargs)



class LiteralRule(Rule, name='literal'):
	@classmethod
	def from_data(cls, target: str, data: dict[str, Any]):
		options = data['args']
		return cls(target, options)


	def __init__(self, target: str, options: list[str], **kwargs):
		super().__init__(target, **kwargs)
		self.include(SimpleDecision(target, options, **kwargs))



# class ReferenceRule(LiteralRule, kind='ref'):
# 	def __init__(self, name: str, options: list[str], **kwargs):
# 		super().__init__(name, options, **kwargs)
#
#
# 	# def _as_abstract(self, ctx: 'AbstractGame'):
# 	# 	return super()._commit(ctx, ctx[self.choice_gizmo], self.name)
#
#
# 	def _commit(self, ctx: 'AbstractGame', choice: CHOICE, gizmo: str) -> Any:
# 		return ctx.grab(super()._commit(ctx, choice, gizmo))


# class SubTemplates(Rule, GadgetDecision, kind='sub'):
# 	_Template = Template
# 	def __init__(self, name: str, templates: list[str], **kwargs):
# 		gadgets = [self._Template(template) for template in templates]
# 		super().__init__(name, templates, choices=gadgets, choice_gizmo=f'{name}_choice', **kwargs)
#
#
# 	def _as_abstract(self, ctx: 'AbstractGame'):
# 		return ctx[self.choice_gizmo]



