from .imports import *


from .rules import build_rule



class Concept(ToolKit):
	def _process_patterns(self, data: dict[str, Any]):
		raise NotImplementedError


	_Multi_Template = GadgetDecision
	_Single_Template = Template
	def _process_templates(self, target: str, templates: str | Iterable[str] | Mapping[str, str]):
		if isinstance(templates, str):
			templates = {0: templates}
		elif not isinstance(templates, Mapping) and isinstance(templates, Iterable):
			templates = {i: template for i, template in enumerate(templates)}
		assert templates is not None and len(templates), (f'No templates were specified for {self.__name__} '
														  f'(set {self.__name__}._templates '
														  f'or pass `templates` as an argument).')

		templates = {key: self._Single_Template(template=template, gizmo=target) for key, template in templates.items()}
		if len(templates) == 1:
			template = next(iter(templates.values()))
			gadget = template
		else:
			gadget = self._Multi_Template(templates, choice_gizmo=f'{target}_id')

		return gadget


	def _process_rules(self, rules: dict[str, Any]):
		for target, info in rules.items():
			rule = build_rule(target, info)
			yield rule



class Statement(ToolKit):
	_statement_gizmo = 'statement'
	def indexify(self, index: int):
		self.gauge_apply({self._statement_gizmo: f'{self._statement_gizmo}{index}'})



class Question(Statement):
	_statement_gizmo = 'question'
	def __init__(self, query: str, **kwargs):
		super().__init__(**kwargs)
		self._query = query
		self.gauge_apply({'query': query})


	@property
	def query(self):
		return self._query


	@tool('query_ident')
	def get_query_ident(self):
		return self.query


	@tool('question')
	def statement(self, query: str):
		return f'How many {query} remain?'





