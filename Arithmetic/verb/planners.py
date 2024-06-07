from .imports import *

from .relations import Relation
from .concepts import Question



class ProblemPlanner(ToolKit, fig.Configurable):
	def connect(self, ctxs: Iterable[Controller]) -> Controller:
		raise NotImplementedError


	@tool.from_context('answer')
	def solve(self, ctx: Controller) -> Any:
		if ctx.grab('question', None):
			return ctx[f'{ctx["query_ident"]}_value']


	@tool.from_context('nl')
	def verbalize(self, ctx: Controller) -> str:
		raise NotImplementedError



@fig.component('independent')
class IndependentStatements(ProblemPlanner):
	def connect(self, ctxs: Iterable[Controller]) -> Controller:
		prob_ctx = Controller()
		q = None
		for i, ctx in enumerate(ctxs):
			for vendor in ctx.vendors():
				if isinstance(vendor, Relation):
					vendor.indexify(i, )  # ctxs[:i], ctxs[i + 1:])
				if isinstance(vendor, Question):
					assert q is None, f'Cant have more than 1 question'
					q = vendor
				prob_ctx.include(vendor)

		prob_ctx.include(DictGadget({'num_statements': len(ctxs) - int(q is not None)}))
		prob_ctx.include(self)
		return prob_ctx


	@tool.from_context('nl')
	def verbalize(self, ctx: Controller) -> str:
		lines = []
		for i in range(ctx['num_statements']):
			lines.append(ctx[f'statement{i}'])

		q = ctx.grab('question', None)
		if q is not None:
			lines.append(q)

		return ' '.join(lines)




