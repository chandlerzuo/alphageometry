from .imports import *

from pathlib import Path

from .common import repo_root
from .definitions import StatementVerbalization, load_patterns


class AbstractVerbalization:
    def generate_fl_problem(self, seed: int = None) -> str:
        raise NotImplementedError('coming soon')


    def problem_fl_2_nl(self, fl_problem: str, *, seed: int = None) -> str:
        raise NotImplementedError


    def problem_nl_2_fl(self, nl_problem: str) -> str:
        raise NotImplementedError(f'this is too hard - is it even possible?')



class IndependentStatementVerbalization(AbstractVerbalization):
    '''
    verbalizes each statement in the problem independently, which is reasonable, but not ideal as it
    significantly simplifies translation, and limits the expressiveness of the NL
    '''
    def __init__(self, defs_path: str):
        if defs_path is None:
            defs_path = str(repo_root() / 'assets' / 'def-patterns.yml')
        defs_path = Path(defs_path)
        assert defs_path.exists(), f'Pattern file not found: {defs_path}'
        # Load the patterns
        self.defs = load_patterns(defs_path)


    def parse_fl(self, fl_statement: str) -> Controller:
        return Controller(StatementVerbalization(), DictGadget({'formal': fl_statement}))


    def fl_2_nl(self, clause_fl: str) -> str:
        ctx = self.parse_fl(clause_fl)
        return ctx['statement']


    def problem_fl_2_nl(self, fl_problem: str, *, seed: int = None) -> str:
        statements = fl_problem.split(';')
        verbs = []
        for fl_statement in statements:
            nl_statement = self.fl_2_nl(fl_statement)
            verbs.append(nl_statement)
        return ' '.join(verbs)


# class Verbalization:
#     def __init__(self, defs_path: str):
#         if defs_path is None:
#             defs_path = str(repo_root() / 'assets' / 'demo-patterns.yml')
#         defs_path = Path(defs_path)
#         assert defs_path.exists(), f'Pattern file not found: {defs_path}'
#         # Load the patterns
#         self.defs = ClauseGenerator.load_patterns(defs_path)
#         self.ctx = Controller(StatementVerbalization(None))  # uses all definitions
#
#     def parse_fl(self, fl_statement: str) -> str:
#         '''assumes `fl_statement` is a single statement, but may have multiple clauses'''
#         assert ';' not in fl_statement
#
#         full = fl_statement.strip()
#
#         clauses = full.split(',')
#
#         name_and_args = parts[1].split()
#
#         # Extract geometric element name and its arguments
#         element_name = name_and_args[0]
#         arguments = name_and_args[1:]
#
#         self.ctx.update({'definition': element_name, })
#
#         pass
#
#     def fl_2_nl(self, clause_fl: str) -> str:
#         self.ctx.clear_cache() # remove all cached values (e.g. from previous samples)
#
#         self.ctx['formal'] = clause_fl
#         return self.ctx['statement']
#
#     def problem_fl_2_nl(self, fl_problem: str) -> str:
#         clauses = fl_problem.split(';')
#         all_formal_atomic_clauses = []
#         for clause in clauses:
#             if clause.find(',') == -1:
#                 # this clause is atomic
#                 all_formal_atomic_clauses.append(clause)
#             else:
#                 # this clause contains two or more atomic clauses
#                 # todo: for now treating , as same thing as ;. Perhaps we can do better
#                 all_formal_atomic_clauses.extend(clause.split(','))
#
#         nl = ''
#         for clause in all_formal_atomic_clauses:
#             nl_this_clause = self.fl_2_nl(clause)
#             if not nl_this_clause.endswith('.'):
#                 nl_this_clause += '.'
#             nl += self.fl_2_nl(clause)
#
#         return nl