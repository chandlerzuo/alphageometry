from .imports import *

from pathlib import Path

from .common import repo_root
from .definitions import StatementGenerator


class Verbalization:
    def __init__(self, defs_path: str):
        if defs_path is None:
            defs_path = str(repo_root() / 'assets' / 'demo-patterns.yml')
        defs_path = Path(defs_path)
        assert defs_path.exists(), f'Pattern file not found: {defs_path}'
        # Load the patterns
        self.defs = StatementGenerator.load_patterns(defs_path)
        self.ctx = Controller(StatementGenerator(None))  # uses all definitions

    def fl_2_nl(self, clause_fl: str) -> str:
        self.ctx.clear_cache() # remove all cached values (e.g. from previous samples)

        self.ctx['formal'] = clause_fl  #TODO(felix): This is not correct syntax fix!
        return self.ctx['clause']

    def problem_fl_2_nl(self, fl_statement: str) -> str:
        clauses = fl_statement.split(';')
        all_formal_atomic_clauses = []
        for clause in clauses:
            if clause.find(',') == -1:
                # this clause is atomic
                all_formal_atomic_clauses.append(clause)
            else:
                # this clause contains two or more atomic clauses
                # todo: for now treating , as same thing as ;. Perhaps we can do better
                all_formal_atomic_clauses.extend(clause.split(','))

        nl = ''
        for clause in all_formal_atomic_clauses:
            nl_this_clause = self.fl_2_nl(clause)
            if not nl_this_clause.endswith('.'):
                nl_this_clause += '.'
            nl += self.fl_2_nl(clause)

        return nl