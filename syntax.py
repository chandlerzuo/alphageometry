from typing import Any, Iterable
import json, yaml


def vlist(ls: Iterable[str]) -> str:
	return '[' + ', '.join(ls) + ']'


class AbstractTransform:
	def geo2me(self, geo: str) -> Any:
		raise NotImplementedError

	def me2geo(self, me: Any) -> str:
		raise NotImplementedError


class ClauseParser(AbstractTransform):
	def geo2me(self, geo: str) -> Any:
		parts = geo.strip().split(' = ')
		pre, fn = parts if len(parts) == 2 else (None, parts[0])
		sym, *args = fn.split()
		terms = {}
		if pre:
			let = pre.split()
			terms['let'] = let
		terms['symbol'] = sym
		terms['args'] = args
		return terms

	def me2geo(self, me: Any) -> str:
		let = ' '.join(me.get('let', []))
		rel = ' '.join(map(str, [me['symbol'], *me['args']]))
		return f'{let} = {rel}' if 'let' in me else rel


class StatementParser(AbstractTransform):
	def __init__(self):
		self.clause = ClauseParser()

	def split(self, geo: str) -> list[str]:
		return geo.split(',')

	def join(self, me: Iterable[str]) -> str:
		return ', '.join(me)

	def geo2me(self, geo: str) -> Any:
		return list(map(self.clause.geo2me, self.split(geo)))

	def me2geo(self, me: Any) -> str:
		return self.join(map(self.clause.me2geo, me))


class Parser(AbstractTransform):
	def __init__(self):
		self.statement = StatementParser()

	def split(self, geo: str) -> list[str]:
		return geo.split(';')

	def join(self, me: Iterable[str]) -> str:
		return '; '.join(me)

	def geo2me(self, geo: str) -> Any:
		return list(map(self.statement.geo2me, self.split(geo)))

	def me2geo(self, me: Any) -> str:
		return self.join(map(self.statement.me2geo, me))


class SyntaxTransform(AbstractTransform):
	def __init__(self):
		self.parser = Parser()

	def geo2me(self, geo: str) -> str:
		raise NotImplementedError

	def me2geo(self, me: str) -> str:
		raise NotImplementedError


class JSON(SyntaxTransform):
	def geo2me(self, geo: str) -> str:
		return json.dumps(self.parser.geo2me(geo))

	def me2geo(self, me: str) -> str:
		return self.parser.me2geo(json.loads(me))


class JSONL(JSON):
	def geo2me(self, geo: str) -> str:
		return '\n'.join(json.dumps(self.parser.statement.geo2me(statement)) for statement in self.parser.split(geo))

	def me2geo(self, me: str) -> str:
		return self.parser.join(self.parser.statement.me2geo(json.loads(statement)) for statement in me.split('\n'))


def test_json():
	tfm = JSON()
	assert all(tfm.me2geo(tfm.geo2me(geo)) == geo for geo in _example_formal_problems)
	geo = 'A B C D = eq_trapezoid A B C D; E = on_circle E D B'
	assert (tfm.geo2me(geo)
			== '[[{"let": ["A", "B", "C", "D"], "symbol": "eq_trapezoid", "args": ["A", "B", "C", "D"]}], '
			   '[{"let": ["E"], "symbol": "on_circle", "args": ["E", "D", "B"]}]]')

	tfm = JSONL()
	assert all(tfm.me2geo(tfm.geo2me(geo)) == geo for geo in _example_formal_problems)
	assert tfm.geo2me(geo) == '''[{"let": ["A", "B", "C", "D"], "symbol": "eq_trapezoid", "args": ["A", "B", "C", "D"]}]
[{"let": ["E"], "symbol": "on_circle", "args": ["E", "D", "B"]}]'''


class YAML(SyntaxTransform):
	def to_yaml(self, data: Any) -> str:
		if 'let' in data:
			return f'{{let: {vlist(data["let"])}, symbol: {data["symbol"]}, args: {vlist(data["args"])}}}'
		return f'{{symbol: {data["symbol"]}, args: {vlist(data["args"])}}}'

	def geo2me(self, geo: str) -> str:
		return vlist(vlist(self.to_yaml(clause) for clause in statement) for statement in self.parser.geo2me(geo))

	def me2geo(self, me: str) -> str:
		return self.parser.me2geo(yaml.load(me, Loader=yaml.SafeLoader))


class YAMLL(YAML):
	def geo2me(self, geo: str) -> str:
		return '\n'.join(vlist(map(self.to_yaml, statement)) for statement in self.parser.geo2me(geo))

	def me2geo(self, me: str) -> str:
		return self.parser.join(self.parser.statement.me2geo(yaml.load(statement, Loader=yaml.SafeLoader))
						 for statement in me.split('\n'))


class YAML_min(YAML):
	def geo2me(self, geo: str) -> str:
		return '\n'.join(' '.join(self.to_yaml(clause) for clause in statement)
						 for statement in self.parser.geo2me(geo))

	def me2geo(self, me: str) -> str:
		return self.parser.me2geo([
			[yaml.load(f'{clause}' if clause.endswith('}') else f'{clause}'+'}', Loader=yaml.SafeLoader)
						 for clause in statement.split('} ')] for statement in me.split('\n')])


def test_yaml():
	tfm = YAML()
	assert all(tfm.me2geo(tfm.geo2me(geo)) == geo for geo in _example_formal_problems)
	geo = 'A B C D = eq_trapezoid A B C D; E = on_circle E D B'
	assert tfm.geo2me(geo) == ('[[{let: [A, B, C, D], symbol: eq_trapezoid, args: [A, B, C, D]}], '
							   '[{let: [E], symbol: on_circle, args: [E, D, B]}]]')

	tfm = YAMLL()
	assert all(tfm.me2geo(tfm.geo2me(geo)) == geo for geo in _example_formal_problems)
	assert tfm.geo2me(geo) == '''[{let: [A, B, C, D], symbol: eq_trapezoid, args: [A, B, C, D]}]
[{let: [E], symbol: on_circle, args: [E, D, B]}]'''

	tfm = YAML_min()
	assert all(tfm.me2geo(tfm.geo2me(geo)) == geo for geo in _example_formal_problems)
	assert tfm.geo2me(geo) == '''{let: [A, B, C, D], symbol: eq_trapezoid, args: [A, B, C, D]}
{let: [E], symbol: on_circle, args: [E, D, B]}'''



class PseudoPython(SyntaxTransform):
	def geo2me(self, geo: str) -> str:
		return '\n'.join('; '.join(
			f'{", ".join(clause["let"])} = {clause["symbol"]}({", ".join(clause["args"])})' if 'let' in clause
			else f'{clause["symbol"]}({", ".join(clause["args"])})'
		for clause in statement) for statement in self.parser.geo2me(geo))

	def _parse_clause(self, clause: str) -> dict[str, Any]:
		parts = clause.split(' = ')
		pre, fn = parts if len(parts) == 2 else (None, parts[0])
		sym, args = fn.split('(')
		args = args[:-1].split(', ')
		terms = {}
		if pre:
			let = pre.split(', ')
			terms['let'] = let
		terms['symbol'] = sym
		terms['args'] = args
		return terms

	def me2geo(self, me: str) -> str:
		return self.parser.me2geo([[self._parse_clause(clause) for clause in statement.split('; ')]
								   for statement in me.split('\n')])


def test_pseudopython():
	tfm = PseudoPython()
	assert all(tfm.me2geo(tfm.geo2me(geo)) == geo for geo in _example_formal_problems)
	geo = 'A B C D = eq_trapezoid A B C D; E = on_circle E D B'
	assert tfm.geo2me(geo) == '''A, B, C, D = eq_trapezoid(A, B, C, D)
E = on_circle(E, D, B)'''




_example_formal_problems = ['A B C D = eq_trapezoid A B C D; E = on_circle E D B', 'A B C D = r_trapezoid A B C D', 'A B C D = eq_quadrangle A B C D; E = on_circle E D A; F = eqdistance F E D C', 'A B C D = r_trapezoid A B C D; E = angle_mirror E A C D; F G H I = excenter2 F G H I D A B; J = excenter J A I B; K = intersection_pp K B H D F A G', 'A B C D = trapezoid A B C D; E F G H I = pentagon E F G H I; J = reflect J G E B; K = foot K F A E; L M N O = ninepoints L M N O D H F; P = on_tline P K F J', 'A B C D = isquare A B C D; E = midpoint E A C', 'A B C = risos A B C; D = angle_mirror D A B C', 'A B C = triangle12 A B C; D = lc_tangent D B C; E F G = ieq_triangle E F G', 'A B C D = r_trapezoid A B C D; E = on_line E D C', 'A B C D = eq_trapezoid A B C D; E = on_pline E A C B', 'A B C = triangle A B C; D = circumcenter D B C A; E = foot E C A D; F = on_bline F A B; G = lc_tangent G F A', 'A B C D = quadrangle A B C D; E = foot E A C B; F = eqdistance F E B A; G = on_dia G D A, eqangle3 G F A B D C', 'A B C D = isquare A B C D; E F G = triangle E F G; H = on_line H C D; I = on_aline I H A B C G', 'A B C D = eqdia_quadrangle A B C D; E F G = triangle E F G; H = orthocenter H C E B', 'A B C D = isquare A B C D; E F = trisegment E F B C; G H I = triangle G H I; J = midpoint J H I', 'A B C = iso_triangle A B C; D = lc_tangent D C A; E = foot E B D A; F = intersection_lt F B C D A E; G = reflect G D F A', 'A B C = r_triangle A B C; D E F G = incenter2 D E F G C A B; H I J K = quadrangle H I J K; L = lc_tangent L D I', 'A B C D = quadrangle A B C D; E = on_opline E B A; F = on_opline F B E', 'A B C D = rectangle A B C D; E = intersection_ll E C A B D; F = on_opline F A E; G = lc_tangent G A C', 'A B C = r_triangle A B C; D = shift D B A C; E F G H = r_trapezoid E F G H; I = eqdistance I C H G, on_bline I E F', 'A B C = triangle12 A B C; D = on_pline D C A B', 'A B C = risos A B C; D = on_pline D A B C', 'A B C D = eqdia_quadrangle A B C D', 'A B C = iso_triangle A B C; D = shift D C A B; E = reflect E B C D; F G H I = quadrangle F G H I', 'A B C D = isquare A B C D; E F G H I = pentagon E F G H I; J K L = ieq_triangle J K L', 'A B C D = quadrangle A B C D; E = on_bline E C D; F = on_pline F C A D, angle_bisector F D B E', 'A B C D = r_trapezoid A B C D; E F = segment E F; G = eqangle3 G C A B D E; H = angle_bisector H A D G', 'A B C D = r_trapezoid A B C D; E = circumcenter E D C B; F G H I = eq_trapezoid F G H I; J = on_bline J C G, lc_tangent J H F; K = eqdistance K C J A', 'A B C D = eq_trapezoid A B C D; E F G H = quadrangle E F G H; I J K L = incenter2 I J K L A B F; M N = square A G M N; O P = square G B O P', 'A B C = triangle12 A B C; D = eqdistance D C A B', 'A B = segment A B; C D E F G = pentagon C D E F G; H I J K = trapezoid H I J K', 'A B C D = trapezoid A B C D; E = lc_tangent E D C; F G H I = isquare F G H I; J = on_circle J A G; K = on_bline K H B, on_pline K E I H', 'A B C D E = pentagon A B C D E; F = on_line F A E; G = mirror G F C; H I = segment H I', 'A B C D = isquare A B C D; E = on_tline E A B C; F G H I = excenter2 F G H I D B E', 'A B C D = eq_trapezoid A B C D; E = lc_tangent E C A, eqdistance E B A C; F G H I = centroid F G H I A B E', 'A B C = r_triangle A B C; D = on_opline D C A; E F = square A B E F', 'A B C D = eq_quadrangle A B C D; E = on_opline E C D; F G H = iso_triangle F G H; I = eqdistance I D F G; J K L = triangle12 J K L; M = on_tline M K F H; N = intersection_tt N C F I L H G; O = eqangle3 O M B C G H, angle_mirror O L I D', 'A B C = triangle A B C; D = on_pline D B A C', 'A B C = triangle A B C; D = on_bline D C A; E F = tangent E F A D B; G = angle_mirror G B F D', 'A B C = triangle A B C; D = on_tline D C B A', 'A B C D = trapezoid A B C D; E = on_opline E B A', 'A B C D = rectangle A B C D; E = reflect E C B D; F G H I = ninepoints F G H I A D C', 'A B C = r_triangle A B C; D = on_circle D B C; E = on_opline E D C; F G H I = incenter2 F G H I C A D', 'A B C = iso_triangle A B C; D = on_line D C A; E = on_dia E D B', 'A B C D = r_trapezoid A B C D; E = on_line E A B; F = intersection_ll F A C D B; G H I = triangle G H I', 'A B C D = eq_trapezoid A B C D', 'A B C D = trapezoid A B C D; E = eqdistance E D C A, on_circle E B C', 'A B C D = eqdia_quadrangle A B C D; E = angle_mirror E C A D; F G H I = cc_tangent F G H I B A E D; J K L M = r_trapezoid J K L M', 'A B C D = eq_quadrangle A B C D; E = on_dia E C B', 'A B = segment A B; C = midpoint C B A', 'A = free A; B C D = iso_triangle B C D; E = reflect E D B A; F = on_tline F A C D; G H I = triangle G H I; J = psquare J G A', 'A B C D = eq_trapezoid A B C D; E = incenter E C A B; F G H I = centroid F G H I E B D; J = on_tline J C D I; K = intersection_cc K J E D; L = foot L C G K; M = on_bline M L E', 'A B C D = eqdia_quadrangle A B C D; E F = segment E F; G = on_pline G B D C', 'A B C = iso_triangle A B C; D = midpoint D B A; E = incenter E B C A', 'A B C D = r_trapezoid A B C D; E = parallelogram A C D E; F = free F; G = nsquare G B D; H = on_aline H G A D C F', 'A B C D = eq_quadrangle A B C D; E = intersection_cc E D A B; F = on_opline F C B', 'A B C D = eq_trapezoid A B C D; E = on_line E D A; F G = tangent F G B D C', 'A B C = ieq_triangle A B C; D = on_circle D C A; E F G H = excenter2 E F G H C A D; I = angle_bisector I B C G; J K L = 3peq J K L D I C', 'A B = segment A B; C D E F G = pentagon C D E F G; H = foot H F D B; I = lc_tangent I G F, on_dia I A E; J = nsquare J E F; K = eqangle3 K F H I J G; L = on_bline L F E', 'A B C = r_triangle A B C; D = intersection_lc D A B C; E = incenter E B C A; F = angle_bisector F B C E; G H I J = eq_quadrangle G H I J; K = intersection_ll K H C B F; L = excenter L H E J; M = on_tline M H B A', 'A B C = triangle12 A B C; D E F G = centroid D E F G A C B; H = incenter H F D C; I J = square E B I J; K = free K; L = intersection_lc L J G F', 'A B C D = isquare A B C D; E F G H = incenter2 E F G H C B D; I J K L = eq_quadrangle I J K L; M N O = triangle M N O', 'A B C D = eq_trapezoid A B C D; E = on_circle E A D; F G H I = isquare F G H I; J = angle_bisector J I H E', 'A B C D = eqdia_quadrangle A B C D; E = intersection_ll E A B D C; F = on_bline F E D; G H I J = quadrangle G H I J', 'A B C D = eq_quadrangle A B C D; E = circle E D C A; F = on_pline F A E B; G H I J = rectangle G H I J; K = on_dia K I H; L = on_dia L J C', 'A B C D = eqdia_quadrangle A B C D; E F G = r_triangle E F G; H = on_line H E G; I J = trisegment I J C E; K = on_circle K B E; L = reflect L C H F; M = angle_mirror M K E J', 'A B C = r_triangle A B C; D = mirror D B A; E = incenter E D C B; F = eqdistance F D C B, on_bline F C E', 'A B C D = isquare A B C D; E = on_circum E D C B', 'A B C = iso_triangle A B C; D = psquare D C B', 'A B C = risos A B C', 'A B C = triangle A B C; D = intersection_lc D A C B; E = psquare E C B; F G H I = trapezoid F G H I; J = on_opline J F E; K L = tangent K L E H F; M = on_circle M L H, s_angle E G M 90', 'A B C = risos A B C; D = incenter D A B C; E F G H = centroid E F G H C D A', 'A B C D = eqdia_quadrangle A B C D; E = shift E C B D; F = circumcenter F D A B; G H I J = r_trapezoid G H I J; K = parallelogram J C H K', 'A B C = ieq_triangle A B C; D E F G = quadrangle D E F G; H I J K = incenter2 H I J K G B D', 'A B C = triangle12 A B C', 'A B C D = eq_trapezoid A B C D; E F G = r_triangle E F G; H = mirror H E C; I = on_circle I B G', 'A = free A; B C D = iso_triangle B C D; E F G = risos E F G; H = on_opline H C F', 'A B C D E = pentagon A B C D E; F G H I = quadrangle F G H I; J = eqdistance J G F H; K = eqangle3 K I D A H G; L = excenter L I A H; M = on_aline M E I L J D', 'A B C D = rectangle A B C D', 'A B C = triangle12 A B C; D = on_circum D C B A; E F G H = r_trapezoid E F G H; I = on_aline I H E F A D, lc_tangent I F A', 'A B C = triangle A B C; D E F = risos D E F; G = on_aline G D A C B F; H = on_circle H B E', 'A B C D = isquare A B C D; E F G = triangle12 E F G; H I J K = isquare H I J K']

