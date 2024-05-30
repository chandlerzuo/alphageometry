
import ast

def unnest_expression(expression):
	tree = ast.parse(expression, mode='eval')

	# This list will hold all our operations
	operations = []
	# This counter will help us create unique variable names
	var_counter = [0]

	def handle_node(node):
		if isinstance(node, ast.Num):  # For numbers
			var_name = f'v{var_counter[0]}'
			var_counter[0] += 1
			operations.append((var_name, None, node.n))
			return var_name
		elif isinstance(node, ast.Name):  # For variables
			var_name = f'v{var_counter[0]}'
			var_counter[0] += 1
			operations.append((var_name, None, node.id))
			return var_name
		elif isinstance(node, ast.Call):
			args = [handle_node(arg) for arg in node.args]
			func_name = node.func.id
			var_name = f'v{var_counter[0]}'
			var_counter[0] += 1
			operations.append((var_name, func_name, args))
			return var_name
		else:
			raise ValueError("Unsupported expression type")

	handle_node(tree.body)

	operations[-1] = (None, *operations[-1][1:])

	return operations

def test_unnest():
	result = unnest_expression("f(g(1), h(x))")
	expected = [
		('v0', None, 1), ('v1', 'g', ['v0']), ('v2', None, 'x'), ('v3', 'h', ['v2']), (None, 'f', ['v1', 'v3'])
	]
	assert result == expected

	result = unnest_expression('f(g(h(x), y))')
	


