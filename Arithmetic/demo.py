
import random
import json, yaml
from tqdm import tqdm
from pathlib import Path
import pandas as pd
import omnifig as fig
from tabulate import tabulate
from collections import Counter

from .verb import repo_root, Verbalization
from .symbolic_arithmetic_problem_generator import SymArithmeticProbGen


@fig.script('generate-toy')
def generate(cfg: fig.Configuration):
	'''
	Generate samples of formal and natural language statements from a set of definitions.

	Parameters:
	- relations-path (str, default: 'Arithmetic/arithmetic-def-patterns.yml') - The path to the yaml file containing the relation patterns.
	- entity-path (str, default: 'Arithmetic/arithmetic-entities.yml') - The path to the yaml file containing the entity patterns.
	- out (str, default: 'demo.csv') - The path to save the generated samples.
	- N, n (int, default: 100) - The number of samples to generate.
	- formal (bool, default: True) - Whether to store the formal language statement in the generated samples.
	- certificates (bool, default: False) - Whether to store the certificates in the generated samples.
	- seed (int, default: None) - The random seed to use for generating the samples.
	- pbar (bool, default: False) - Whether to show a progress bar while generating the samples.
	- overwrite (bool, default: False) - Whether to overwrite the output file if it already exists.

	'''
	pbar = cfg.pull('pbar', not cfg.pull('no-pbar', False, silent=True), silent=True)

	relations_path = cfg.pull('relations-path', str(repo_root() / 'Arithmetic' / 'arithmetic-def-patterns.yml'))
	relations_path = Path(relations_path)
	assert relations_path.exists(), f'Relations pattern file not found: {relations_path}'

	entity_path = cfg.pull('entity-path', str(repo_root() / 'Arithmetic' / 'arithmetic-entities.yml'))
	entity_path = Path(entity_path)
	assert entity_path.exists(), f'Entity pattern file not found: {entity_path}'

	cfg.push('planner._type', 'independent', overwrite=False, silent=True)
	planner = cfg.pull('planner')

	generator = cfg.pull('generator', None)
	if generator is None:
		generator = SymArithmeticProbGen()

	outpath = cfg.pull('out', 'demo.csv')
	outpath = Path(outpath)
	if outpath.exists() and not cfg.pull('overwrite', False):
		raise FileExistsError(f'Output file already exists (overwrite with `--overwrite`): {outpath}')

	N_samples = cfg.pulls('N', 'n', default=100)

	store_formal = cfg.pull('formal', True)
	store_certificates = cfg.pulls('certificates', 'cert', default=False)
	if store_certificates: # TODO: fix
		print(f'WARNING: Storing certificates is currently not supported.')
		store_certificates = False

	seed = cfg.pull('seed', None)
	if seed is not None:
		random.seed(seed)

	print(f'Generating {N_samples} samples and saving to: {outpath}')

	# Load the patterns
	verb = Verbalization(planner, relation_path=relations_path, entity_path=entity_path)

	# Generate the samples
	itr = range(N_samples)
	if pbar:
		itr = tqdm(itr)

	# count = Counter({d.name: 0 for d in defs})
	samples = []
	for _ in itr:
		sample = {}

		generator.generate_expression()
		formal = generator.decompose_expression()
		if store_formal:
			sample['formal'] = formal

		ctx = verb.parse_problem(formal)
		sample['natural'] = ctx['nl']

		if store_certificates:
			sample['certificate'] = json.dumps(ctx.certificate())

		sample['answer'] = ctx['answer']

		# d = ctx['definition']
		# count[d] += 1

		samples.append(sample)

	# print(tabulate(count.most_common(), headers=['Definition', 'Count']))

	# Save the samples as csv using pandas
	df = pd.DataFrame(samples)
	df.to_csv(outpath, index=False)

	print(f'Saved {len(samples)} samples to: {outpath}')

	return samples


if __name__ == '__main__':
	fig.entry('generate')


