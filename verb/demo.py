from .imports import *

import json, yaml
from tqdm import tqdm
from pathlib import Path
import pandas as pd
import omnifig as fig
from tabulate import tabulate
from collections import Counter

from .common import repo_root
from .definitions import StatementGenerator


@fig.script('generate')
def generate(cfg: fig.Configuration):
	'''
	Generate samples of formal and natural language statements from a set of definitions.

	Parameters:
	- path (str, default: 'assets/demo-patterns.yml') - The path to the yaml file containing the definition patterns.
	- out (str, default: 'demo.csv') - The path to save the generated samples.
	- N, n (int, default: 100) - The number of samples to generate.
	- definitions (str | list[str], default: None) - The names of the definitions to use. If None, all available definitions are used.
	- random-order (bool, default: True) - Whether to randomize the order of the arguments in the generated samples.
	- random-selection (bool, default: True) - Whether to randomize the selection of arguments in the generated samples.
	- formal (bool, default: True) - Whether to store the formal language statement in the generated samples.
	- certificates (bool, default: False) - Whether to store the certificates in the generated samples.
	- seed (int, default: None) - The random seed to use for generating the samples.
	- pbar (bool, default: False) - Whether to show a progress bar while generating the samples.
	- overwrite (bool, default: False) - Whether to overwrite the output file if it already exists.

	'''
	pbar = cfg.pull('pbar', not cfg.pull('no-pbar', False, silent=True), silent=True)

	path = cfg.pull('path', str(repo_root() / 'assets' / 'demo-patterns.yml'))
	path = Path(path)
	assert path.exists(), f'Pattern file not found: {path}'

	outpath = cfg.pull('out', 'demo.csv')
	outpath = Path(outpath)
	if outpath.exists() and not cfg.pull('overwrite', False):
		raise FileExistsError(f'Output file already exists (overwrite with `--overwrite`): {outpath}')

	definitions = cfg.pull('definitions', None) # defaults to using all available

	rand_order = cfg.pull('random-order', True)
	rand_sel = cfg.pull('random-selection', True)

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
	defs = StatementGenerator.load_patterns(path)
	print(f'Loaded {len(defs)} definitions from: {path}')

	ctx = Controller(StatementGenerator(definitions))
	if not rand_order or not rand_sel:
		fixed = {}
		if not rand_order:
			fixed['order_id'] = 0
		if not rand_sel:
			fixed['selection_id'] = 0
		ctx.include(DictGadget(fixed))


	# Generate the samples
	itr = range(N_samples)
	if pbar:
		itr = tqdm(itr)

	count = Counter({d.name: 0 for d in defs})
	samples = []
	for _ in itr:
		ctx.clear_cache() # remove all cached values (e.g. from previous samples)

		sample = {}

		if store_formal:
			sample['formal'] = ctx['formal']

		sample['natural'] = ctx['clause']

		if store_certificates:
			sample['certificate'] = json.dumps(ctx.certificate())

		d = ctx['definition']
		count[d] += 1

		samples.append(sample)

	print(tabulate(count.most_common(), headers=['Definition', 'Count']))

	# Save the samples as csv using pandas
	df = pd.DataFrame(samples)
	df.to_csv(outpath, index=False)

	print(f'Saved {len(samples)} samples to: {outpath}')

	return samples




