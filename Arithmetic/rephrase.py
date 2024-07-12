import os
from pathlib import Path
from openai import OpenAI
from tqdm import tqdm
import pandas as pd
import omnifig as fig

from .verb import repo_root

@fig.script('rephrase')
def rephrase(cfg: fig.Configuration):

    api_key = cfg.pull('api-key', os.environ.get('OPENAI_API_KEY', None), silent=True)
    if api_key is None:
        raise ValueError(f'No OPENAI API key found, pass argument using "--api-key" or set env var "OPENAI_API_KEY"')

    client = OpenAI(
        base_url=cfg.pull('api-base', None, silent=True),
        api_key=api_key,
    )

    overwrite = cfg.pull('overwrite', False)

    template_path = cfg.pull('template_path', str(repo_root() / 'Arithmetic' / 'rephrase_template.txt'))
    template_path = Path(template_path)
    template = template_path.read_text()

    pbar = cfg.pull('pbar', not cfg.pull('no-pbar', False, silent=True), silent=True)

    max_tokens = cfg.pull('max-tokens', 1000)

    root = cfg.pull('path')
    root = Path(root)
    if root.is_dir():
        paths = list(root.glob('*.csv')) + list(root.glob('*.CSV'))
    else:
        root = None
        paths = [root]

    outdir = cfg.pull('outdir', 'rephrases')
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f'Will save rephrases of {len(paths)} file/s to: {outdir}')

    skip_confirm = cfg.pull('skip-confirm', False)
    while not skip_confirm:
        inp = input('Begin? ([y]/n) ').lower()
        if inp.startswith('n'):
            print('Ending without doing anything.')
        elif not len(inp) or inp.startswith('y'):
            break
        else:
            print('Try again.')

    # path_itr = tqdm(paths) if pbar and len(paths) > 1 else paths
    path_itr = paths
    for path in path_itr:
        if len(paths) == 1:
            print(f'Loading {path}')
        df = pd.read_csv(path)

        outpath = (outdir / path.name) if root is None else (outdir / path.relative_to(root))

        fixed = []

        # item_itr = tqdm(df.iterrows(), total=len(df)) if pbar and len(paths) == 1 else df.iterrows()
        item_itr = tqdm(df.iterrows(), total=len(df))
        for i, item in item_itr:
            assert overwrite or 'rephrase' not in item, 'Item has already been rephrased'

            prompt = template.format(**item)

            response = client.chat.completions.create(
                messages=[{"role": "system", "content": "You are a helpful assistant."},
                          {"role": "user", "content": prompt}],
                model="gpt-3.5-turbo",
                max_tokens=max_tokens,
            )
            rephrased = response.choices[0].message.content

            item['rephrase'] = rephrased.replace('\n', ' ')

            fixed.append(item)

        gold = pd.DataFrame(fixed)
        gold.to_csv(outpath, index=False)

        pass

    pass


if __name__ == '__main__':
    fig.entry('rephrase')

