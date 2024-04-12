

# Verbalization


## Installation

Make sure to install all the dependencies mentioned in `requirements.in` (and especially that the bottom ones starting with `omni` are all installed with the correct versions).

## Running

Then you should be able to run the demo script with:

```bash
python -m verb
```

Which will generate a sample file `demo.csv`.

## Usage

You can modify the generation behavior by passing in parameters described in the help message. To print the help message run:

```bash
python -m verb -h
```

So for example, to generate a file with 500 samples, save it to `my_file.csv`, using seed 1, and the definitions in `assets/def-patterns.yml`, you can run:

```bash
python -m verb --n 500 --out my_file.csv --seed 1 --path assets/def-patterns.yml
```


## Development

### Definitions

When creating templates for definitions, I suggest you directly modify the `assets/def-patterns.yml` file. For each definition you need to specify the number of args `num_args` and at least one template in `templates`. Optionally you can also define any number of intermediate terms each of which will be interpretted as a specific "rule." The available rules are:

- `literal` - choosing one of the strings based on the provided ones (used for interchangable phrases/words)
- `ref` - choosing one of the terms from the provided list (used for interchangable terms/concepts)
- `conjunction` - listing the terms as a conjunction with a random order
- `ord-conjunction` - listing the terms as a conjunction with a specific order
- `equality` - listing the terms as an equality with a random order
- Basics: `point`, `line`, `angle`, `triagnle`, etc.

For a full list of available rules, see `verb/rules.py`.


### Code

The code that is run in the demo `python -m verb` is found in `verb/demo.py`. The code to load and prepare definitions is in `verb/definitions.py`, and all the rules are in `verb/rules.py`. The variable selection and other utilities are in `verb/common.py`.

To see some examples of how to use the code, I suggest you look at `verb/demo.py` and the unit tests in `verb/unit_test.py`

To run the unit tests (all of which should pass), run:

```bash
pytest verb
```





















