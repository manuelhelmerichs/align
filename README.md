# align

`align` is a research CLI for post-processing posterior neural-network weight
samples. It removes exact continuous scale symmetries by canonicalization and
aligns discrete or matrix-valued symmetries by matching samples to a common
reference.

## Install

The project targets Python 3.13 and uses [`uv`](https://docs.astral.sh/uv/):

```bash
uv sync --extra dev
```

Optional accelerator environments:

```bash
uv sync --extra cuda --extra dev  # CUDA 13
uv sync --extra mps --extra dev   # experimental Apple Silicon MPS
```

## Quick start

Start from one of the configurations in [`configs/examples`](configs/examples):

```bash
uv run align configs/examples/align.yaml --validate-only
uv run align configs/examples/align.yaml --dry-run
uv run align configs/examples/align.yaml --output-dir results/example/align/lap
```

An input experiment contains posterior samples grouped by chain and a matching
pickled JAX tree definition. See the [input artifact
contract](https://github.com/manuelhelmerichs/align/wiki/Input-artifact-contract)
for the exact layout.

## Documentation

The [project wiki](https://github.com/manuelhelmerichs/align/wiki) contains:

- a [theory and concepts overview](https://github.com/manuelhelmerichs/align/wiki/Theory-and-concepts),
  including the supported architecture and algorithm matrix;
- architecture pages organized like `align.architectures`;
- the [developer reference](https://github.com/manuelhelmerichs/align/wiki/Developer-reference),
  configuration reference, runtime design, and artifact contracts.

The wiki source is also available in [`wiki/`](wiki) as a Git submodule. Clone
it together with this repository using:

```bash
git clone --recurse-submodules https://github.com/manuelhelmerichs/align.git
```

For an existing checkout, run `git submodule update --init`.

## Development

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

`align` is unpublished research software and its interfaces may change as the
symmetry model evolves.
