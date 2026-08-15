# align

`align` is symmetry-aware post-processing for neural-network weight samples. It
makes networks that differ only by known, function-preserving symmetries
comparable in weight space:

- **canonicalization** removes exact continuous symmetries (scale, and the
  softmax-head translation) by mapping each sample to a canonical
  representative of its orbit;
- **matching** aligns the remaining discrete or matrix-valued symmetries
  (permutations, sign flips, attention-head and rotation structure) by matching
  every sample to a common reference.

Both are function-preserving by construction: the network a sample represents
is unchanged, only its coordinates are.

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

Start from one of the configurations in [`configs/`](configs):

```bash
uv run align configs/align.yaml --validate-only
uv run align configs/align.yaml --dry-run
uv run align configs/align.yaml --output-dir results/example/align/lap
```

An input experiment is a directory of weight samples grouped by chain plus a
matching pickled JAX tree definition. `align` does not care which sampler wrote
it; see the [input artifact
contract](https://github.com/manuelhelmerichs/align/wiki/Input-artifact-contract)
for the exact layout.

## Benchmarks

`align/benchmarks` holds the alignment benchmark suite: exact synthetic orbits
with mathematically known optimal solutions, posterior-level quality metrics
over many chains, and a regression harness with thresholds.

```bash
uv run python -m align.benchmarks --help
uv run python -m align.benchmarks regression --fast
uv run python -m align.benchmarks posterior --experiment-root <dir> --architecture mlp
```

Function-space metrics need the model and data behind a run, which only the
sampler that produced it can rebuild. Supply them with
`--evaluator module:function`, a factory taking the experiment root and
returning loader keyword arguments.

## Documentation

The [project wiki](https://github.com/manuelhelmerichs/align/wiki) contains:

- a [theory and concepts overview](https://github.com/manuelhelmerichs/align/wiki/Theory-and-concepts),
  including the supported architecture and algorithm matrix;
- architecture pages organized like `align.architectures`;
- the [developer reference](https://github.com/manuelhelmerichs/align/wiki/Developer-reference),
  configuration reference, runtime design, and artifact contracts.

## Development

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

`align` is consumed as a submodule by
[`bnn-posterior-samples`](https://github.com/manuelhelmerichs/bnn-posterior-samples),
which supplies the samplers, campaigns, and experiments that use it.
