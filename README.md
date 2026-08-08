What does symmetry canonicalization change in the geometry of sampling-based inference (SAI), and when do those geometric changes translate into inferential benefit?

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

## Sampling posteriors

The repository bundles a sampler for producing such experiments in
[`sampling/`](sampling), a vendored port of
[EmanuelSommer/SMILE](https://github.com/EmanuelSommer/SMILE) (MILE/SMILE
Bayesian deep ensembles). It runs from the repository root and writes
artifacts that `align` consumes directly:

```bash
uv run python -m sampling -c configs/sampling/uci_benchmarks/tabular_regr_mile.yaml -d 10
```

See [`sampling/README.md`](sampling/README.md) and the wiki's [sampling
posteriors](https://github.com/manuelhelmerichs/align/wiki/Sampling-posteriors)
page. Image and HuggingFace/BPE text experiments need
`uv sync --extra dev --extra sampling`.

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

To regenerate the research program from a clean clone, start with the
[reproducibility profiles and dependency graph](experiments/REPRODUCING.md).
It distinguishes reusable sampler campaigns from protocol-specific aligned
caches and experiment results.

## Development

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

The source distribution installs `align`, `sampling`, `benchmarks`, and the
active `experiments` packages (the deferred archive is excluded). The benchmark
and experiment interfaces are research tooling rather than stable public APIs;
run their command-line harnesses from the repository root:

```bash
uv run python -m benchmarks --help
uv run python -m benchmarks regression --fast
uv run python -m benchmarks posterior \
  --experiment-root runs/reference/uci_airfoil_mile_8x200 \
  --architecture mlp \
  --recipe-kwargs '{"parameter_root":"params.fcn"}' \
  --samples-per-chain 25 --sample-step 8 --canonicalize
```

For saved sampling bundles, the posterior command reconstructs the exact model
and data split by default. Its report includes invariant chain-level functional
disagreement, the weight-averaging/BMA gap, and a raw-output symmetry-drift
certificate alongside weight-space metrics. Use `--no-functional-metrics` for
a weight-only run.
