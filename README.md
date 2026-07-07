# align

`align` is a standalone CLI for post-processing posterior neural-network weight samples. It removes known scale symmetries and permutation symmetries so sampled parameter trees lie in a common basin.

## Install

The pinned JAX stack currently supports Python 3.11 or 3.12. The repo includes
`.python-version` so `uv` uses Python 3.12 automatically.

```bash
uv sync --extra dev
```

For CUDA 12 JAX wheels:

```bash
uv sync --extra cuda --extra dev  # CUDA 12 JAX wheels
```

## Input contract

An experiment directory should contain:

```text
<experiment_root>/
  samples/
    0/
      sample_0.npz
      sample_1.npz
    1/
      sample_0.npz
  tree_sampling  # preferred when present
  tree           # fallback
  module_graph.json  # optional, used by the resnet adapter
```

The default sample format is `pytree_npz`: each `.npz` stores leaves in the same order as `jax.tree.flatten(position)`, and the tree file is the matching pickled `PyTreeDef`. `align` decodes this into a canonical in-memory `WeightSample` before adapters or solvers run.

## MILE and SMILE

The public `EmanuelSommer/MILE` and `EmanuelSommer/SMILE` repositories use the same basic sample layout for posterior samples: `<experiment>/samples/<chain_id>/sample_<n>.npz` plus a pickled tree file. For tabular FCNs, use:

```yaml
paths:
  sample_format: pytree_npz
architecture: dense_mlp
adapter:
  layer_root: params.fcn
```

SMILE ResNet-style runs use a `core` module with `Conv_*`, `FRN_*`, and `Dense_*` names that match the built-in `resnet` adapter. Residual networks also need residual topology metadata, either as `<experiment_root>/module_graph.json` or via explicit `adapter.residual_joins`.

See [docs/producer_artifact_contract.md](docs/producer_artifact_contract.md) for details.

## Usage

```bash
align configs/examples/align.yaml --validate-only
align configs/examples/align.yaml --dry-run
align configs/examples/align.yaml --output-dir results/example/align/lap
```

Useful variants:

- `align <config> --resume`
- `align <config> --list-samples`
- `align <config> --force-cpu`
- `align <config> --force-gpu --per-device-batch 128`

With `uv`, prefix the same commands with `uv run`, for example:

```bash
uv run align configs/examples/align.yaml --validate-only
uv run align configs/examples/align.yaml --dry-run
uv run align configs/examples/align.yaml --output-dir results/example/align/lap
```

## Documentation

- [docs/theory.md](docs/theory.md): the symmetry model (scale + permutation) and objective/schedule background
- [docs/developer_reference.md](docs/developer_reference.md): CLI, config schema, runtime, and artifact layout
- [docs/producer_artifact_contract.md](docs/producer_artifact_contract.md): expected sampler output layout

Roadmap and open work are tracked as GitHub issues (see the paper and JOSS umbrella issues).

## Development

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```
