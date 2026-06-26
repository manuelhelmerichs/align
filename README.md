# align

`align` is a standalone CLI for post-processing posterior neural-network weight samples. It removes known scale symmetries and permutation symmetries so sampled parameter trees lie in a common basin.

## Install

The pinned JAX stack currently supports Python 3.11 or 3.12. The repo includes
`.python-version` so `uv` uses Python 3.12 automatically.

```bash
cd ~/projects/align
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

For CUDA 12 JAX wheels:

```bash
python -m pip install -e ".[cuda,dev]"
```

With `uv`:

```bash
cd ~/projects/align
uv sync --extra dev
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

Each `.npz` must store leaves in the same order as `jax.tree.flatten(position)`, and the tree file must be the matching pickled `PyTreeDef`.

## MILE and SMILE

The public `EmanuelSommer/MILE` and `EmanuelSommer/SMILE` repositories use the same basic sample layout for posterior samples: `<experiment>/samples/<chain_id>/sample_<n>.npz` plus a pickled tree file. For tabular FCNs, use:

```yaml
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
align configs/examples/align.yaml --output-dir results/example/align/weight_matching
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
uv run align configs/examples/align.yaml --output-dir results/example/align/weight_matching
```

## Documentation

- [docs/align.md](docs/align.md): alignment background and supported methods
- [docs/developer_reference_align.md](docs/developer_reference_align.md): CLI, config schema, runtime, and artifact layout
- [docs/producer_artifact_contract.md](docs/producer_artifact_contract.md): expected sampler output layout
- [docs/issue_roadmap.md](docs/issue_roadmap.md): open issue priorities and implementation plan

## Development

```bash
python -m pytest
python -m ruff check .
python -m ruff format --check .
```

With `uv`:

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```
