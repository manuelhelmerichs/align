# AGENTS.md

## Project context

- `align` is symmetry-aware post-processing for neural-network weight samples.
  It removes exact continuous scale symmetries by canonicalization and aligns
  discrete or matrix-valued symmetries by matching samples to a common
  reference.
- It is a library and a CLI. Its input is a directory of weight samples grouped
  by chain plus a pickled JAX tree definition -- the input artifact contract in
  the [`wiki/`](wiki).

## Development stance

- Prefer clean, coherent designs even when they require breaking changes.
- Do not add compatibility shims, deprecated aliases, legacy config bridges,
  migration layers, or silent fallbacks unless explicitly requested.
- If an API, config schema, artifact layout, or abstraction is wrong, replace
  it cleanly and update all callers, tests, docs, and example configs in the
  same change.
- Keep changes scoped, but do not preserve accidental complexity just because
  it already exists.
- Treat benchmark and invariance failures as design feedback, not as cases to
  paper over.

## Boundaries

- `align` depends on nothing in a consuming project. No import of a sampler, an
  experiment, or a campaign registry, and no repository-root path assumption:
  no `__file__`-relative walk to a `runs/`, `results/`, or `cache/` tree.
  Everything a run needs arrives through `PathConfig` or a call argument.
- `align/benchmarks/` is the alignment benchmark suite -- exact synthetic
  orbits, posterior-level metrics, the regression harness, and
  `python -m align.benchmarks`. It may import `align`; `align` must never
  import it.
- Function-space metrics need the model and data behind a run, which only the
  sampler that wrote it can rebuild. `python -m align.benchmarks posterior
  --evaluator module:function` is that seam. Do not close it by reaching for a
  sampler here.
- Reusable research models and exact-orbit builders belong to
  `align/benchmarks/` under public names. Test modules must not import helpers
  from other test modules; shared test-only support belongs in an explicitly
  named private helper module.

## Workflow

- Use the project toolchain: `uv sync --extra dev`, then `uv run ...`.
- Read through the relevant pages of the [`wiki/`](wiki).
- Standard checks:
  - `uv run pytest`
  - `uv run ruff check .`
  - `uv run ruff format --check .`
- The benchmark regression gate:
  - `uv run python -m align.benchmarks regression --fast`
- Update the relevant pages in the `wiki/` submodule and the example configs in
  [`configs/`](configs) when behavior or user-facing configuration changes.
  Commit and push wiki changes before updating the submodule pointer here.
