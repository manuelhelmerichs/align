# AGENTS.md

## Project context

- `align` is an unpublished research tool for post-processing posterior neural-network weight samples.
- The main research context is Bayesian deep learning, weight-space generative modeling, and treating posterior or weight samples as a data modality.
- The current user is the sole user. Optimize for correctness, clarity, and research velocity over preserving old interfaces.

## Development stance

- Prefer clean, coherent designs even when they require breaking changes.
- Do not add compatibility shims, deprecated aliases, legacy config bridges, migration layers, or silent fallbacks unless explicitly requested.
- If an API, config schema, artifact layout, or abstraction is wrong, replace it cleanly and update all callers, tests, docs, and example configs in the same change.
- Keep changes scoped, but do not preserve accidental complexity just because it already exists.
- Treat benchmark and invariance failures as design feedback, not as cases to paper over.

## Workflow

- Use the project toolchain: `uv sync --extra dev`, then `uv run ...`.
- Read through the relevant pages of the [`wiki/`](wiki).
- Standard checks:
  - `uv run pytest`
  - `uv run ruff check .`
  - `uv run ruff format --check .`
- Update the relevant pages in the `wiki/` submodule and example configs when
  behavior or user-facing configuration changes. Commit and push wiki changes
  before updating the submodule pointer in this repository.

## Experiments and results

- [`experiments/RESULTS.md`](experiments/RESULTS.md) is the central synthesis of
  completed experiment results and derived theory. Update it whenever an
  experiment changes a conclusion.
- `experiments/` contains group subfolders of thematically related
  experiments. Each group has:
  - `README.md` -- exactly what each experiment does and how to run it;
  - `RESULTS.md` -- all results and interpretation;
  - `TODO.md` -- open follow-up steps (only if any exist).
  README and RESULTS document only implemented and completed experiments;
  plans belong in TODO.md.
- Experiments in a group are numbered contiguously: one runnable script per
  experiment, named `e1_<slug>.py` … `eN_<slug>.py`, no gaps. Renumber (and
  rename the stored results accordingly) when experiments are added or
  removed.
- Shared group code (loaders, models, helpers, tests) lives in
  `experiments/<group>/shared/`, not next to the numbered scripts. Group tests
  run in the standard suite: `experiments` is on `testpaths`.
- `experiments/` and `benchmarks/` are installed packages. Import across them
  by dotted path (`from experiments.<group>.shared import ...`); never add a
  `sys.path` entry, and never give a module a name generic enough to collide
  (`common`, `utils`).
- Importing an experiment module must have no side effects: all work sits
  behind `main()` under `if __name__ == "__main__":`, and nothing mutates
  global JAX state (`jax.config.update`) at import time.
- Three generated repository-root trees, all gitignored:
  - `runs/` -- sampler campaign output, the *input* to experiments;
  - `results/` -- experiment output;
  - `cache/` -- derived intermediates shared across experiments.
  Nothing else is written outside them, and never inside `experiments/`.
- Every experiment writes exactly one directory,
  `results/<group>/<experiment>/`, holding `manifest.json`, `summary.json`,
  optional `summary.md`, and `artifacts/`. Use
  `experiment_run(__file__, ...)` from `benchmarks.results`, which derives the
  directory from the script path so the two cannot drift. Read another
  experiment's result with `load_summary(group, experiment)`, never by path.
- Experiments import `align`/`sampling`/`benchmarks` but never modify them,
  and never import from another experiment group. Glue that several groups
  need (campaign registry, loaders, chain-level alignment, the result layout)
  belongs in `benchmarks/`.
- Delete superseded experiment code and results when a newer implementation
  replaces them; do not keep parallel stale copies.
