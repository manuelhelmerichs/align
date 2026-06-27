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
- Standard checks:
  - `uv run pytest`
  - `uv run ruff check .`
  - `uv run ruff format --check .`
- Update `docs/align.md`, `docs/developer_reference_align.md`, and example configs when behavior or user-facing configuration changes.
- Use the open GitHub issues (`gh issue list`) as the current priority map.

## Research notes

- For nontrivial research or design decisions, create a concise action zettel in `~/Projects/zettelkasten/Action zettel` with the tag `#project/align`.
