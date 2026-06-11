# Testing Strategy

Kuchikae uses layered tests so lightweight development stays fast and model-backed validation stays explicit.

## Test layers

- `unit`: pure functions, dataclasses, and deterministic helpers
- `contract`: backend interface compliance and behavioral contracts
- `golden`: text transformation quality regression cases
- `e2e`: browser-driven Gradio behavior
- `slow`: real model loading and inference

## Dependency policy

- Core `uv sync` must stay lightweight.
- Heavy model dependencies belong in optional extras.
- Real model tests must not run as part of the default fast path.

## Skip policy

- Skip only when the required external backend is unavailable.
- Prefer `xfail` for known temporary gaps in non-critical contract coverage.
- Do not use skip to hide broken default behavior.

## Baseline commands

```bash
uv sync
uv run pytest
uv run pytest -m contract
uv run pytest -m golden
uv run pytest -m e2e
```
