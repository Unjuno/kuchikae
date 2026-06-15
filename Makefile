PYTHON ?= uv run python

.PHONY: install setup run run-real run-streaming doctor test lint typecheck security check-all clean

install:
	uv sync

install-real:
	uv sync --extra real

setup: install-real
	uv run kuchikae setup-models --all
	uv run kuchikae doctor

run: install
	uv run kuchikae serve --dummy

run-real: install-real setup
	uv run kuchikae serve --real

run-streaming: install-real setup
	uv run kuchikae serve --real --streaming

doctor:
	uv run kuchikae doctor

test: install
	uv run pytest -q -m "not slow and not e2e"

lint:
	uv run ruff check kuchikae

typecheck:
	uv run mypy kuchikae

security:
	uv run pip-audit --strict

check-all: lint typecheck security
	uv run python -m compileall kuchikae
	uv run pytest -q -x -m "not slow and not e2e"

clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache .mypy_cache
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
