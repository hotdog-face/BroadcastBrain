# BroadcastBrain -- development tasks.
PYTHON ?= python3
VENV   ?= .venv
BIN    := $(VENV)/bin

.PHONY: help venv install install-all test test-fast lint format typecheck check clean run watch dashboard doctor

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

venv:  ## Create the virtualenv
	$(PYTHON) -m venv $(VENV)

install: venv  ## Install the package plus dev tools
	$(BIN)/pip install -U pip
	$(BIN)/pip install -e '.[dev]'

install-all: venv  ## Install every optional backend
	$(BIN)/pip install -U pip
	$(BIN)/pip install -e '.[all,dev]'

test:  ## Run the whole suite
	$(BIN)/pytest

test-fast:  ## Skip the slow / ffmpeg-dependent tests
	$(BIN)/pytest -m "not slow and not requires_ffmpeg"

lint:  ## Lint with ruff
	$(BIN)/ruff check src tests

format:  ## Auto-fix lint findings
	$(BIN)/ruff check --fix src tests

typecheck:  ## Type-check with mypy --strict
	$(BIN)/mypy

check: lint typecheck test  ## Everything CI would run

run:  ## Process Incoming once
	$(BIN)/bb run

watch:  ## Process Incoming and keep watching
	$(BIN)/bb run --watch

dashboard:  ## Start the local dashboard
	$(BIN)/bb dashboard

doctor:  ## Report installed capabilities
	$(BIN)/bb doctor

clean:  ## Remove build and cache artifacts
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
