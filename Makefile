.ONESHELL:
SHELL := bash

.PHONY: install
install: ## Install the virtual environment and install the pre-commit hooks
	@echo "🚀 Creating virtual environment using uv"
	@uv sync
	@uv run pre-commit install

.PHONY: check _check

## Public target --------------------------------------------------------------
check:                     # 1st run; if anything fails we run a 2nd pass
	@$(MAKE) --no-print-directory _check \
	 || (echo '🔄 1st pass failed – trying once more …' >&2 ; \
	     $(MAKE) --no-print-directory _check)

## Private target (real work) -------------------------------------------------
_check:
	# stop on first error in each pass
	set -euo pipefail
	echo "🚀 Checking lock file consistency with 'pyproject.toml'"
	uv lock
	echo "🚀 Ruff lint (may auto-fix)"
	uv run ruff check src tests
	echo "🚀 pre-commit hooks (may auto-fix)"
	uv run pre-commit run -a
	echo "🚀 mypy static types"
	uv run mypy --config-file mypy.ini src
	echo "🚀 deptry – unused / missing deps"
	uv run deptry src || echo "⚠️ Deptry found dependency issues, but continuing with checks"

.PHONY: pbdiff
pbdiff: ## Copy git diff to clipboard
	 @git diff -- . ':(exclude)uv.lock' | pbcopy

.PHONY: streamlit
streamlit:
	@uv run streamlit run src/biomcp_trialgpt/streamlit_app/main.py

.PHONY: pyclean
pyclean: ## Remove temporary files
	@echo "Running pyclean to remove .pyc and cache"
	@uv run pyclean src tests

.PHONY: test
test: ## Test the code with pytest
	@echo "🚀 Testing code: Running pytest"
	@uv run python -m pytest -x --ff

.PHONY: cov
cov: ## Generate HTML coverage report
	@echo "🚀 Generating HTML coverage report"
	@uv run python -m pytest --cov --cov-config=pyproject.toml --cov-report=html --cov-fail-under=90

.PHONY: build
build: clean-build ## Build wheel file
	@echo "🚀 Creating wheel file"
	@uvx --from build pyproject-build --installer uv

.PHONY: clean-build
clean-build: ## Clean build artifacts
	@echo "🚀 Removing build artifacts"
	@uv run python -c "import shutil; import os; shutil.rmtree('dist') if os.path.exists('dist') else None"

.PHONY: publish
publish: ## Publish a release to PyPI.
	@echo "🚀 Publishing."
	@uvx twine upload --repository-url https://upload.pypi.org/legacy/ dist/*

.PHONY: build-and-publish
build-and-publish: build publish ## Build and publish.

.PHONY: docs-test
docs-test: ## Test if documentation can be built without warnings or errors
	@uv run mkdocs build -s

.PHONY: docs
docs: ## Build and serve the documentation
	@uv run mkdocs serve

.PHONY: help
help:
	@uv run python -c "import re; \
	[[print(f'\033[36m{m[0]:<20}\033[0m {m[1]}') for m in re.findall(r'^([a-zA-Z_-]+):.*?## (.*)$$', open(makefile).read(), re.M)] for makefile in ('$(MAKEFILE_LIST)').strip().split()]"

.DEFAULT_GOAL := help
