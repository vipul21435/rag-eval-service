# Developer entry points. Everything runs through uv; `make install` first.
UV ?= uv

.PHONY: install lint format typecheck test demo ci docker-build docker-up

install:  ## Install Python 3.12, the locked dependencies, all extras and the dev tools
	$(UV) sync --locked --all-extras --dev

lint:  ## Ruff lint and formatting check
	$(UV) run ruff check .
	$(UV) run ruff format --check .

format:  ## Apply ruff formatting and safe fixes
	$(UV) run ruff check --fix .
	$(UV) run ruff format .

typecheck:  ## Strict mypy over src, tests and examples
	$(UV) run mypy

test:  ## Test suite with coverage; needs no network, models or credentials
	$(UV) run pytest --cov=ragsvc --cov-report=term-missing

demo:  ## Ingest the sample documents and run three queries; the script pins the hash embedder (offline)
	$(UV) run python examples/demo.py

ci: lint typecheck test  ## What GitHub Actions runs

docker-build:  ## Build the container image
	docker build -t recallmcp:dev .

docker-up:  ## Start the service with docker compose on port 17995
	docker compose up --build
