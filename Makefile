.PHONY: install install-dev test lint format clean build

install:
	pip install .

install-dev:
	pip install -e ".[dev,interactive,full,docs]"

test:
	python -m pytest tests/ --cov=scope_rx

lint:
	ruff check .
	mypy scope_rx

format:
	ruff check --fix .
	black .
	isort .

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build:
	python -m build
