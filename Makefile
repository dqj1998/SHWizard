.PHONY: install dev test clean build binary

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

test:
	pytest tests/ -v

clean:
	rm -rf build/ dist/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build:
	python setup.py sdist bdist_wheel

binary:
	pip install -e ".[build]"
	python build.py

format:
	black shwizard/ tests/

lint:
	flake8 shwizard/ --max-line-length=100 --ignore=E203,W503

all: clean install binary
