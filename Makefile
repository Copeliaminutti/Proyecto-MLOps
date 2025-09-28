.PHONY: setup format lint test dvc-repro

setup:
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

format:
	black . && isort .

lint:
	flake8 .

test:
	pytest -q

dvc-repro:
	dvc repro
