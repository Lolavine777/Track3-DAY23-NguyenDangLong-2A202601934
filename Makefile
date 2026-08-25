.PHONY: install test lint typecheck run-scenarios grade-local clean

install:
	.venv/bin/pip install -e '.[dev]'

test:
	.venv/bin/pytest

lint:
	.venv/bin/ruff check src tests

typecheck:
	.venv/bin/mypy src

run-scenarios:
	.venv/bin/python -m langgraph_agent_lab.cli run-scenarios --config configs/lab.yaml --output outputs/metrics.json

grade-local:
	.venv/bin/python -m langgraph_agent_lab.cli validate-metrics --metrics outputs/metrics.json

judge:
	.venv/bin/python -m langgraph_agent_lab.cli judge --config configs/lab.yaml

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov dist build *.egg-info outputs/*.json


