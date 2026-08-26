PYTHON ?= python3

.PHONY: test lint format check

test:
	PYTHONPATH=. $(PYTHON) -m unittest discover -s tests -v

lint:
	ruff check rtnav agent_node.py tests scripts

format:
	ruff format rtnav agent_node.py tests scripts
	ruff check --fix rtnav agent_node.py tests scripts

check: lint test
