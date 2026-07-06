PYTHON ?= .venv/bin/python
PIP ?= .venv/bin/pip
PYTEST ?= .venv/bin/pytest
RUFF ?= .venv/bin/ruff
FORMATTER ?= .venv/bin/ruff format
COVERAGE ?= .venv/bin/coverage

.PHONY: setup test coverage lint format-check experiment-v2-cora experiment-v2-cross-dataset aggregate-v2 build-site reproduce-v2-standard smoke-v2 benchmark-v2 tune-v2 diagnostics-v2

setup:
	$(PIP) install -r requirements.txt

test:
	$(PYTEST) -q

coverage:
	$(COVERAGE) run -m pytest
	$(COVERAGE) report

lint:
	$(RUFF) check src scripts tests

format-check:
	$(FORMATTER) --check src scripts tests

smoke-v2:
	$(PYTHON) scripts/run_v2_experiments.py --config configs/v2_fixed_cora.yaml --output-root results/v2/smoke --epochs 3 --seeds 42 --optimizers Adam,RMSProp --perturbations clean,feature_masking --severities 0.20 --max-runs 4 --label smoke
	$(PYTHON) scripts/aggregate_v2.py --raw-dir results/v2/smoke/raw --output-dir results/v2/smoke/aggregated

experiment-v2-cora:
	$(PYTHON) scripts/run_v2_experiments.py --config configs/v2_fixed_cora.yaml

experiment-v2-cross-dataset:
	$(PYTHON) scripts/run_v2_experiments.py --config configs/v2_cross_dataset.yaml

aggregate-v2:
	$(PYTHON) scripts/aggregate_v2.py

benchmark-v2:
	$(PYTHON) scripts/benchmark_v2_runtime.py --repeats 5 --epochs 5 --label local

diagnostics-v2:
	$(PYTHON) scripts/collect_v2_diagnostics.py --config configs/v2_fixed_cora.yaml

tune-v2:
	$(PYTHON) scripts/tune_v2_hyperparameters.py --config configs/v2_tuned_cora.yaml --label local

build-site:
	$(PYTHON) scripts/build_frontend_data.py
	$(PYTHON) scripts/generate_v2_embedding.py --epochs 5
	$(PYTHON) scripts/build_v2_site_data.py

reproduce-v2-standard: test lint format-check experiment-v2-cora aggregate-v2 build-site
