PYTHON ?= .venv/bin/python
PIP ?= .venv/bin/pip
PYTEST ?= .venv/bin/pytest
RUFF ?= .venv/bin/ruff
FORMATTER ?= .venv/bin/ruff format
COVERAGE ?= .venv/bin/coverage
PY_SCRIPTS := $(shell find scripts -name '*.py' -type f)

.PHONY: setup test coverage lint format-check smoke experiment-cora experiment-cross-dataset experiment-tuned experiment-inference aggregate build-site reproduce-final experiment-v2-cora experiment-v2-cross-dataset aggregate-v2 reproduce-v2-standard smoke-v2 benchmark-v2 tune-v2 diagnostics-v2 statistics-v2

setup:
	$(PIP) install -r requirements.txt

test:
	$(PYTEST) -q

coverage:
	$(COVERAGE) run -m pytest
	$(COVERAGE) report

lint:
	$(RUFF) check src tests $(PY_SCRIPTS)

format-check:
	$(FORMATTER) --check src tests $(PY_SCRIPTS)

smoke:
	$(PYTHON) scripts/run_v2_experiments.py --config configs/v2_fixed_cora.yaml --output-root results/ci_smoke --epochs 3 --seeds 42 --optimizers Adam,RMSProp --perturbations clean,feature_masking --severities 0.20 --max-runs 4 --label smoke
	$(PYTHON) scripts/aggregate_v2.py --raw-dir results/ci_smoke/raw --output-dir results/ci_smoke/aggregated

smoke-v2: smoke

experiment-v2-cora:
	$(PYTHON) scripts/run_v2_experiments.py --config configs/v2_fixed_cora.yaml

experiment-cora: experiment-v2-cora

experiment-v2-cross-dataset:
	$(PYTHON) scripts/run_v2_experiments.py --config configs/v2_cross_dataset.yaml

experiment-cross-dataset: experiment-v2-cross-dataset

experiment-tuned:
	$(PYTHON) scripts/tune_v2_hyperparameters.py --config configs/v2_tuned_cora.yaml --label final
	$(PYTHON) scripts/run_tuned_v2_evaluation.py --config configs/v2_tuned_cora.yaml --locked results/v2/tuned/locked_hyperparameters_final.json

experiment-inference:
	$(PYTHON) scripts/run_v2_experiments.py --config configs/v2_inference_robustness.yaml

aggregate-v2:
	$(PYTHON) scripts/aggregate_v2.py

statistics-v2:
	$(PYTHON) scripts/compare_v2_statistics.py

aggregate: aggregate-v2 statistics-v2

benchmark-v2:
	$(PYTHON) scripts/benchmark_v2_runtime.py --repeats 5 --epochs 5 --label local

diagnostics-v2:
	$(PYTHON) scripts/collect_v2_diagnostics.py --config configs/v2_fixed_cora.yaml

tune-v2:
	$(PYTHON) scripts/tune_v2_hyperparameters.py --config configs/v2_tuned_cora.yaml --label local

build-site:
	$(PYTHON) scripts/build_frontend_data.py
	$(PYTHON) scripts/generate_v2_embedding.py --epochs 5
	$(PYTHON) scripts/build_final_documents.py
	$(PYTHON) scripts/build_v2_site_data.py

reproduce-v2-standard: test lint format-check experiment-v2-cora aggregate-v2 build-site

reproduce-final: test lint format-check experiment-cora experiment-cross-dataset experiment-tuned experiment-inference aggregate diagnostics-v2 build-site
