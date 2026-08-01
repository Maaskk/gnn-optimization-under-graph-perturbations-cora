# Contributing

Contributions must improve the scientific reproducibility, validation, tests, or
documentation of this project. Authorship belongs to the person who performs
and submits the work. Do not rewrite commit authors or add ceremonial commits.

## Setup

Python 3.11 or 3.12 is supported.

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

## Workflow

1. Choose an open issue with independently reviewable acceptance criteria.
2. Create a branch from `main` under your own GitHub account.
3. Implement the task and add or update tests for the behavior.
4. Run the task-specific validation command from the issue.
5. Run the standard quality checks below.
6. Open a pull request and explain the evidence, limitations, and commands run.
7. Keep the pull request focused on one task.

## Standard validation

```bash
make test
make lint
make format-check
```

Experiment-related changes should also run the bounded smoke protocol:

```bash
make smoke
```

Frontend changes must run the frontend contract tests:

```bash
.venv/bin/pytest -q tests/test_frontend_contract.py
```

## Scientific rules

- Do not select hyperparameters using the test split.
- Preserve deterministic seeds and record any new source of randomness.
- Do not replace raw experimental evidence with hand-written aggregates.
- Distinguish random perturbations from adversarial attacks.
- Report limitations and avoid universal claims from one dataset.
- Do not commit credentials, private datasets, caches, or local environments.

## Authorship and review

For future repository changes, each contributor must author their own commit and
submit their own branch or pull request. A maintainer may suggest, review, and
merge the work, but must not impersonate the contributor. This is not a request
for the academic team to repeat work that has already been completed and
submitted. The complete team and its completed project responsibilities are
documented in [CONTRIBUTORS.md](CONTRIBUTORS.md) and
[docs/TEAM_CONTRIBUTION_PLAN.md](docs/TEAM_CONTRIBUTION_PLAN.md).
