# Genuine contribution plan

This plan gives teammates who are not yet represented in GitHub's automatic
default-branch contributor graph a small, useful task they can author and submit
themselves. The tasks were checked against the repository on 2026-08-01 and do
not duplicate an existing implementation or test.

| Teammate | Suggested task | Files likely involved | Difficulty | Acceptance criteria | Validation command | Already completed by someone else? |
| --- | --- | --- | --- | --- | --- | --- |
| Mohamed Amine Kar-any (`@mohamed-kar1`) | Add a manifest and CSV consistency validator for the 650-run proof artifacts | `scripts/validate_proof_artifacts.py`, `tests/test_proof_artifact_validation.py`, `docs/REPRODUCIBILITY.md` | Medium | Detect mismatched run counts, epoch counts, missing optimizer/condition combinations, and malformed manifests; return a non-zero exit code with an actionable error; document usage | `.venv/bin/pytest -q tests/test_proof_artifact_validation.py && .venv/bin/python scripts/validate_proof_artifacts.py --manifest results/v2/proof/full_core_epoch_history_manifest.json --runs results/v2/proof/full_core_run_results.csv --epochs results/v2/proof/full_core_epoch_history.csv` | No. Existing tests assert fixed file counts but there is no reusable command-line consistency validator. |
| Hamza Elhaddaji (`@HamzaElhaddaji`) | Add cross-configuration protocol contract tests | `tests/test_v2_config_cross_contract.py`, optionally `src/gnn_robustness/v2_config.py` | Small to medium | Load all four v2 YAML configurations; verify shared optimizer names, seed policy, perturbation names, severity bounds, split rules, and that only the tuned protocol contains tuning grids; failures identify the config and field | `.venv/bin/pytest -q tests/test_v2_config_cross_contract.py && make lint && make format-check` | No. Current tests cover individual fields but not consistency across all v2 protocols. |
| Iliass Ouchida (`@Iliassouchida`) | Add an accessibility and reduced-motion contract for the public dashboard | `docs/index.html`, `docs/assets/styles.css`, `docs/assets/app.js`, `tests/test_frontend_contract.py` | Medium | Add a keyboard-visible skip link, clear focus styles, a reduced-motion path that disables nonessential animation, and accessible canvas fallbacks or labels; retain the current visual design; cover every requirement with tests | `.venv/bin/pytest -q tests/test_frontend_contract.py && make lint && make format-check` | No. Existing frontend tests cover the visual system and interactions, not keyboard navigation or reduced motion. |

## Pull request expectations

Every task should arrive as a separate pull request authored by the teammate who
implemented it. The pull request must include:

- the issue or task identifier;
- a short explanation of the gap being fixed;
- the exact validation commands and their results;
- any scientific or usability limitation that remains;
- no unrelated generated artifacts.

The maintainer should review the actual diff and merge only after the acceptance
criteria pass. Completing these tasks provides real project value and genuine
authored history; it is not a badge-only exercise.
