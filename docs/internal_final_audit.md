# Internal Final Audit

This file is not linked from the public site. It records the final evidence state used for packaging.

## Result Counts

- Core Cora matrix: 650 raw rows.
- Cross-dataset validation: 300 raw rows across Cora, CiteSeer, and PubMed.
- Validation-locked tuned Cora evaluation: 100 raw rows.
- Inference-time robustness: 75 raw rows.
- Combined raw rows after aggregation: 1125.
- Aggregate summary rows: 140.
- Gradient diagnostics: 10000 epoch-level rows.
- Paired optimizer statistics: 20 comparison rows.

## Final Deliverables

- Final report: `reports/Final_Project_Report_GNN_Robustness.md`, `.tex`, `.pdf`.
- Defense deck: `deliverables/GCN_Robustness_Final_Defense.pptx`, `.pdf`.
- Defense script: `deliverables/Script_Soutenance_GNN_FR.md`, `.pdf`.
- Jury Q&A: `deliverables/Questions_Reponses_Jury_GNN_FR.md`, `.pdf`.
- GitHub Pages site source: `docs/`.
- Deployable downloads: `docs/assets/downloads/`.

## Methodology Notes

- Hyperparameter tuning is validation-only; test metrics are evaluated only after the locked choice file is written.
- Cross-dataset and inference-time experiments are complementary validations, not replacements for the main Cora protocol.
- Perturbations are random graph/input perturbations. Claims must not be generalized to optimized attacks or all GNN architectures.
- The active GitHub Actions workflow file is intentionally left local and uncommitted because workflow updates require a token with workflow permission.
