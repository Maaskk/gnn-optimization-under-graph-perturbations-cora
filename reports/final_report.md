# Final Report - GNN Optimizer Robustness on Cora

## Project

**Title:** Optimization of Graph Neural Networks under Graph Perturbations: Adam vs AdamW vs RMSProp vs AdaGrad vs SGD on the Cora Citation Network

**Option:** Option 4 - Robustesse des GNN face au bruit dans le graphe

**Dataset:** Cora Citation Network

## Dataset And Model

The experiments use the Cora citation network from PyTorch Geometric's Planetoid dataset. Cora contains 2708 nodes, 10556 directed citation edges, 1433 node features, and 7 classes. The standard split contains 140 training nodes, 500 validation nodes, and 1000 test nodes.

All experiments use a 2-layer Graph Convolutional Network with 16 hidden channels, dropout 0.5, learning rate 0.01, weight decay 0.0005, seed 42, and 200 epochs. Keeping this protocol fixed makes the optimizer comparison fair across clean training, feature noise, edge removal, and fake edge addition.

## Perturbations

- **Clean graph:** original Cora graph and features.
- **Feature noise:** Gaussian noise added to node features at severities 0.05, 0.10, 0.20, and 0.30.
- **Edge removal:** random removal of a fraction of graph edges at the same severities.
- **Fake edge addition:** random non-existing directed edges added at the same severities.

## Clean Graph Results

| optimizer | test_accuracy | macro_f1 | final_loss | training_time_seconds |
| --- | --- | --- | --- | --- |
| Adam | 0.799 | 0.790 | 0.316 | 12.051 |
| AdamW | 0.799 | 0.790 | 0.058 | 21.081 |
| RMSProp | 0.795 | 0.787 | 0.381 | 41.659 |
| AdaGrad | 0.719 | 0.719 | 1.411 | 24.490 |
| SGD | 0.148 | 0.056 | 1.944 | 19.419 |

Adam and AdamW achieve the strongest clean accuracy in this run, both reaching 0.799 test accuracy. RMSProp is very close, while AdaGrad and especially SGD are weaker under the fixed hyperparameters.

## Robustness Summary

| optimizer | perturbation_type | mean_test_accuracy | mean_macro_f1 | mean_training_time_seconds |
| --- | --- | --- | --- | --- |
| Adam | clean | 0.799 | 0.790 | 12.051 |
| AdamW | clean | 0.799 | 0.790 | 21.081 |
| RMSProp | clean | 0.795 | 0.787 | 41.659 |
| AdaGrad | clean | 0.719 | 0.719 | 24.490 |
| SGD | clean | 0.148 | 0.056 | 19.419 |
| Adam | edge_removal | 0.783 | 0.776 | 48.636 |
| AdamW | edge_removal | 0.781 | 0.771 | 86.078 |
| RMSProp | edge_removal | 0.781 | 0.773 | 53.886 |
| AdaGrad | edge_removal | 0.714 | 0.710 | 25.123 |
| SGD | edge_removal | 0.141 | 0.054 | 49.701 |
| Adam | fake_edge_addition | 0.754 | 0.743 | 84.941 |
| RMSProp | fake_edge_addition | 0.751 | 0.741 | 51.504 |
| AdamW | fake_edge_addition | 0.728 | 0.719 | 87.740 |
| AdaGrad | fake_edge_addition | 0.649 | 0.654 | 39.936 |
| SGD | fake_edge_addition | 0.202 | 0.074 | 72.228 |
| Adam | feature_noise | 0.543 | 0.540 | 23.700 |
| RMSProp | feature_noise | 0.539 | 0.539 | 25.056 |
| AdaGrad | feature_noise | 0.534 | 0.528 | 16.098 |
| AdamW | feature_noise | 0.529 | 0.526 | 29.074 |
| SGD | feature_noise | 0.180 | 0.117 | 19.807 |

Best mean test accuracy by setting:

- Clean graph: **Adam** with 0.799.
- Feature noise: **Adam** with 0.543.
- Edge removal: **Adam** with 0.783.
- Fake edge addition: **Adam** with 0.754.

## Final Conclusion

Across all result rows, the best overall optimizer by mean test accuracy is **Adam** with 0.702 mean accuracy and 0.694 mean macro F1.

The main pattern is that adaptive optimizers are much stronger than SGD for this Cora GCN setup. Adam, AdamW, and RMSProp are the most competitive optimizers. Perturbations reduce performance compared with the clean graph, but the adaptive methods remain more stable than SGD.

## Reproducibility

Run the project tracks with:

```bash
python scripts/run_ossama_track.py --epochs 200
python scripts/run_structural_track.py --epochs 200
python scripts/finalize_project.py
```

Run tests with:

```bash
python -m pytest tests -q
```

Generated final files:

- `results/all_results.csv`
- `results/final_optimizer_summary.csv`
- `reports/final_report.md`
- `reports/ossama_section.md`
- figures in `results/figures/`
