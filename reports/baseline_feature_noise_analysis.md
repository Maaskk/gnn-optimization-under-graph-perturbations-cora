# Baseline and Feature-Noise Analysis

## Experimental Setup

The experiment uses the Cora citation network, as requested in the project topic. Cora contains 2708 scientific papers, 10556 directed citation edges, 1433 node features, and 7 node classes. The standard Planetoid split is used: 140 training nodes, 500 validation nodes, and 1000 test nodes.

The model is a 2-layer Graph Convolutional Network (GCN) with 16 hidden channels, ReLU activation, dropout of 0.5, and cross-entropy loss on the training mask. Each optimizer is trained for 200 epochs using the same learning rate, weight decay, model architecture, seed, and dataset split to keep the comparison fair.

The required optimizers compared in this track are Adam, AdamW, RMSProp, AdaGrad, and SGD. AMSGrad is also available in the code as an optional bonus optimizer because the professor's PDF lists it among possible optimizers.

## Clean Graph Results

| Optimizer | Test Accuracy | Macro F1 | Final Loss | Training Time (s) |
|---|---:|---:|---:|---:|
| Adam | 0.799 | 0.790 | 0.316 | 12.05 |
| AdamW | 0.799 | 0.790 | 0.058 | 21.08 |
| RMSProp | 0.795 | 0.787 | 0.381 | 41.66 |
| AdaGrad | 0.719 | 0.719 | 1.411 | 24.49 |
| SGD | 0.148 | 0.056 | 1.944 | 19.42 |

Adam and AdamW achieved the best clean test accuracy at 79.9%. RMSProp was very close at 79.5%. AdaGrad converged more weakly under the same hyperparameters, and SGD performed poorly with the fixed learning rate and epoch budget.

## Feature-Noise Robustness

Feature noise is added as Gaussian noise to the node feature matrix. The tested severities are 0.05, 0.10, 0.20, and 0.30. Each optimizer is trained and evaluated on the noisy feature setting.

Average performance across the four feature-noise levels:

| Optimizer | Mean Test Accuracy | Mean Macro F1 |
|---|---:|---:|
| Adam | 0.543 | 0.540 |
| RMSProp | 0.539 | 0.539 |
| AdaGrad | 0.534 | 0.528 |
| AdamW | 0.529 | 0.526 |
| SGD | 0.180 | 0.117 |

Adam had the best average test accuracy under feature noise, followed very closely by RMSProp. RMSProp was the strongest at the smallest perturbation level, while Adam was slightly better on average across all tested severities. SGD remained unstable and did not learn a useful classifier under these fixed conditions.

## Interpretation

On the clean Cora graph, adaptive optimizers clearly outperform SGD. Adam, AdamW, and RMSProp are the strongest choices for the GCN baseline. Under feature noise, the performance of all adaptive optimizers drops substantially, which shows that noisy node attributes make the classification task harder even when the graph structure is unchanged.

For this track, Adam is the best overall optimizer because it matches the best clean accuracy and has the highest mean robustness under feature noise. RMSProp is also competitive, especially at low noise. AdamW reaches excellent clean performance but is slightly weaker than Adam and RMSProp under feature noise in this run.

## Files Produced

- `results/clean_optimizer_results.csv`
- `results/feature_noise_results.csv`
- `results/clean_loss_history.csv`
- `results/cora_dataset_summary.csv`
- `results/figures/clean_loss_convergence.png`
- `results/figures/clean_accuracy_macro_f1.png`
- `results/figures/feature_noise_test_accuracy.png`
- `results/figures/feature_noise_macro_f1.png`
