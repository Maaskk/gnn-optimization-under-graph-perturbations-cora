# Task Division

The project is split into two independent experiment tracks. Each person can load Cora, train a GCN, run their assigned experiments, save results, and write their report section without waiting for the other person.

Both tracks compare the same optimizers:

- Adam
- AdamW
- RMSProp
- AdaGrad
- SGD

## Shared Output Contract

Every result CSV should use these columns so final merging is easy:

```text
track,optimizer,perturbation_type,severity,seed,epochs,hidden_channels,dropout,learning_rate,weight_decay,test_accuracy,macro_f1,final_loss,training_time_seconds
```

Use these values for `perturbation_type`:

```text
clean
feature_noise
edge_removal
fake_edge_addition
```

## Person 1: Ossama

### Track A: Clean Training + Feature Noise

Ossama owns the clean optimizer comparison and feature-noise robustness. This is a full standalone part of the project.

### Step 1: Clean GCN Benchmark

- Load the Cora dataset with PyTorch Geometric.
- Implement or use a 2-layer GCN.
- Train the model on the clean graph.
- Compare Adam, AdamW, RMSProp, AdaGrad, and SGD.
- Save results to:

```text
results/ossama_clean_results.csv
```

### Step 2: Feature Noise Robustness

- Add Gaussian noise to node features.
- Test severity levels:

```text
0.05, 0.10, 0.20, 0.30
```

- Run all five optimizers for each noise level.
- Save results to:

```text
results/ossama_feature_noise_results.csv
```

### Step 3: Ossama Figures

Create:

- Clean graph loss convergence plot.
- Clean graph accuracy and macro F1 comparison.
- Feature-noise accuracy vs severity plot.
- Feature-noise macro F1 vs severity plot.

Save figures in:

```text
results/figures/
```

### Step 4: Ossama Report Section

Write:

- Dataset description.
- GCN architecture.
- Optimizer comparison protocol.
- Clean graph results.
- Feature-noise robustness interpretation.

## Person 2: Teammate

### Track B: Structural Graph Perturbations

The teammate owns structural robustness: edge removal and fake edge addition. This is also a full standalone part of the project.

### Step 1: Edge Removal Robustness

- Load the Cora dataset with PyTorch Geometric.
- Implement or use the same 2-layer GCN idea.
- Randomly remove graph edges.
- Test severity levels:

```text
0.05, 0.10, 0.20, 0.30
```

- Run Adam, AdamW, RMSProp, AdaGrad, and SGD.
- Save results to:

```text
results/teammate_edge_removal_results.csv
```

### Step 2: Fake Edge Addition Robustness

- Add random fake edges between nodes.
- Avoid duplicate edges when possible.
- Test severity levels:

```text
0.05, 0.10, 0.20, 0.30
```

- Run Adam, AdamW, RMSProp, AdaGrad, and SGD.
- Save results to:

```text
results/teammate_fake_edge_results.csv
```

### Step 3: Teammate Figures

Create:

- Edge-removal accuracy vs severity plot.
- Edge-removal macro F1 vs severity plot.
- Fake-edge accuracy vs severity plot.
- Fake-edge macro F1 vs severity plot.

Save figures in:

```text
results/figures/
```

### Step 4: Teammate Report Section

Write:

- Edge-removal method.
- Fake-edge-addition method.
- Structural robustness results.
- Interpretation of which optimizer is most stable under graph structure noise.

## Final Integration Together

This part happens only after both independent tracks are finished.

1. Combine all result CSVs into one table:

```text
results/all_results.csv
```

2. Create one final summary table comparing optimizers across:

- Clean graph
- Feature noise
- Edge removal
- Fake edge addition

3. Decide the final conclusion:

- Best clean accuracy.
- Best macro F1-score.
- Fastest convergence.
- Most robust optimizer under perturbations.
- Best overall optimizer for the project.

4. Merge both report sections into the final report.

## Branches

Ossama:

```bash
git checkout ossama/baseline
```

Teammate:

```bash
git checkout teammate/robustness-experiments
```

## Shared Rules

- Both people can work independently.
- Do not wait for the other person's code to begin experiments.
- Use the same optimizer list.
- Use the same CSV columns.
- Keep seeds fixed for fair comparison.
- Commit small changes with clear messages.
- Open pull requests into `main` when a track is ready.

## Step-by-Step Project Roadmap

1. Each person clones the repository.
2. Each person switches to their own branch.
3. Each person creates a runnable notebook or script for their track.
4. Ossama runs clean graph and feature-noise experiments.
5. Teammate runs edge-removal and fake-edge experiments.
6. Each person saves CSV files and figures.
7. Each person writes their own report section.
8. Merge both branches into `main`.
9. Combine result CSVs.
10. Build final plots, final conclusion, and final report.

