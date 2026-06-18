# Task Division

This plan splits the project into two parallel workstreams: one person owns the reliable baseline and experiment infrastructure, and the other owns perturbations, robustness evaluation, and final analysis. Both people should review each other's pull requests before merging.

## Person 1: Ossama

### Step 1: Project Setup

- Create Python virtual environment.
- Install PyTorch, PyTorch Geometric, scikit-learn, pandas, matplotlib, and tqdm.
- Create the first runnable script structure under `src/`.
- Verify that Cora loads correctly.

### Step 2: Baseline GCN

- Implement the GCN model in `src/model.py`.
- Implement training and evaluation loops in `src/train.py`.
- Track train loss, validation accuracy, test accuracy, macro F1-score, and training time.
- Run the first baseline with Adam.

### Step 3: Optimizer Comparison

- Add Adam, AdamW, RMSProp, AdaGrad, and SGD configurations.
- Keep the same model, seed, train split, hidden size, dropout, learning-rate policy, and epoch budget for fair comparison.
- Save raw metrics to `results/optimizer_comparison.csv`.
- Generate plots for loss curves and accuracy/F1 comparison.

### Step 4: Baseline Report Section

- Write the dataset description.
- Explain the GCN architecture.
- Explain the optimizer comparison protocol.
- Prepare baseline result tables and figures.

## Person 2: Teammate

### Step 1: Perturbation Functions

- Implement edge removal in `src/perturbations.py`.
- Implement fake edge addition in `src/perturbations.py`.
- Implement feature noise in `src/perturbations.py`.
- Make perturbation severity configurable, for example `0.05`, `0.10`, `0.20`, and `0.30`.

### Step 2: Robustness Evaluation

- Reuse Ossama's training/evaluation loop.
- Evaluate every optimizer under every perturbation type and severity.
- Save raw metrics to `results/robustness_results.csv`.
- Measure robustness drop compared with the clean graph baseline.

### Step 3: Robustness Plots

- Plot accuracy vs perturbation severity.
- Plot macro F1-score vs perturbation severity.
- Plot robustness drop for each optimizer.
- Compare optimizer stability using mean and standard deviation across seeds.

### Step 4: Final Analysis Section

- Explain each perturbation method.
- Interpret which optimizer is most robust.
- Discuss trade-offs between convergence speed and robustness.
- Write limitations and possible improvements.

## Shared Rules

- Use branches: `ossama/baseline` and `teammate/perturbations`.
- Commit small changes with clear messages.
- Put generated metrics and figures in `results/`.
- Keep random seeds fixed when comparing optimizers.
- Merge through pull requests so both people review the work.

## Step-by-Step Project Roadmap

1. Create environment and install dependencies.
2. Load and inspect the Cora dataset.
3. Build a clean GCN baseline.
4. Validate the baseline with Adam.
5. Add all five optimizers.
6. Run clean-graph optimizer comparison.
7. Add perturbation functions.
8. Run robustness experiments.
9. Generate plots and tables.
10. Write the final report.
11. Prepare presentation slides if required.
12. Final review: check reproducibility, figures, and conclusions.

