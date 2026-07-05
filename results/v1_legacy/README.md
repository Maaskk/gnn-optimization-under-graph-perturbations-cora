# Legacy V1 Provenance

These files preserve the original class-project outputs before the V2 reproducibility upgrade.

V1 was a single-seed, fixed-protocol experiment:

- Dataset: Cora only.
- Seed: 42.
- Epochs: 200.
- Model: 2-layer GCN with 16 hidden channels and dropout 0.5.
- Optimizers: Adam, AdamW, RMSProp, AdaGrad, SGD.
- Hyperparameters: one fixed protocol shared by all optimizers.
- Feature perturbation: Gaussian feature-noise standard deviation sigma values 0.05, 0.10, 0.20, 0.30.
- Structural perturbations: V1 directed-edge removal and directed fake-edge addition.
- Measurement setting: training after perturbing graph/features, then testing on the perturbed data.

V1 is useful as a legacy baseline, but it should not be described as multi-seed evidence, tuned optimizer evidence, adversarial robustness, or percentage-based feature corruption.
