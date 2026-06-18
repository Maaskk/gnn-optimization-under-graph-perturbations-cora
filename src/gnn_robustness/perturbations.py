import torch


def add_feature_noise(features: torch.Tensor, severity: float, seed: int) -> torch.Tensor:
    """Add deterministic Gaussian feature noise for robustness experiments."""

    if severity < 0:
        raise ValueError("severity must be non-negative")
    noisy_features = features.clone()
    if severity == 0:
        return noisy_features

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    noise = torch.randn(
        features.shape,
        generator=generator,
        dtype=features.dtype,
        device="cpu",
    ).to(features.device)
    return noisy_features + (severity * noise)

