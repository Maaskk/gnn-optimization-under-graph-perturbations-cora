import torch

from gnn_robustness.perturbations import add_feature_noise


def test_zero_feature_noise_returns_equal_clone_without_aliasing():
    features = torch.tensor([[0.0, 1.0], [1.0, 0.0]])

    noisy = add_feature_noise(features, severity=0.0, seed=123)

    assert torch.equal(noisy, features)
    assert noisy.data_ptr() != features.data_ptr()


def test_feature_noise_is_deterministic_for_same_seed():
    features = torch.ones((4, 3))

    first = add_feature_noise(features, severity=0.2, seed=42)
    second = add_feature_noise(features, severity=0.2, seed=42)

    assert torch.allclose(first, second)


def test_feature_noise_changes_values_and_preserves_shape():
    features = torch.zeros((5, 4))

    noisy = add_feature_noise(features, severity=0.2, seed=7)

    assert noisy.shape == features.shape
    assert not torch.equal(noisy, features)


def test_feature_noise_rejects_negative_severity():
    features = torch.zeros((2, 2))

    try:
        add_feature_noise(features, severity=-0.1, seed=1)
    except ValueError as exc:
        assert "severity" in str(exc)
    else:
        raise AssertionError("Expected ValueError for negative severity")

