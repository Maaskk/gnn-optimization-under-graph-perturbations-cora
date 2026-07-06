from pathlib import Path

from gnn_robustness.v2_config import load_v2_config, resolved_seed


def test_v2_fixed_config_documents_primary_protocol():
    config = load_v2_config(Path("configs/v2_fixed_cora.yaml"))

    assert config.experiment_id == "v2_fixed_cora"
    assert config.version == "v2"
    assert config.datasets == ("Cora",)
    assert config.protocol == "fixed"
    assert config.robustness_setting == "training_time"
    assert config.seeds == (42, 43, 44, 45, 46, 47, 48, 49, 50, 51)
    assert config.perturbations == (
        "clean",
        "feature_masking",
        "edge_removal",
        "fake_edge_addition",
    )
    assert config.severities == (0.05, 0.10, 0.20, 0.30)
    assert config.training.epochs == 200
    assert config.training.hidden_channels == 16
    assert config.training.dropout == 0.5


def test_tuned_config_parses_sgd_momentum_grid():
    config = load_v2_config(Path("configs/v2_tuned_cora.yaml"))

    assert config.tuning.enabled is True
    assert config.tuning.sgd_momentum_values == (0.0, 0.9)


def test_resolved_seed_is_deterministic_and_context_specific():
    first = resolved_seed(
        dataset="Cora",
        optimizer="Adam",
        base_seed=42,
        experiment_mode="fixed",
        perturbation_type="feature_masking",
        severity=0.2,
    )
    second = resolved_seed(
        dataset="Cora",
        optimizer="Adam",
        base_seed=42,
        experiment_mode="fixed",
        perturbation_type="feature_masking",
        severity=0.2,
    )
    changed = resolved_seed(
        dataset="Cora",
        optimizer="AdamW",
        base_seed=42,
        experiment_mode="fixed",
        perturbation_type="feature_masking",
        severity=0.2,
    )

    assert first == second
    assert first != changed
    assert 0 <= first < 2**31
