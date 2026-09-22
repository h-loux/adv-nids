"""Unit tests for the projection module.

Run:  python -m pytest tests/test_projection.py -v
"""
import numpy as np
import pytest

from nids.attacks.projection import project


class TestProjection:
    """Tests for constraint enforcement in adversarial attacks."""

    def test_immutable_features_unchanged(self):
        """Immutable features (one-hot, protocol, etc.) must not change."""
        x_clean = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        x_adv = np.array([2.0, 3.0, 3.5, 5.0, 6.0])
        mask = np.array([True, True, True, True, False])  # last feature immutable
        eps = 1.0
        result = project(x_adv, x_clean, eps, mask)
        assert result[4] == x_clean[4], "Immutable feature was changed"

    def test_output_within_epsilon_ball(self):
        """Perturbation must not exceed eps on any mutable feature."""
        x_clean = np.random.randn(10)
        x_adv = x_clean + np.random.uniform(-10, 10, 10)
        mask = np.ones(10, dtype=bool)
        eps = 0.5
        result = project(x_adv, x_clean, eps, mask)
        delta = result - x_clean
        assert np.all(np.abs(delta) <= eps + 1e-6), "Perturbation exceeds eps"

    def test_integer_features_stay_integer(self):
        """Integer features (like packet counts) must remain integers."""
        x_clean = np.array([1.0, 10.0, 2.0])
        x_adv = np.array([1.3, 10.7, 2.2])
        mask = np.array([True, True, True])
        eps = 100.0
        result = project(x_adv, x_clean, eps, mask, int_features=[1])
        assert result[1] == int(result[1]), "Integer feature not rounded"

    def test_one_hot_group_sums_to_one(self):
        """One-hot encoded groups must still sum to 1 per sample after projection."""
        x_clean = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])  # 2 one-hot groups
        x_adv = np.array([0.5, 0.4, 0.1, 0.8, 0.3, -0.1])
        mask = np.array([True, True, True, True, True, True])
        eps = 10.0
        one_hot = {"proto": [0, 1, 2], "state": [3, 4, 5]}
        result = project(x_adv, x_clean, eps, mask, one_hot_groups=one_hot)
        assert result[0:3].sum() == 1.0, "One-hot group 1 does not sum to 1"
        assert result[3:6].sum() == 1.0, "One-hot group 2 does not sum to 1"
        assert np.all(result[0:3] >= 0) and np.all(result[0:3] <= 1)

    def test_batch_processing(self):
        """Projection works for batches (n, d)."""
        x_clean = np.random.randn(5, 10)
        x_adv = x_clean + np.random.uniform(-1, 1, (5, 10))
        mask = np.ones(10, dtype=bool)
        result = project(x_adv, x_clean, 0.3, mask)
        assert result.shape == x_adv.shape
