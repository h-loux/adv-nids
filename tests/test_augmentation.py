from types import SimpleNamespace

import numpy as np
import pytest

import nids.augmentation as augmentation


class _ToyModel:
    mutable_mask = np.array([True, True])
    meta = {}

    def predict_proba(self, X):
        X = np.asarray(X)
        attack = X[:, 0] < 10.0
        return np.column_stack((np.where(attack, 0.1, 0.9), np.where(attack, 0.9, 0.1)))

    def to_scaled(self, X):
        return np.asarray(X)

    def to_raw(self, X):
        return np.asarray(X)


def _meta():
    return {
        "feature_names": ["duration", "packet_count"],
        "mutable_mask": [True, True],
        "bounds": {
            "duration": {"min": -1.0, "max": 12.0},
            "packet_count": {"min": -1.0, "max": 12.0},
        },
        "int_features": [],
        "one_hot_groups": {},
    }


def test_generator_uses_only_detected_attack_training_rows_and_inherits_labels(monkeypatch):
    def fake_fgsm(model, source, eps, mask):
        candidate = np.asarray(source).copy()
        candidate[:, 0] += 0.1
        return SimpleNamespace(x_adv=candidate)

    monkeypatch.setattr(augmentation, "fgsm", fake_fgsm)
    train = {
        "X_raw": np.array([[0.0, 0.0], [1.0, 1.0], [9.95, 2.0], [11.0, 3.0]]),
        "y": np.array([0, 1, 1, 1]),
    }

    generated, stats = augmentation.generate_adversarial_training_augmentations(
        _ToyModel(), train, _meta(), method="FGSM", eps=0.2, max_samples=8, seed=7,
    )

    assert generated["X_raw"].shape == (2, 2)
    np.testing.assert_array_equal(generated["source_indices"], np.array([1, 2]))
    np.testing.assert_array_equal(generated["y"], np.array([1, 1]))
    np.testing.assert_allclose(generated["X_raw"][:, 0], np.array([1.1, 10.05]))
    assert stats["eligible_source_rows"] == 2
    assert stats["generated_valid_rows"] == 2
    assert stats["valid_candidates_predicted_benign"] == 1
    assert stats["data_scope"].endswith("held-out test split not used")


def test_generator_refuses_when_no_attack_rows_are_correctly_detected():
    model = _ToyModel()
    train = {"X_raw": np.array([[1.0, 0.0], [11.0, 1.0]]), "y": np.array([0, 1])}

    with pytest.raises(ValueError, match="No correctly detected malicious rows"):
        augmentation.generate_adversarial_training_augmentations(
            model, train, _meta(), method="FGSM", eps=0.2, max_samples=4,
        )


def test_generator_rejects_unknown_method():
    with pytest.raises(ValueError, match="FGSM or PGD"):
        augmentation.generate_adversarial_training_augmentations(
            _ToyModel(),
            {"X_raw": np.array([[1.0, 0.0]]), "y": np.array([1])},
            _meta(),
            method="CW",
        )
