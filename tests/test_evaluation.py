import numpy as np
import pytest

from saude_fetal.evaluation import build_scorers, compute_metrics

CLASSES = {1: "Normal", 2: "Suspeito", 3: "Patologico"}


def test_compute_metrics_perfect():
    y = np.array([1, 1, 2, 3])
    m = compute_metrics(y, y, CLASSES)
    assert m["f1_macro"] == 1.0
    assert m["recall_patologico"] == 1.0


def test_compute_metrics_per_class_values():
    y_true = np.array([1, 1, 2, 3, 3])
    y_pred = np.array([1, 2, 2, 3, 1])
    m = compute_metrics(y_true, y_pred, CLASSES)
    assert m["recall_patologico"] == pytest.approx(0.5)  # 1 de 2 patológicos detectado
    assert m["precision_suspeito"] == pytest.approx(0.5)  # 1 de 2 alertas "suspeito" correto
    assert m["recall_normal"] == pytest.approx(0.5)


def test_zero_division_is_zero_for_never_predicted_class():
    y_true = np.array([1, 2, 3])
    y_pred = np.array([1, 1, 1])  # comportamento do Dummy most_frequent
    m = compute_metrics(y_true, y_pred, CLASSES)
    assert m["precision_patologico"] == 0.0
    assert m["recall_patologico"] == 0.0


def test_scorers_match_compute_metrics(synthetic_df):
    from sklearn.dummy import DummyClassifier

    from .conftest import FEATURES, TARGET

    X, y = synthetic_df[FEATURES], synthetic_df[TARGET].astype(int)
    model = DummyClassifier(strategy="stratified", random_state=0).fit(X, y)
    pred = model.predict(X)
    expected = compute_metrics(y, pred, CLASSES)
    for name, scorer in build_scorers(CLASSES).items():
        assert scorer(model, X, y) == pytest.approx(expected[name]), name
