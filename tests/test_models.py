import pytest

from saude_fetal.config import load_config
from saude_fetal.model import build_pipeline

from .conftest import FEATURES, TARGET


def test_all_configured_models_fit_and_predict(synthetic_df):
    """Todos os modelos de configs/model.yaml constroem, ajustam e predizem."""
    model_cfg = load_config("model")
    X, y = synthetic_df[FEATURES], synthetic_df[TARGET].astype(int)
    for name, cfg in model_cfg["models"].items():
        pipe = build_pipeline(cfg, FEATURES, random_state=0).fit(X, y)
        pred = pipe.predict(X)
        assert len(pred) == len(X), name
        assert set(pred) <= {1, 2, 3}, name
        assert pipe.predict_proba(X).shape == (len(X), 3), name


def test_unknown_estimator_raises():
    with pytest.raises(KeyError):
        build_pipeline({"estimator": "nao_existe"}, FEATURES)


def test_random_state_injected_and_reproducible(synthetic_df):
    cfg = {"estimator": "random_forest", "params": {"n_estimators": 20}}
    X, y = synthetic_df[FEATURES], synthetic_df[TARGET].astype(int)
    p1 = build_pipeline(cfg, FEATURES, random_state=1).fit(X, y).predict_proba(X)
    p2 = build_pipeline(cfg, FEATURES, random_state=1).fit(X, y).predict_proba(X)
    assert (p1 == p2).all()
