"""Construção dos pipelines de modelo a partir de ``configs/model.yaml``."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted

from saude_fetal.preprocess import build_preprocessor


class EncodedClassifier(ClassifierMixin, BaseEstimator):
    """Adapta o XGBoost a rótulos externos 1/2/3 em cada ajuste."""

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        self.label_encoder_ = LabelEncoder().fit(y)
        self.classes_ = self.label_encoder_.classes_
        self.estimator_ = clone(self.estimator).fit(X, self.label_encoder_.transform(y))
        return self

    def predict(self, X):
        check_is_fitted(self, "estimator_")
        encoded = np.asarray(self.estimator_.predict(X), dtype=int)
        return self.label_encoder_.inverse_transform(encoded)

    def predict_proba(self, X):
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict_proba(X)


def build_estimator(name: str, params: dict[str, Any] | None, random_state: int | None):
    """Instancia o estimador; injeta ``random_state`` se ele aceitar e não vier em params."""
    kwargs = dict(params or {})
    kwargs.setdefault("random_state", random_state)
    if name == "xgboost":
        from xgboost import XGBClassifier

        return EncodedClassifier(XGBClassifier(**kwargs))
    if name == "random_forest":
        return RandomForestClassifier(**kwargs)
    raise KeyError(f"Estimador desconhecido: {name}. Opções: random_forest, xgboost")


def pipeline_param_grid(pipeline: Pipeline, params: dict[str, list[Any]]) -> dict[str, list[Any]]:
    """Traduz os nomes da grade YAML para os parâmetros do pipeline."""
    prefix = (
        "model__estimator__"
        if isinstance(pipeline.named_steps["model"], EncodedClassifier)
        else "model__"
    )
    return {f"{prefix}{key}": values for key, values in params.items()}


def build_pipeline(
    model_cfg: dict[str, Any], features: list[str], random_state: int | None = None
) -> Pipeline:
    """Pipeline completo: pré-processamento + estimador (serializável como unidade)."""
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor(features, scale=bool(model_cfg.get("scale", False)))),
            (
                "model",
                build_estimator(model_cfg["estimator"], model_cfg.get("params"), random_state),
            ),
        ]
    )
