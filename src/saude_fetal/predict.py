"""Inferência com um pipeline salvo (MLflow ou joblib).

Nada é ajustado aqui: o pipeline é carregado e apenas ``predict``/``predict_proba``
são chamados.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from saude_fetal.config import load_config
from saude_fetal.data import validate_schema


def load_model(model_uri: str, *, tracking_backend: str | None = None):
    """Carrega o pipeline sklearn de um URI MLflow (ex.: ``runs:/<run_id>/model``)
    ou de um arquivo ``.joblib`` local confiável."""
    if model_uri.endswith(".joblib"):
        import joblib

        return joblib.load(model_uri)
    import mlflow.sklearn

    from saude_fetal.tracking import tracking_session

    mlflow_cfg = load_config("experiment")["mlflow"]
    with tracking_session(mlflow_cfg, backend=tracking_backend):
        return mlflow.sklearn.load_model(model_uri)


def predict(model, df: pd.DataFrame, data_cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    """Valida o schema de entrada e devolve classe prevista + scores por classe."""
    cfg = data_cfg or load_config("data")
    features = list(cfg["features"])
    X = validate_schema(
        df,
        features=features,
        target=None,
        classes=None,
        signed_features=cfg.get("signed_features", ()),
    )[features]
    class_names = {int(k): v for k, v in cfg["target"]["classes"].items()}
    out = pd.DataFrame({"prediction": model.predict(X)}, index=df.index)
    out["prediction_label"] = out["prediction"].map(class_names)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        for i, label in enumerate(model.classes_):
            out[f"score_{class_names.get(int(label), label)}"] = proba[:, i]
    return out
