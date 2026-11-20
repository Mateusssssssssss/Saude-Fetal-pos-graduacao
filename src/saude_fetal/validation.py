"""Esquema de validação cruzada (somente no TRAIN)."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import numpy as np
import pandas as pd
from sklearn import config_context
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedGroupKFold,
    StratifiedKFold,
    cross_validate,
)

from saude_fetal.model import pipeline_param_grid


def feature_groups(X: pd.DataFrame, data_cfg: dict[str, Any]) -> np.ndarray | None:
    """Dá o mesmo grupo a exames com vetores de features idênticos."""
    if not data_cfg["split"].get("group_duplicate_features", False):
        return None
    return pd.MultiIndex.from_frame(X[data_cfg["features"]]).factorize()[0]


def make_cv(
    validation_cfg: dict[str, Any], random_state: int, *, grouped: bool = False
) -> StratifiedKFold | StratifiedGroupKFold:
    """Estratifica a divisão e mantém grupos inteiros quando configurados."""
    shuffle = bool(validation_cfg.get("shuffle", True))
    splitter = StratifiedGroupKFold if grouped else StratifiedKFold
    return splitter(
        n_splits=int(validation_cfg["n_splits"]),
        shuffle=shuffle,
        random_state=random_state if shuffle else None,
    )


def run_cv(
    pipeline, X: pd.DataFrame, y: pd.Series, cv: StratifiedKFold, scorers: dict[str, Any]
) -> tuple[pd.DataFrame, np.ndarray]:
    """Executa CV e gera predições OOF.

    Retorna:
    - DataFrame por fold com scores de treino do fold (``train_*``) e de
      validação do fold (``val_*``);
    - vetor OOF: cada exame recebe a predição de um modelo que NÃO o viu no ajuste.

    As predições OOF reaproveitam os estimadores ajustados na CV.
    """
    folds, fitted, validation_indices = evaluate_folds(pipeline, X, y, cv, scorers)
    oof_pred = np.empty(len(y), dtype=y.dtype)
    for estimator, val_idx in zip(fitted, validation_indices, strict=True):
        oof_pred[val_idx] = estimator.predict(X.iloc[val_idx])
    return folds, oof_pred


def evaluate_folds(pipeline, X, y, cv, scorers, *, groups=None):
    """Reutiliza os folds e estimadores para os diagnósticos fora da amostra."""
    routing = config_context(enable_metadata_routing=True) if groups is not None else nullcontext()
    with routing:
        res = cross_validate(
            pipeline,
            X,
            y,
            cv=cv,
            scoring=scorers,
            params={"groups": groups} if groups is not None else None,
            return_train_score=True,
            return_estimator=True,
            return_indices=True,
            error_score="raise",
        )
    folds = pd.DataFrame({"fold": np.arange(1, cv.get_n_splits() + 1)})
    for name in scorers:
        folds[f"train_{name}"] = res[f"train_{name}"]
        folds[f"val_{name}"] = res[f"test_{name}"]  # "test" do sklearn = fold de validação
    folds["fit_time_s"] = res["fit_time"]
    folds["n_train"] = [len(idx) for idx in res["indices"]["train"]]
    folds["n_validation"] = [len(idx) for idx in res["indices"]["test"]]
    return folds, res["estimator"], res["indices"]["test"]


def make_grid_search(
    pipeline, grid, validation_cfg, seed, scorers, *, grouped=False
) -> GridSearchCV:
    """Cria a busca interna, que também pode ser reajustada no TRAIN completo."""
    inner_cv = make_cv(
        {"n_splits": validation_cfg["inner_n_splits"], "shuffle": validation_cfg["shuffle"]},
        random_state=seed,
        grouped=grouped,
    )
    return GridSearchCV(
        pipeline,
        param_grid=pipeline_param_grid(pipeline, grid),
        cv=inner_cv,
        scoring=scorers,
        refit=validation_cfg["primary_metric"],
        return_train_score=True,
        error_score="raise",
        n_jobs=1,
    )


def summarize_folds(folds: pd.DataFrame) -> dict[str, float]:
    """Média e desvio-padrão por métrica de validação dos folds (``cv_<m>_mean/std``)."""
    out: dict[str, float] = {}
    for col in folds.columns:
        if col.startswith("val_"):
            name = col.removeprefix("val_")
            out[f"cv_{name}_mean"] = float(folds[col].mean())
            out[f"cv_{name}_std"] = float(folds[col].std(ddof=1))
        elif col.startswith("train_"):
            name = col.removeprefix("train_")
            out[f"cv_train_{name}_mean"] = float(folds[col].mean())
    return out
