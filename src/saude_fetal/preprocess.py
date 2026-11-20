"""Pré-processamento dentro de Pipeline (ajustado apenas no treino de cada fold)."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_preprocessor(features: list[str], scale: bool) -> ColumnTransformer:
    """Imputação pela mediana (robustez a ausentes futuros) + scaling opcional.

    - O imputador e o scaler aprendem estatísticas somente no ``fit``; como ficam
      dentro do Pipeline, a CV os reajusta no treino de cada fold (sem leakage).
    - Scaling é útil para modelos lineares/distância; árvores não precisam.
    """
    steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    return ColumnTransformer(
        transformers=[("num", Pipeline(steps), list(features))],
        remainder="drop",
        verbose_feature_names_out=False,
    )
