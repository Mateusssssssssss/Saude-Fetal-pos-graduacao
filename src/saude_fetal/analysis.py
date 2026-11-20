"""Funções de apoio à EDA (usadas no notebook 01_eda).

Tudo que orienta decisões de modelagem (associações, importâncias, faixas)
deve ser calculado SOMENTE no TRAIN.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold


def feature_summary(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Resumo por feature: ausentes, zeros, cardinalidade, quantis e assimetria."""
    rows = []
    for col in features:
        s = df[col]
        rows.append(
            {
                "feature": col,
                "dtype": str(s.dtype),
                "n": int(s.notna().sum()),
                "missing": int(s.isna().sum()),
                "zeros_prop": float((s == 0).mean()),
                "n_unique": int(s.nunique()),
                "min": s.min(),
                "p25": s.quantile(0.25),
                "median": s.median(),
                "p75": s.quantile(0.75),
                "p99": s.quantile(0.99),
                "max": s.max(),
                "mean": s.mean(),
                "std": s.std(),
                "skew": s.skew(),
            }
        )
    return pd.DataFrame(rows).set_index("feature")


def target_distribution(y: pd.Series, class_names: dict[int, str]) -> pd.DataFrame:
    """Contagem e proporção (0-1) por classe, na ordem fixa das classes."""
    counts = y.value_counts().reindex(list(class_names), fill_value=0)
    return pd.DataFrame(
        {
            "classe": [class_names[c] for c in counts.index],
            "n": counts.to_numpy(),
            "prop": (counts / counts.sum()).to_numpy(),
        },
        index=counts.index,
    )


def learn_bins(train_values: pd.Series, max_bins: int = 5) -> np.ndarray | None:
    """Aprende limites de faixas no TRAIN.

    - Se a feature tiver poucos valores distintos (<= max_bins), devolve ``None``
      e as faixas são os próprios valores.
    - Caso contrário, usa quantis (descartando limites repetidos, comuns em
      contagens com muitos zeros) e estende os extremos para ±inf, cobrindo
      valores futuros fora do intervalo observado.
    """
    s = train_values.dropna()
    if s.nunique() <= max_bins:
        return None
    edges = np.unique(np.quantile(s, np.linspace(0, 1, max_bins + 1)))
    if len(edges) < 3:  # quase constante: separa "== mínimo" de "> mínimo"
        edges = np.array([edges[0], edges[0] + 1e-9, s.max()])
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


def apply_bins(values: pd.Series, edges: np.ndarray | None) -> pd.Series:
    """Aplica faixas aprendidas no TRAIN."""
    if edges is None:
        return values.astype("category")
    return pd.cut(values, bins=edges, include_lowest=True)


def rate_by_bins(
    X: pd.DataFrame,
    y: pd.Series,
    feature: str,
    class_names: dict[int, str],
    max_bins: int = 5,
    edges: np.ndarray | None = None,
) -> pd.DataFrame:
    """Contagens e taxas do desfecho por faixa da feature (denominador = exames da faixa).

    Se ``edges`` não for informado, as faixas são aprendidas no próprio ``X``
    (use apenas com o TRAIN).
    """
    if edges is None:
        edges = learn_bins(X[feature], max_bins=max_bins)
    bins = apply_bins(X[feature], edges)
    table = pd.crosstab(bins, y).reindex(columns=list(class_names), fill_value=0)
    table.columns = [f"n_{class_names[c]}" for c in table.columns]
    table.insert(0, "n_total", table.sum(axis=1))
    for c in class_names.values():
        # proporção 0-1; NaN quando a faixa não tem exames
        table[f"taxa_{c}"] = table[f"n_{c}"] / table["n_total"].replace(0, np.nan)
    table.index = table.index.astype(str)
    table.index.name = feature
    return table


def association_table(
    X: pd.DataFrame, y: pd.Series, random_state: int = 42
) -> pd.DataFrame:
    """Associação feature–target (somente TRAIN).

    - Spearman: o target é ordinal (Normal < Suspeito < Patológico), então a
      correlação de postos mede associação monotônica com a gravidade.
    - Informação mútua: captura dependências não lineares (não tem sinal).
    """
    spearman = X.apply(lambda col: col.corr(y, method="spearman"))
    mi = mutual_info_classif(X, y, discrete_features=False, random_state=random_state)
    out = pd.DataFrame({"spearman": spearman, "abs_spearman": spearman.abs(), "mutual_info": mi})
    return out.sort_values("abs_spearman", ascending=False)


def cv_permutation_importance(
    estimator,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    scoring: str = "f1_macro",
    n_repeats: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """Permutation importance calculada no fold de validação de cada split do TRAIN.

    A importância é a queda média do ``scoring`` ao embaralhar a feature
    (na escala da métrica; pode ser negativa e não soma 1).
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    per_fold = []
    for fold, (tr, va) in enumerate(cv.split(X, y), start=1):
        model = clone(estimator).fit(X.iloc[tr], y.iloc[tr])
        res = permutation_importance(
            model,
            X.iloc[va],
            y.iloc[va],
            scoring=scoring,
            n_repeats=n_repeats,
            random_state=random_state,
        )
        per_fold.append(pd.Series(res.importances_mean, index=X.columns, name=fold))
    imp = pd.concat(per_fold, axis=1)
    return pd.DataFrame(
        {"perm_importance_mean": imp.mean(axis=1), "perm_importance_std": imp.std(axis=1, ddof=1)}
    ).sort_values("perm_importance_mean", ascending=False)


def duplicate_report(df: pd.DataFrame, features: list[str], target: str) -> dict[str, Any]:
    """Quantifica linhas repetidas (mesmas features) e conflitos de rótulo."""
    dup_full = int(df.duplicated(subset=features + [target]).sum())
    dup_feat = int(df.duplicated(subset=features).sum())
    conflicts = df.groupby(features)[target].nunique()
    return {
        "linhas": len(df),
        "duplicadas_features_e_target": dup_full,
        "duplicadas_somente_features": dup_feat,
        "combinacoes_unicas_de_features": int(len(conflicts)),
        "combinacoes_com_rotulos_conflitantes": int((conflicts > 1).sum()),
    }
