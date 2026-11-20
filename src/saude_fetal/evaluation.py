"""Métricas e gráficos de avaliação para classificação multiclasse.

Convenções:
- Proporções em escala 0-1 (conversão para % só na exibição).
- ``zero_division=0``: quando um modelo não prevê uma classe, precision dessa
  classe é reportada como 0 (ex.: baseline Dummy). Escolha explícita.
- Ordem das classes fixa, definida em ``configs/data.yaml``.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)

# Cores fixas por métrica (mesma convenção em todos os notebooks).
METRIC_COLORS = {"f1": "#1f77b4", "recall": "#ff7f0e", "precision": "#2ca02c"}


def class_names_from_cfg(data_cfg: dict[str, Any]) -> dict[int, str]:
    """{rótulo: nome} na ordem do YAML."""
    return {int(k): str(v) for k, v in data_cfg["target"]["classes"].items()}


def build_scorers(class_names: dict[int, str]) -> dict[str, Any]:
    """Scorers para ``cross_validate`` (macro, weighted e por classe)."""
    labels = list(class_names)
    scorers: dict[str, Any] = {
        "accuracy": make_scorer(accuracy_score),
        "balanced_accuracy": make_scorer(balanced_accuracy_score),
        "f1_macro": make_scorer(f1_score, labels=labels, average="macro", zero_division=0),
        "recall_macro": make_scorer(recall_score, labels=labels, average="macro", zero_division=0),
        "precision_macro": make_scorer(
            precision_score, labels=labels, average="macro", zero_division=0
        ),
        "f1_weighted": make_scorer(f1_score, labels=labels, average="weighted", zero_division=0),
    }
    for label, name in class_names.items():
        key = name.lower()
        for metric_name, fn in (
            ("recall", recall_score),
            ("precision", precision_score),
            ("f1", f1_score),
        ):
            scorers[f"{metric_name}_{key}"] = make_scorer(
                fn, labels=[label], average="macro", zero_division=0
            )
    return scorers


def compute_metrics(
    y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray, class_names: dict[int, str]
) -> dict[str, float]:
    """Métricas a partir de rótulos previstos (mesmos nomes dos scorers)."""
    labels = list(class_names)
    out: dict[str, float] = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "recall_macro": recall_score(
            y_true, y_pred, labels=labels, average="macro", zero_division=0
        ),
        "precision_macro": precision_score(
            y_true, y_pred, labels=labels, average="macro", zero_division=0
        ),
        "f1_weighted": f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0),
    }
    rec = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    prec = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    for i, name in enumerate(class_names.values()):
        key = name.lower()
        out[f"recall_{key}"] = rec[i]
        out[f"precision_{key}"] = prec[i]
        out[f"f1_{key}"] = f1[i]
    return {k: float(v) for k, v in out.items()}


def classification_report_dict(y_true, y_pred, class_names: dict[int, str]) -> dict[str, Any]:
    """Relatório por classe (inclui support) em formato JSON-serializável."""
    return classification_report(
        y_true,
        y_pred,
        labels=list(class_names),
        target_names=list(class_names.values()),
        output_dict=True,
        zero_division=0,
    )


def plot_confusion_matrices(y_true, y_pred, class_names: dict[int, str], title: str) -> plt.Figure:
    """Matriz de confusão por contagem e normalizada pela classe real (linhas)."""
    labels = list(class_names)
    names = list(class_names.values())
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    with np.errstate(invalid="ignore", divide="ignore"):
        cm_norm = cm / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, mat, subtitle, fmt in (
        (axes[0], cm, "Contagens", "d"),
        (axes[1], cm_norm, "Normalizada pela classe real", ".2f"),
    ):
        im = ax.imshow(mat, cmap="Blues")
        ax.set_xticks(range(len(names)), names)
        ax.set_yticks(range(len(names)), names)
        ax.set_xlabel("Classe prevista")
        ax.set_ylabel("Classe real")
        ax.set_title(subtitle)
        vmax = np.nanmax(mat) if np.size(mat) else 0
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat[i, j]
                text = "nan" if np.isnan(val) else format(val, fmt)
                color = "white" if (not np.isnan(val) and val > vmax / 2) else "black"
                ax.text(j, i, text, ha="center", va="center", color=color)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_metric_comparison(
    df: pd.DataFrame,
    model_col: str = "model",
    metrics: tuple[str, str, str] = ("f1_macro", "recall_macro", "precision_macro"),
    std_suffix: str | None = "_std",
    title: str = "Comparação de modelos",
    source_label: str = "CV (média ± desvio)",
) -> plt.Figure:
    """Barras agrupadas F1 / Recall / Precision por modelo, com rótulos numéricos.

    ``df`` deve ter uma linha por modelo e as colunas das métricas (e,
    opcionalmente, ``<metrica><std_suffix>`` para barras de erro).
    Ordenado pela primeira métrica (principal).
    """
    data = df.sort_values(metrics[0], ascending=False).reset_index(drop=True)
    x = np.arange(len(data))
    width = 0.27
    fig, ax = plt.subplots(figsize=(max(8, 1.6 * len(data)), 5))
    display_names = {
        "f1_macro": "F1 macro",
        "recall_macro": "Recall macro",
        "precision_macro": "Precision macro",
    }
    for i, metric in enumerate(metrics):
        base = metric.split("_")[0]
        values = data[metric].to_numpy(dtype=float)
        err = None
        if std_suffix and f"{metric}{std_suffix}" in data:
            err = data[f"{metric}{std_suffix}"].to_numpy(dtype=float)
        bars = ax.bar(
            x + (i - 1) * width,
            values,
            width,
            yerr=err,
            capsize=3,
            label=display_names.get(metric, metric),
            color=METRIC_COLORS.get(base),
        )
        offsets = err if err is not None else np.zeros_like(values)
        for bar, v, offset in zip(bars, values, offsets):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                min(v + offset + 0.018, 1.075),
                f"{v:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.set_xticks(x, data[model_col])
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score (escala 0-1)")
    ax.set_title(f"{title} — {source_label}")
    ax.legend(loc="upper right", ncol=3)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig
