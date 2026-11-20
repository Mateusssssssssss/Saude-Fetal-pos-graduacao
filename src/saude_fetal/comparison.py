"""Diagnósticos comparativos da CV aninhada e das probabilidades OOF."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from saude_fetal.evaluation import plot_metric_comparison

MODEL_LABELS = {"random_forest": "Random Forest", "xgboost": "XGBoost"}


def plot_external_comparison(results):
    """Exibe a média e a dispersão da CV externa por família."""
    comparison = results.rename(
        columns={
            f"cv_{metric}_{stat}": f"{metric}{'' if stat == 'mean' else '_std'}"
            for metric in ("f1_macro", "recall_macro", "precision_macro")
            for stat in ("mean", "std")
        }
    )
    comparison["model"] = comparison["model"].replace(MODEL_LABELS)
    return plot_metric_comparison(comparison, title="Validação externa", source_label="CV aninhada")


def compare_probabilities(y, probabilities, classes, *, prefix: str = "oof") -> dict[str, float]:
    """Avalia a ordenação por classe e a perda multiclasse para o conjunto indicado."""
    # O XGBoost produz float32. Corrige o arredondamento na soma após alinhar
    # as colunas em float64 para calcular a perda multiclasse.
    normalized = probabilities / probabilities.sum(axis=1, keepdims=True)
    metrics = {
        f"{prefix}_log_loss": log_loss(
            y, normalized[:, np.argsort(list(classes))], labels=sorted(classes)
        )
    }
    for index, (label, name) in enumerate(classes.items()):
        binary = y == label
        scores = probabilities[:, index]
        metrics[f"{prefix}_ap_{name.lower()}"] = average_precision_score(binary, scores)
        metrics[f"{prefix}_roc_auc_{name.lower()}"] = roc_auc_score(binary, scores)
    return metrics


def plot_comparison_diagnostics(
    results,
    folds_by_model,
    grids_by_model,
    probabilities_by_model,
    y,
    classes,
    critical_class,
    primary,
) -> dict[str, plt.Figure]:
    """Compila os gráficos agregados compartilhados pelo notebook e pelo MLflow."""
    names = results["model"].tolist()
    display = {name: MODEL_LABELS.get(name, name) for name in names}
    colors = {name: f"C{i}" for i, name in enumerate(names)}
    figures = {}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    for name in names:
        folds = folds_by_model[name]
        kwargs = {"color": colors[name], "marker": "o"}
        axes[0].plot(folds["fold"], folds[f"val_{primary}"], label=display[name], **kwargs)
        axes[1].plot(
            folds["fold"], folds[f"train_{primary}"], label=f"{display[name]} · treino", **kwargs
        )
        axes[1].plot(
            folds["fold"],
            folds[f"val_{primary}"],
            linestyle="--",
            label=f"{display[name]} · validação",
            **kwargs,
        )
    for ax, title in zip(
        axes, ["Mesmo fold para os dois modelos", "Treino versus validação"], strict=True
    ):
        ax.set(title=title, xlabel="Fold externo", ylabel=primary, ylim=(0, 1))
        ax.set_xticks(folds_by_model[names[0]]["fold"])
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
    figures["fold_diagnostics"] = fig

    fig, axes = plt.subplots(
        1, len(classes), figsize=(12, 4.2), layout="constrained", squeeze=False
    )
    ticks = np.arange(3)
    width = 0.8 / len(names)
    for ax, class_name in zip(axes[0], classes.values(), strict=True):
        for index, name in enumerate(names):
            row = results.loc[results["model"] == name].iloc[0]
            scores = [
                row[f"oof_{metric}_{class_name.lower()}"]
                for metric in ("f1", "recall", "precision")
            ]
            bars = ax.bar(
                ticks + (index - (len(names) - 1) / 2) * width,
                scores,
                width,
                color=colors[name],
                label=display[name],
            )
            ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
        ax.set(title=class_name, ylim=(0, 1.1))
        ax.set_xticks(ticks, ["F1", "Recall", "Precision"])
        ax.grid(axis="y", alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Métricas por classe · predições OOF")
    figures["oof_class_metrics"] = fig

    fig, axes = plt.subplots(
        2, len(classes), figsize=(13, 7.6), layout="constrained", squeeze=False
    )
    for index, (label, class_name) in enumerate(classes.items()):
        binary = y == label
        pr, roc = axes[:, index]
        pr.axhline(
            binary.mean(), color="gray", linestyle=":", label=f"Prevalência {binary.mean():.2f}"
        )
        roc.plot([0, 1], [0, 1], color="gray", linestyle=":")
        for name in names:
            scores = probabilities_by_model[name][:, index]
            precision, recall, _ = precision_recall_curve(binary, scores)
            fpr, tpr, _ = roc_curve(binary, scores)
            pr.plot(
                recall,
                precision,
                color=colors[name],
                label=f"{display[name]} · AP {average_precision_score(binary, scores):.2f}",
            )
            roc.plot(
                fpr,
                tpr,
                color=colors[name],
                label=f"{display[name]} · AUC {roc_auc_score(binary, scores):.2f}",
            )
        pr.set(title=class_name, xlabel="Recall", ylabel="Precision")
        roc.set(xlabel="Taxa de falsos positivos", ylabel="Recall")
        for ax in (pr, roc):
            ax.set(xlim=(0, 1), ylim=(0, 1.02))
            ax.legend(fontsize=7, loc="lower right")
            ax.grid(alpha=0.25)
    fig.suptitle("Precision–recall e ROC por classe · um contra todos · OOF")
    figures["oof_pr_roc"] = fig

    fig, (reliability, distribution) = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    binary = y == critical_class
    critical_index = list(classes).index(critical_class)
    reliability.plot([0, 1], [0, 1], color="gray", linestyle=":")
    for name in names:
        scores = probabilities_by_model[name][:, critical_index]
        observed, predicted = calibration_curve(binary, scores, n_bins=6, strategy="quantile")
        reliability.plot(predicted, observed, marker="o", color=colors[name], label=display[name])
        for mask, group, style in ((binary, "Patológico", "-"), (~binary, "Demais", "--")):
            distribution.hist(
                scores[mask],
                bins=np.linspace(0, 1, 11),
                weights=np.full(mask.sum(), 1 / mask.sum()),
                histtype="step",
                color=colors[name],
                linestyle=style,
                label=f"{display[name]} · {group}",
            )
    reliability.set(
        title="Calibração OOF",
        xlabel="Score médio",
        ylabel="Fração observada",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    distribution.set(
        title="Distribuição dos scores OOF", xlabel="Score patológico", ylabel="Proporção no grupo"
    )
    for ax in (reliability, distribution):
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
    fig.suptitle(f"Classe crítica · {classes[critical_class]}")
    figures["oof_critical_probabilities"] = fig

    fig, axes = plt.subplots(
        1,
        len(names),
        figsize=(5.5 * len(names), 4.3),
        layout="constrained",
        squeeze=False,
        sharey=True,
    )
    for ax, name in zip(axes[0], names, strict=True):
        grid = grids_by_model[name]
        positions = np.arange(1, len(grid) + 1)
        ax.errorbar(
            positions,
            grid[f"mean_test_{primary}"],
            yerr=grid[f"std_test_{primary}"],
            fmt="o",
            capsize=4,
            label="Validação interna",
        )
        ax.plot(
            positions,
            grid[f"mean_train_{primary}"],
            marker="o",
            linestyle="--",
            label="Treino interno",
        )
        ax.axvspan(0.6, 1.4, color="#dbeafe")
        ax.set(
            title=display[name], xlabel="Candidato na ordem da tabela do notebook", ylabel=primary
        )
        ax.set_xticks(positions)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
    fig.suptitle("GridSearchCV no TRAIN completo · seleção de hiperparâmetros")
    figures["grid_rank"] = fig
    return figures
