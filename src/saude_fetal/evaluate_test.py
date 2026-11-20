"""Avalia no TEST as duas receitas XGBoost congeladas antes da comparação."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from saude_fetal.comparison import compare_probabilities
from saude_fetal.config import PROJECT_ROOT, REPORTS_DIR, load_config, resolve_path
from saude_fetal.data import file_sha256, load_split
from saude_fetal.evaluation import (
    class_names_from_cfg,
    classification_report_dict,
    compute_metrics,
    plot_confusion_matrices,
    plot_metric_comparison,
)
from saude_fetal.tracking import tracking_session

OUTPUT = REPORTS_DIR / "test" / "xgboost_full_vs_reduced_v1"
FULL_VERSION = "models:/saude-fetal-xgboost/1"
REDUCED_PARENT = "be00b3caa438436eb7288def11dd2bea"
REDUCED_RUN = "029dd288f1284c45ab54708dcf94a3e2"
REDUCED_SHA256 = "eed1eaaeb453c6d2c42666eb9c6de5a6356ca1ad2309a71818f609c520d65dee"


def read_json(path):
    return json.loads(resolve_path(path).read_text(encoding="utf-8"))


def load_frozen_test(cfg, training_sha):
    split = cfg["split"]
    manifest = read_json(split["manifest_path"])
    test_path = resolve_path(split["test_path"])
    if (
        manifest["train_sha256"] != training_sha
        or file_sha256(test_path) != manifest["test_sha256"]
    ):
        raise ValueError(f"Split incompatível com o modelo congelado: {test_path}")
    row_ids = pd.read_csv(test_path, usecols=["row_id"])["row_id"]
    if not row_ids.is_unique or len(row_ids) != manifest["n_rows"]["test"]:
        raise ValueError(f"Identificadores inválidos no TEST: {test_path}")
    X, y = load_split("test", cfg)
    return X, y, row_ids, manifest["test_sha256"]


def save_figure(fig, directory, filename):
    fig.savefig(directory / filename, dpi=120, bbox_inches="tight")
    plt.close(fig)


def run_evaluation() -> pd.DataFrame:
    if OUTPUT.exists():
        return pd.read_csv(OUTPUT / "metrics/test_model_metrics.csv")

    full_cfg, reduced_cfg = load_config("data_full"), load_config("data")
    classes = class_names_from_cfg(full_cfg)
    critical = full_cfg["target"]["critical_class"]
    bundle = PROJECT_ROOT / "best_model"
    full = read_json(bundle / "metadata.json")
    reduced = read_json(
        REPORTS_DIR / "runs" / REDUCED_PARENT / "experiments/gridsearch_cv_manifest.json"
    )
    binary = bundle / full["artifact"]
    if full["mlflow"]["version_uri"] != FULL_VERSION or file_sha256(binary) != full["sha256"]:
        raise ValueError("O pacote Joblib diverge do XGBoost versão 1.")
    if reduced["models"]["xgboost"] != REDUCED_RUN or reduced["test_set_used"]:
        raise ValueError("O run congelado da versão reduzida diverge do manifesto.")
    with tracking_session(load_config("experiment")["mlflow"], backend="local"):
        reduced_uri = f"runs:/{REDUCED_RUN}/model"
        artifact = Path(
            mlflow.artifacts.download_artifacts(artifact_uri=f"{reduced_uri}/model.pkl")
        )
        if file_sha256(artifact) != REDUCED_SHA256:
            raise ValueError("O artefato da versão reduzida diverge do run congelado.")
        reduced_model = mlflow.sklearn.load_model(reduced_uri)

    plans = [
        (
            "full",
            "XGBoost · 21 features",
            full_cfg,
            full,
            joblib.load(binary),
            FULL_VERSION,
            full["sha256"],
        ),
        (
            "reduced",
            "XGBoost · 4 features",
            reduced_cfg,
            reduced,
            reduced_model,
            reduced_uri,
            REDUCED_SHA256,
        ),
    ]
    evaluations = [
        load_frozen_test(cfg, source["dataset"]["train_sha256"]) for _, _, cfg, source, *_ in plans
    ]
    reference_y, reference_ids = evaluations[0][1:3]
    if not reference_y.equals(evaluations[1][1]) or not reference_ids.equals(evaluations[1][2]):
        raise ValueError("As versões precisam usar os mesmos exames e rótulos no TEST.")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=OUTPUT.parent) as temp:
        staged = Path(temp) / OUTPUT.name
        metrics, figures = staged / "metrics", staged / "figures"
        metrics.mkdir(parents=True)
        figures.mkdir()
        rows, class_rows, provenance = [], [], []
        fig, (pr, roc) = plt.subplots(1, 2, figsize=(10, 4.2), layout="constrained")
        pathological = reference_y == critical
        pr.axhline(pathological.mean(), color="gray", ls=":", label="Prevalência no TEST")
        roc.plot([0, 1], [0, 1], color="gray", ls=":")
        for (key, label, _, _, model, uri, sha), (X, y, _, split_sha) in zip(
            plans, evaluations, strict=True
        ):
            if list(model.feature_names_in_) != X.columns.tolist() or list(model.classes_) != list(
                classes
            ):
                raise ValueError(f"Schema do modelo incompatível: {label}")
            predicted = model.predict(X)
            probability = model.predict_proba(X)
            report = classification_report_dict(y, predicted, classes)
            result = compute_metrics(y, predicted, classes)
            result.update(compare_probabilities(y, probability, classes, prefix="test"))
            flagged = predicted == critical
            result.update(
                tp_patologico=int((pathological & flagged).sum()),
                fn_patologico=int((pathological & ~flagged).sum()),
                fp_patologico=int((~pathological & flagged).sum()),
                tn_patologico=int((~pathological & ~flagged).sum()),
            )
            rows.append({"model": label, "n_features": X.shape[1], "n_test": len(y), **result})
            for name in classes.values():
                class_rows.append({"model": label, "classe": name, **report[name]})
            provenance.append(
                {
                    "model": label,
                    "model_uri": uri,
                    "model_sha256": sha,
                    "test_sha256": split_sha,
                    "features": X.columns.tolist(),
                }
            )
            scores = probability[:, list(classes).index(critical)]
            precision, recall, _ = precision_recall_curve(pathological, scores)
            fpr, tpr, _ = roc_curve(pathological, scores)
            pr.plot(recall, precision, label=f"{label} · AP {result['test_ap_patologico']:.3f}")
            roc.plot(fpr, tpr, label=f"{label} · AUC {result['test_roc_auc_patologico']:.3f}")
            save_figure(
                plot_confusion_matrices(y, predicted, classes, f"{label} · TEST"),
                figures,
                f"confusion_{key}.png",
            )

        pr.set(title="Precision–Recall", xlabel="Recall", ylabel="Precision")
        roc.set(title="ROC", xlabel="Taxa de falsos positivos", ylabel="Recall")
        for ax in (pr, roc):
            ax.set(xlim=(0, 1), ylim=(0, 1.02))
            ax.legend(fontsize=8)
        fig.suptitle("Classe patológica · TEST")
        save_figure(fig, figures, "critical_pr_roc.png")
        results = pd.DataFrame(rows)
        save_figure(
            plot_metric_comparison(
                results, std_suffix=None, title="Modelos congelados", source_label="TEST"
            ),
            figures,
            "macro_comparison.png",
        )
        results.to_csv(metrics / "test_model_metrics.csv", index=False)
        pd.DataFrame(class_rows).to_csv(metrics / "test_class_metrics.csv", index=False)
        manifest = {
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
            "split": "TEST",
            "n_test": len(reference_y),
            "n_pathological": int(pathological.sum()),
            "same_exams": True,
            "decision_rule": "argmax",
            "models": provenance,
        }
        (staged / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        staged.rename(OUTPUT)
    return results


if __name__ == "__main__":
    columns = [
        "model",
        "accuracy",
        "f1_macro",
        "recall_patologico",
        "precision_patologico",
        "test_ap_patologico",
    ]
    print(run_evaluation()[columns].to_string(index=False))
