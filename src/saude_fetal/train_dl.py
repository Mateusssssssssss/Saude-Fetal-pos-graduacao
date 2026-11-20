"""CNN tabular em TensorFlow/Keras com CV estratificada e refit no TRAIN."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from time import perf_counter

import keras
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import tensorflow as tf
from mlflow.models import infer_signature
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_recall_curve, roc_curve

from saude_fetal.comparison import compare_probabilities
from saude_fetal.config import REPORTS_DIR
from saude_fetal.evaluation import (
    class_names_from_cfg,
    classification_report_dict,
    compute_metrics,
    plot_confusion_matrices,
    plot_metric_comparison,
)
from saude_fetal.tracking import environment_tags, setup_mlflow, tracking_session
from saude_fetal.validation import feature_groups, make_cv, summarize_folds

logger = logging.getLogger(__name__)
PREFIX = "cnn_cv"


def build_cnn(X: np.ndarray, n_classes: int, config: dict) -> keras.Model:
    """Incorpora imputação e normalização aprendidas no bloco de ajuste."""
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    imputed = imputer.fit_transform(X)
    inputs = keras.Input(shape=(X.shape[1],), dtype="float32", name="features")
    medians = keras.ops.convert_to_tensor(imputer.statistics_, dtype="float32")
    filled = keras.ops.where(keras.ops.isnan(inputs), medians, inputs)
    normalized = keras.layers.Normalization(mean=imputed.mean(axis=0), variance=imputed.var(axis=0))
    values = normalized(filled)
    values = keras.layers.Reshape((X.shape[1], 1))(values)
    values = keras.layers.Conv1D(config["filters"], config["kernel_size"], activation="relu")(
        values
    )
    values = keras.layers.Flatten()(values)
    values = keras.layers.Dense(config["dense_units"], activation="relu")(values)
    outputs = keras.layers.Dense(n_classes, activation="softmax", name="class_scores")(values)
    model = keras.Model(inputs, outputs, name="cnn_tabular")
    model.compile(
        optimizer=keras.optimizers.Adam(config["learning_rate"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def _dataset(X, y, batch_size, seed=None):
    data = tf.data.Dataset.from_tensor_slices((X, y))
    if seed is not None:
        data = data.shuffle(len(y), seed=seed)
    options = tf.data.Options()
    options.threading.private_threadpool_size = 1
    return data.batch(batch_size).with_options(options)


def _fit_cnn(X, y, n_classes, config, seed, validation=None):
    keras.backend.clear_session()
    keras.utils.set_random_seed(seed)
    model = build_cnn(X, n_classes, config)
    history = model.fit(
        _dataset(X, y, config["batch_size"], seed),
        validation_data=(
            _dataset(*validation, config["batch_size"]) if validation is not None else None
        ),
        epochs=config["epochs"],
        shuffle=False,  # O embaralhamento ocorre no tf.data.Dataset de treino.
        verbose=0,
    )
    return model, pd.DataFrame(history.history).assign(epoch=range(1, config["epochs"] + 1))


def _figures(y, oof_pred, probabilities, history, summary, classes, critical):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for ax, metric, title in zip(axes, ("loss", "accuracy"), ("Perda", "Acurácia"), strict=True):
        for column, label in ((metric, "Treino"), (f"val_{metric}", "Validação")):
            per_epoch = history.groupby("epoch")[column]
            mean, std = per_epoch.mean(), per_epoch.std()
            ax.plot(mean.index, mean, label=label)
            ax.fill_between(mean.index, mean - std, mean + std, alpha=0.15)
        ax.set(title=title, xlabel="Época")
        ax.legend()
    fig.suptitle("CNN 1D · média e desvio entre folds")
    yield "learning_curves", fig

    bars = {"model": "CNN 1D"}
    for metric in ("f1_macro", "recall_macro", "precision_macro"):
        bars[metric] = summary[f"cv_{metric}_mean"]
        bars[f"{metric}_std"] = summary[f"cv_{metric}_std"]
    yield "macro_metrics", plot_metric_comparison(pd.DataFrame([bars]), title="CNN 1D")
    yield "confusion_oof", plot_confusion_matrices(y, oof_pred, classes, "CNN 1D · OOF no TRAIN")

    target = y == critical
    scores = probabilities[:, list(classes).index(critical)]
    name = classes[critical].lower()
    precision, recall, _ = precision_recall_curve(target, scores)
    fpr, tpr, _ = roc_curve(target, scores)
    fig, (pr, roc) = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    pr.plot(recall, precision, label=f"AP {summary[f'oof_ap_{name}']:.3f}")
    pr.axhline(target.mean(), color="gray", ls=":", label=f"Prevalência {target.mean():.3f}")
    pr.set(
        xlabel="Recall", ylabel="Precision", title="Precision–Recall", xlim=(0, 1), ylim=(0, 1.02)
    )
    roc.plot(fpr, tpr, label=f"AUC {summary[f'oof_roc_auc_{name}']:.3f}")
    roc.plot([0, 1], [0, 1], color="gray", ls=":")
    roc.set(
        xlabel="Taxa de falsos positivos", ylabel="Recall", title="ROC", xlim=(0, 1), ylim=(0, 1.02)
    )
    pr.legend()
    roc.legend()
    fig.suptitle("Classe patológica contra as demais · OOF no TRAIN")
    yield "critical_pr_roc", fig


def train_cnn(
    X, y, data_cfg, exp_cfg, dataset_info, backend=None, tracking_uri=None
) -> pd.DataFrame:
    """Ajusta somente no TRAIN, registra métricas e publica o modelo completo no MLflow."""
    if keras.backend.backend() != "tensorflow":
        raise RuntimeError("Configure KERAS_BACKEND=tensorflow para treinar a CNN.")
    config = exp_cfg["cnn"]
    seed = exp_cfg["experiment"]["random_state"]
    classes = class_names_from_cfg(data_cfg)
    labels = np.array(list(classes))
    features = data_cfg["features"]
    values = X[features].to_numpy(dtype=np.float32)
    encoded = y.map({label: index for index, label in enumerate(labels)}).to_numpy(dtype=np.int32)
    groups = feature_groups(X, data_cfg)
    cv = make_cv(exp_cfg["validation"], seed, grouped=groups is not None)
    mlflow_cfg = exp_cfg["mlflow"]
    started = datetime.now(timezone.utc)

    with (
        tracking_session(mlflow_cfg, backend, tracking_uri) as destination,
        mlflow.start_run(
            experiment_id=setup_mlflow(mlflow_cfg), run_name=f"{PREFIX}_{started:%Y%m%d_%H%M%S}"
        ) as run,
    ):
        root = REPORTS_DIR / "runs" / run.info.run_id
        metrics, figures, experiments = (
            root / name for name in ("metrics", "figures", "experiments")
        )
        for directory in (metrics, figures, experiments):
            directory.mkdir(parents=True, exist_ok=True)
        versions = {**environment_tags(), "tensorflow": tf.__version__, "keras": keras.__version__}
        mlflow.set_tags({"test_set_used": "false", "stage": "fixed_cnn_cv_train_only", **versions})
        mlflow.log_params(
            {**config, "random_state": seed, "n_splits": cv.get_n_splits(), "decision": "argmax"}
        )
        mlflow.log_dict({"features": features, "class_order": labels.tolist()}, "input_schema.json")
        mlflow.log_dict({"data": data_cfg, "experiment": exp_cfg}, "configs.json")

        probabilities = np.empty((len(y), len(labels)))
        folds, histories = [], []
        for fold, (train_idx, val_idx) in enumerate(cv.split(values, y, groups), start=1):
            logger.info("CNN · fold %s/%s · %s épocas", fold, cv.get_n_splits(), config["epochs"])
            start = perf_counter()
            model, history = _fit_cnn(
                values[train_idx],
                encoded[train_idx],
                len(labels),
                config,
                seed + fold,
                validation=(values[val_idx], encoded[val_idx]),
            )
            probabilities[val_idx] = model(values[val_idx], training=False).numpy()
            train_pred = labels[model(values[train_idx], training=False).numpy().argmax(axis=1)]
            val_pred = labels[probabilities[val_idx].argmax(axis=1)]
            row = {"fold": fold, "fit_time_s": perf_counter() - start}
            row.update(
                {
                    f"train_{k}": v
                    for k, v in compute_metrics(y.iloc[train_idx], train_pred, classes).items()
                }
            )
            row.update(
                {
                    f"val_{k}": v
                    for k, v in compute_metrics(y.iloc[val_idx], val_pred, classes).items()
                }
            )
            folds.append(row)
            histories.append(history.assign(fold=fold))
        folds = pd.DataFrame(folds)
        history = pd.concat(histories, ignore_index=True)
        oof_pred = labels[probabilities.argmax(axis=1)]
        summary = summarize_folds(folds)
        summary.update({f"oof_{k}": v for k, v in compute_metrics(y, oof_pred, classes).items()})
        summary.update(compare_probabilities(y, probabilities, classes))
        mlflow.log_metrics(summary)

        logger.info("CNN · refit no TRAIN completo · %s épocas", config["epochs"])
        final_model, refit_history = _fit_cnn(values, encoded, len(labels), config, seed)
        if mlflow_cfg.get("log_models", True):
            example = np.zeros((1, len(features)), dtype=np.float32)
            expected = final_model(example, training=False).numpy()
            mlflow.log_dict({"inputs": example.tolist()}, "request_example.json")
            # O flavor nativo preserva a rede, a imputação e a normalização.
            model_info = mlflow.keras.log_model(
                final_model,
                name="model",
                signature=infer_signature(example, expected),
                metadata={"features": features, "class_order": labels.tolist()},
                pip_requirements=[
                    f"mlflow=={mlflow.__version__}",
                    f"keras=={keras.__version__}",
                    f"tensorflow=={tf.__version__}",
                    f"numpy=={np.__version__}",
                ],
            )
            loaded = mlflow.keras.load_model(model_info.model_uri)
            probe = np.vstack((example, np.full_like(example, np.nan)))
            np.testing.assert_allclose(
                loaded(probe, training=False).numpy(),
                final_model(probe, training=False).numpy(),
                rtol=1e-6,
                atol=1e-7,
            )

        result = pd.DataFrame([{"model": "cnn", "run_id": run.info.run_id, **summary}])
        for suffix, table in (
            ("results", result),
            ("fold_metrics", folds),
            ("epoch_history", history),
            ("refit_history", refit_history),
        ):
            path = metrics / f"{PREFIX}_{suffix}.csv"
            table.to_csv(path, index=False)
            mlflow.log_artifact(str(path), artifact_path="metrics")
        report = classification_report_dict(y, oof_pred, classes)
        mlflow.log_dict(report, f"metrics/{PREFIX}_oof_classification_report.json")
        (metrics / f"{PREFIX}_oof_classification_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        for name, figure in _figures(
            y,
            oof_pred,
            probabilities,
            history,
            summary,
            classes,
            data_cfg["target"]["critical_class"],
        ):
            path = figures / f"{PREFIX}_{name}.png"
            figure.savefig(path, dpi=120, bbox_inches="tight")
            plt.close(figure)
            mlflow.log_artifact(str(path), artifact_path="figures")
        manifest = {
            "created_at": started.isoformat(),
            "run_id": run.info.run_id,
            "experiment_id": run.info.experiment_id,
            "tracking_backend": destination,
            "test_set_used": False,
            "dataset": dataset_info,
            "cv_splitter": type(cv).__name__,
            "environment": versions,
            "features": features,
            "class_order": labels.tolist(),
            "cnn": config,
        }
        path = experiments / f"{PREFIX}_manifest.json"
        path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        mlflow.log_artifact(str(path), artifact_path="experiments")
    return result
