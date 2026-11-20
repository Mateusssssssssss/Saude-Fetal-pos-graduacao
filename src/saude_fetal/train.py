"""GridSearchCV aninhada para Random Forest e XGBoost no TRAIN, com MLflow."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # backend sem janela (scripts/servidores)

import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402

from saude_fetal.comparison import (  # noqa: E402
    MODEL_LABELS,
    compare_probabilities,
    plot_comparison_diagnostics,
    plot_external_comparison,
)
from saude_fetal.config import REPORTS_DIR, ensure_dirs, load_config  # noqa: E402
from saude_fetal.data import file_sha256, load_split, make_split  # noqa: E402
from saude_fetal.evaluation import (  # noqa: E402
    build_scorers,
    class_names_from_cfg,
    classification_report_dict,
    compute_metrics,
    plot_confusion_matrices,
)
from saude_fetal.model import build_pipeline  # noqa: E402
from saude_fetal.tracking import (  # noqa: E402
    environment_tags,
    flatten_dict,
    setup_mlflow,
    tracking_session,
)
from saude_fetal.validation import (
    evaluate_folds,
    feature_groups,
    make_cv,
    make_grid_search,
    summarize_folds,
)  # noqa: E402

logger = logging.getLogger(__name__)


def _synthetic_input_example(X: pd.DataFrame) -> pd.DataFrame:
    """Exemplo fixo no domínio das features numéricas, sem derivar valores do TRAIN."""
    return pd.DataFrame(0.0, columns=X.columns, index=[0])


def _grid_results(search) -> pd.DataFrame:
    results = pd.DataFrame(search.cv_results_)
    results["params"] = results["params"].map(
        lambda params: json.dumps({key.rsplit("__", 1)[-1]: value for key, value in params.items()})
    )
    return results.drop(columns=results.filter(regex="^param_").columns)


def train_one_model(
    name: str,
    model_cfg: dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    data_cfg: dict[str, Any],
    exp_cfg: dict[str, Any],
    output_dirs: dict[str, Path],
    log_model: bool = True,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Treina/valida um modelo dentro de um run MLflow (já ativo como pai)."""
    class_names = class_names_from_cfg(data_cfg)
    seed = int(exp_cfg["experiment"]["random_state"])
    prefix = exp_cfg["experiment"]["prefix"]
    features = list(data_cfg["features"])

    pipeline = build_pipeline(model_cfg, features, random_state=seed)
    validation_cfg = exp_cfg["validation"]
    groups = feature_groups(X, data_cfg)
    cv = make_cv(validation_cfg, random_state=seed, grouped=groups is not None)
    scorers = build_scorers(class_names)
    search = make_grid_search(
        pipeline, model_cfg["grid"], validation_cfg, seed, scorers, grouped=groups is not None
    )

    parent = mlflow.active_run()
    with mlflow.start_run(
        run_name=name, nested=True, experiment_id=parent.info.experiment_id if parent else None
    ) as run:
        mlflow.set_tags(
            {
                "model_name": name,
                "estimator": model_cfg["estimator"],
                "stage": "nested_cv_train_only",
                "test_set_used": "false",
                "metric_sources": "cv_train_* = treino dos folds; demais cv_* = validação dos folds; "
                "oof_* = OOF no TRAIN; "
                "train_refit_* = ajuste e predição no TRAIN completo (otimista)",
            }
        )
        mlflow.log_params(flatten_dict({"model": model_cfg}))
        mlflow.log_params(
            {
                "cv.n_splits": cv.get_n_splits(),
                "cv.inner_n_splits": search.cv.get_n_splits(),
                "cv.splitter": type(cv).__name__,
                "cv.inner_splitter": type(search.cv).__name__,
                "cv.shuffle": cv.shuffle,
                "cv.random_state": cv.random_state,
                "primary_metric": exp_cfg["validation"]["primary_metric"],
                "decision_rule": "argmax(predict_proba) - padrão do classificador",
            }
        )

        # A busca interna de cada fold só recebe seu bloco de treino externo.
        folds, searches, validation_indices = evaluate_folds(
            search, X, y, cv, scorers, groups=groups
        )
        oof_pred = np.empty_like(y.to_numpy())
        oof_proba = np.empty((len(y), len(class_names)))
        outer_grids = []
        folds["best_params"] = ""
        for fold_index, (fitted_search, indices) in enumerate(
            zip(searches, validation_indices, strict=True)
        ):
            fold_grid = _grid_results(fitted_search)
            fold_grid.insert(0, "outer_fold", fold_index + 1)
            outer_grids.append(fold_grid)
            folds.loc[fold_index, "best_params"] = fold_grid.loc[
                fitted_search.best_index_, "params"
            ]
            valid = X.iloc[indices]
            oof_pred[indices] = fitted_search.predict(valid)
            probabilities = pd.DataFrame(
                fitted_search.predict_proba(valid), columns=fitted_search.classes_
            )
            oof_proba[indices] = probabilities[list(class_names)].to_numpy()

        cv_summary = summarize_folds(folds)
        oof_metrics = {f"oof_{k}": v for k, v in compute_metrics(y, oof_pred, class_names).items()}
        oof_metrics.update(compare_probabilities(y, oof_proba, class_names))
        mlflow.log_metrics(cv_summary)
        mlflow.log_metrics(oof_metrics)
        for row in folds.select_dtypes("number").to_dict("records"):
            step = int(row.pop("fold"))
            mlflow.log_metrics(
                {f"fold_{metric}": value for metric, value in row.items()}, step=step
            )

        fold_path = output_dirs["metrics"] / f"{prefix}_{name}_fold_metrics.csv"
        folds.to_csv(fold_path, index=False)
        mlflow.log_artifact(str(fold_path), artifact_path="metrics")
        outer_grid_path = output_dirs["metrics"] / f"{prefix}_{name}_outer_grid_results.csv"
        pd.concat(outer_grids, ignore_index=True).to_csv(outer_grid_path, index=False)
        mlflow.log_artifact(str(outer_grid_path), artifact_path="metrics")

        report = classification_report_dict(y, oof_pred, class_names)
        report_path = output_dirs["metrics"] / f"{prefix}_{name}_oof_classification_report.json"
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        mlflow.log_artifact(str(report_path), artifact_path="metrics")

        fig = plot_confusion_matrices(
            y, oof_pred, class_names, f"{MODEL_LABELS.get(name, name)} — OOF no TRAIN"
        )
        fig_path = output_dirs["figures"] / f"{prefix}_{name}_confusion_oof.png"
        fig.savefig(fig_path, dpi=120, bbox_inches="tight")
        mlflow.log_artifact(str(fig_path), artifact_path="figures")
        plt.close(fig)

        # Busca independente para escolher os parâmetros do pipeline de entrega.
        search.fit(X, y, **({"groups": groups} if groups is not None else {}))
        full_grid = _grid_results(search)
        primary_metric = validation_cfg["primary_metric"]
        full_grid = full_grid.sort_values(f"rank_test_{primary_metric}", kind="stable")
        best_params = full_grid.loc[search.best_index_, "params"]
        mlflow.log_params(flatten_dict({"selected": json.loads(best_params)}))
        mlflow.log_metric(f"full_train_grid_inner_{primary_metric}", search.best_score_)
        grid_path = output_dirs["metrics"] / f"{prefix}_{name}_grid_results.csv"
        full_grid.to_csv(grid_path, index=False)
        mlflow.log_artifact(str(grid_path), artifact_path="metrics")
        final_pipeline = search.best_estimator_
        train_pred = final_pipeline.predict(X)
        train_metrics = {
            f"train_refit_{k}": v for k, v in compute_metrics(y, train_pred, class_names).items()
        }
        mlflow.log_metrics(train_metrics)

        if log_model:
            example = _synthetic_input_example(X)
            signature = infer_signature(example, final_pipeline.predict(example))
            mlflow.sklearn.log_model(
                final_pipeline,
                artifact_path="model",
                code_paths=(
                    [str(Path(__file__).parent)] if model_cfg["estimator"] == "xgboost" else None
                ),
                signature=signature,
                input_example=example,
            )

        summary = {
            "model": name,
            "run_id": run.info.run_id,
            "best_params": best_params,
            f"full_train_grid_inner_{primary_metric}": search.best_score_,
            **cv_summary,
            **oof_metrics,
            **train_metrics,
        }
        return summary, folds, full_grid, oof_proba


def train_models(
    X: pd.DataFrame,
    y: pd.Series,
    data_cfg: dict[str, Any],
    model_cfg: dict[str, Any],
    exp_cfg: dict[str, Any],
    model_names: list[str] | None = None,
    tracking_uri: str | None = None,
    output_dirs: dict[str, Path] | None = None,
    dataset_info: dict[str, Any] | None = None,
    tracking_backend: str | None = None,
) -> pd.DataFrame:
    """Treina os modelos selecionados e devolve a tabela-resumo (uma linha por modelo)."""
    all_models = model_cfg["models"]
    names = model_names or list(all_models)
    unknown = [n for n in names if n not in all_models]
    if unknown:
        raise KeyError(f"Modelos não definidos em model.yaml: {unknown}")

    prefix = exp_cfg["experiment"]["prefix"]
    mlflow_cfg = exp_cfg["mlflow"]
    started = datetime.now(timezone.utc)

    with (
        tracking_session(mlflow_cfg, tracking_backend, tracking_uri) as backend,
        mlflow.start_run(
            run_name=f"{prefix}_{started:%Y%m%d_%H%M%S}", experiment_id=setup_mlflow(mlflow_cfg)
        ) as parent,
    ):
        if output_dirs is None:
            report_dir = REPORTS_DIR / "runs" / parent.info.run_id
            output_dirs = {key: report_dir / key for key in ("metrics", "figures", "experiments")}
            logger.info("Relatórios da execução: %s", report_dir)
        for directory in output_dirs.values():
            directory.mkdir(parents=True, exist_ok=True)
        versions = environment_tags()
        if any(all_models[name]["estimator"] == "xgboost" for name in names):
            from importlib.metadata import version

            versions["xgboost_version"] = version("xgboost")
        mlflow.set_tags({"stage": "nested_cv_train_only", "test_set_used": "false", **versions})
        mlflow.log_params(
            {
                "experiment.prefix": prefix,
                "tracking.backend": backend,
                "models": ",".join(names),
                "n_train": len(X),
                "features": ",".join(X.columns),
                "target": data_cfg["target"]["column"],
                "critical_class": data_cfg["target"]["critical_class"],
                **{
                    f"split.{k}": v for k, v in data_cfg["split"].items() if not k.endswith("_path")
                },
            }
        )
        if dataset_info:
            mlflow.set_tags({f"data.{k}": str(v) for k, v in dataset_info.items()})
        train_df = X.assign(**{data_cfg["target"]["column"]: y.to_numpy()})
        mlflow.log_input(
            mlflow.data.from_pandas(
                train_df,
                source=str(data_cfg["split"]["train_path"]),
                name="train",
                targets=data_cfg["target"]["column"],
            ),
            context="training",
        )
        mlflow.log_dict(
            {"data": data_cfg, "model": model_cfg, "experiment": exp_cfg}, "configs.json"
        )

        rows = []
        folds_by_model = {}
        grids_by_model = {}
        probabilities_by_model = {}
        for name in names:
            logger.info("Treinando %s ...", name)
            summary, folds, grid, probabilities = train_one_model(
                name,
                all_models[name],
                X,
                y,
                data_cfg,
                exp_cfg,
                output_dirs,
                log_model=bool(mlflow_cfg.get("log_models", True)),
            )
            rows.append(summary)
            folds_by_model[name] = folds
            grids_by_model[name] = grid
            probabilities_by_model[name] = probabilities

        results = pd.DataFrame(rows)
        primary = f"cv_{exp_cfg['validation']['primary_metric']}_mean"
        results = results.sort_values(primary, ascending=False, kind="stable").reset_index(
            drop=True
        )
        results_path = output_dirs["metrics"] / f"{prefix}_results.csv"
        results.to_csv(results_path, index=False)
        mlflow.log_artifact(str(results_path), artifact_path="metrics")

        # Gráfico de comparação: médias de CV (validação dos folds) ± desvio-padrão.
        fig = plot_external_comparison(results)
        fig_path = output_dirs["figures"] / f"{prefix}_model_comparison_f1_recall_precision.png"
        fig.savefig(fig_path, dpi=120, bbox_inches="tight")
        mlflow.log_artifact(str(fig_path), artifact_path="figures")
        plt.close(fig)
        for suffix, figure in plot_comparison_diagnostics(
            results,
            folds_by_model,
            grids_by_model,
            probabilities_by_model,
            y,
            class_names_from_cfg(data_cfg),
            data_cfg["target"]["critical_class"],
            exp_cfg["validation"]["primary_metric"],
        ).items():
            path = output_dirs["figures"] / f"{prefix}_{suffix}.png"
            figure.savefig(path, dpi=120, bbox_inches="tight")
            plt.close(figure)
            mlflow.log_artifact(str(path), artifact_path="figures")

        manifest = {
            "created_at": started.isoformat(),
            "parent_run_id": parent.info.run_id,
            "experiment_id": parent.info.experiment_id,
            "experiment_name": mlflow_cfg["experiment_name"],
            "tracking_backend": backend,
            "models": {r["model"]: r["run_id"] for r in rows},
            "best_params": {r["model"]: json.loads(r["best_params"]) for r in rows},
            "primary_metric": primary,
            "validation": exp_cfg["validation"],
            "group_duplicate_features": data_cfg["split"].get("group_duplicate_features", False),
            "features": list(X.columns),
            "test_set_used": False,
            "dataset": dataset_info or {},
            "environment": versions,
        }
        manifest_path = output_dirs["experiments"] / f"{prefix}_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        mlflow.log_artifact(str(manifest_path), artifact_path="experiments")

    return results


def run_training(
    model_names: list[str] | None = None,
    tracking_uri: str | None = None,
    *,
    tracking_backend: str | None = None,
    workflow: str = "trees",
    data_config: str = "data",
) -> pd.DataFrame:
    """Ponto de entrada usado pelo CLI e pelo notebook de treino."""
    ensure_dirs()
    data_cfg = load_config(data_config)
    model_cfg = load_config("model")
    exp_cfg = load_config("experiment")

    split_manifest = make_split(data_cfg)  # não recria se já existir
    X, y = load_split("train", data_cfg)
    dataset_info = {
        "name": data_cfg["dataset"]["name"],
        "raw_sha256": file_sha256(data_cfg["dataset"]["raw_path"]),
        "train_sha256": file_sha256(data_cfg["split"]["train_path"]),
        "split_created_at": split_manifest.get("created_at", "desconhecido"),
        "excluded_train_feature_overlap": split_manifest.get("excluded_train_feature_overlap", 0),
    }
    if workflow not in {"trees", "cnn", "all"}:
        raise ValueError(f"Fluxo desconhecido: {workflow}")
    if workflow == "cnn" and model_names is not None:
        raise ValueError("--models seleciona apenas os modelos de árvores.")
    results = []
    if workflow in {"trees", "all"}:
        results.append(
            train_models(
                X,
                y,
                data_cfg,
                model_cfg,
                exp_cfg,
                model_names=model_names,
                tracking_uri=tracking_uri,
                dataset_info=dataset_info,
                tracking_backend=tracking_backend,
            )
        )
    if workflow in {"cnn", "all"}:
        from saude_fetal.train_dl import train_cnn

        results.append(
            train_cnn(X, y, data_cfg, exp_cfg, dataset_info, tracking_backend, tracking_uri)
        )
    return results[0] if len(results) == 1 else pd.concat(results, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Treino no TRAIN com tracking no MLflow.")
    parser.add_argument("--workflow", choices=("trees", "cnn", "all"), default="trees")
    parser.add_argument("--data-config", choices=("data", "data_full"), default="data")
    parser.add_argument("--models", nargs="*", help="Subconjunto de modelos de model.yaml.")
    parser.add_argument(
        "--tracking-backend", choices=("local", "dagshub"), help="Destino do MLflow."
    )
    parser.add_argument("--tracking-uri", help="Sobrescreve o tracking URI do backend local.")
    args = parser.parse_args()
    if args.tracking_backend == "dagshub" and args.tracking_uri is not None:
        parser.error("Use --tracking-uri com --tracking-backend local.")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if args.workflow == "cnn" and args.models is not None:
        parser.error("--models seleciona apenas os modelos de árvores.")
    results = run_training(
        args.models,
        args.tracking_uri,
        tracking_backend=args.tracking_backend,
        workflow=args.workflow,
        data_config=args.data_config,
    )
    cols = ["model", "cv_f1_macro_mean", "cv_f1_macro_std", "cv_recall_patologico_mean", "run_id"]
    print(results[[c for c in cols if c in results]].to_string(index=False))


if __name__ == "__main__":
    main()
