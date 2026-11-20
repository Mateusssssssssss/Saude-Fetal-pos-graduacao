import json

import mlflow
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict

from saude_fetal.evaluation import build_scorers
from saude_fetal.model import build_pipeline
from saude_fetal.predict import load_model
from saude_fetal.train import train_models
from saude_fetal.validation import (
    evaluate_folds,
    feature_groups,
    make_cv,
    make_grid_search,
    run_cv,
    summarize_folds,
)

from .conftest import FEATURES, TARGET

CLASSES = {1: "Normal", 2: "Suspeito", 3: "Patologico"}


def test_oof_predictions_cover_every_sample_with_one_fit_per_fold(synthetic_df, monkeypatch):
    X, y = synthetic_df[FEATURES], synthetic_df[TARGET].astype(int)
    cv = make_cv({"n_splits": 3, "shuffle": True}, random_state=0)
    pipe = build_pipeline(
        {"estimator": "random_forest", "params": {"n_estimators": 5}}, FEATURES, 0
    )
    expected = cross_val_predict(pipe, X, y, cv=cv)
    fit_calls = []
    original_fit = type(pipe).fit

    def count_fit(self, X_fold, y_fold, **kwargs):
        fit_calls.append(len(X_fold))
        return original_fit(self, X_fold, y_fold, **kwargs)

    monkeypatch.setattr(type(pipe), "fit", count_fit)
    folds, oof = run_cv(pipe, X, y, cv, build_scorers(CLASSES))
    assert len(folds) == 3
    assert len(fit_calls) == 3
    np.testing.assert_array_equal(oof, expected)
    summary = summarize_folds(folds)
    assert "cv_f1_macro_mean" in summary and "cv_f1_macro_std" in summary


def test_cv_folds_are_stratified(synthetic_df):
    y = synthetic_df[TARGET].astype(int)
    cv = make_cv({"n_splits": 5, "shuffle": True}, random_state=0)
    global_prev = (y == 3).mean()
    for _, va in cv.split(synthetic_df[FEATURES], y):
        assert abs((y.iloc[va] == 3).mean() - global_prev) < 0.05


def test_nested_cv_routes_only_training_groups_to_each_inner_search(
    synthetic_df, data_cfg, monkeypatch
):
    X, y = synthetic_df[FEATURES], synthetic_df[TARGET].astype(int)
    data_cfg["split"]["group_duplicate_features"] = True
    groups = feature_groups(X, data_cfg)
    assert len(np.unique(groups)) < len(X)
    cfg = {"n_splits": 3, "inner_n_splits": 2, "shuffle": True, "primary_metric": "f1_macro"}
    outer = make_cv(cfg, 0, grouped=True)
    original_split = StratifiedGroupKFold.split
    expected_inner_inputs = [groups[train] for train, _ in original_split(outer, X, y, groups)]
    recorded_inner_inputs = []

    def checked_split(self, X, y=None, groups=None):
        if self.n_splits == 2:
            recorded_inner_inputs.append(np.asarray(groups))
        for train, valid in original_split(self, X, y, groups):
            assert not set(groups[train]) & set(groups[valid])
            yield train, valid

    monkeypatch.setattr(StratifiedGroupKFold, "split", checked_split)
    scorers = build_scorers(CLASSES)
    pipeline = build_pipeline(
        {"estimator": "random_forest", "params": {"n_estimators": 2}}, FEATURES, 0
    )
    search = make_grid_search(pipeline, {"max_depth": [2]}, cfg, 0, scorers, grouped=True)
    folds, _, _ = evaluate_folds(search, X, y, outer, scorers, groups=groups)
    assert len(folds) == 3
    for expected, received in zip(expected_inner_inputs, recorded_inner_inputs, strict=True):
        np.testing.assert_array_equal(received, expected)


def test_train_models_logs_to_mlflow_without_test(tmp_path, synthetic_df, data_cfg, monkeypatch):
    """Fluxo completo com MLflow em diretório temporário; só recebe o TRAIN."""
    X, y = synthetic_df[FEATURES], synthetic_df[TARGET].astype(int)
    model_cfg = {
        "models": {
            "random_forest": {
                "estimator": "random_forest",
                "params": {"n_estimators": 5},
                "grid": {"max_depth": [2, 4]},
            },
            "xgboost": {
                "estimator": "xgboost",
                "params": {"objective": "multi:softprob", "n_estimators": 5, "n_jobs": 1},
                "grid": {"max_depth": [2, 4]},
            },
        }
    }
    exp_cfg = {
        "experiment": {"prefix": "pytest", "random_state": 0},
        "validation": {
            "n_splits": 3,
            "inner_n_splits": 2,
            "shuffle": True,
            "primary_metric": "f1_macro",
        },
        "mlflow": {
            "tracking_uri": (tmp_path / "mlruns").as_uri(),
            "artifact_dir": str(tmp_path / "artifacts"),
            "experiment_name": "pytest-exp",
            "log_models": True,
        },
    }
    out_dirs = {k: tmp_path / k for k in ("metrics", "figures", "experiments")}
    results = train_models(
        X,
        y,
        data_cfg,
        model_cfg,
        exp_cfg,
        output_dirs=out_dirs,
        tracking_uri=exp_cfg["mlflow"]["tracking_uri"],
    )

    assert set(results["model"]) == {"random_forest", "xgboost"}
    assert (out_dirs["metrics"] / "pytest_results.csv").exists()
    manifest = json.loads((out_dirs["experiments"] / "pytest_manifest.json").read_text("utf-8"))
    assert manifest["test_set_used"] is False
    assert manifest["tracking_backend"] == "local"

    runs = mlflow.search_runs(experiment_names=["pytest-exp"])
    assert len(runs) == 3  # 1 pai + 2 modelos
    child = runs[runs["tags.model_name"] == "xgboost"].iloc[0]
    assert not np.isnan(child["metrics.cv_f1_macro_mean"])
    assert child["tags.test_set_used"] == "false"

    # modelo recarregado do MLflow produz as mesmas predições do pipeline refeito
    monkeypatch.setattr("saude_fetal.predict.load_config", lambda _: exp_cfg)
    loaded = load_model(f"runs:/{child['run_id']}/model")
    selected = manifest["best_params"]["xgboost"]
    xgb_cfg = model_cfg["models"]["xgboost"]
    ref = build_pipeline({**xgb_cfg, "params": {**xgb_cfg["params"], **selected}}, FEATURES, 0).fit(
        X, y
    )
    pd.testing.assert_series_equal(
        pd.Series(loaded.predict(X)), pd.Series(ref.predict(X)), check_names=False
    )
    np.testing.assert_allclose(loaded.predict_proba(X), ref.predict_proba(X))
    np.testing.assert_array_equal(loaded.classes_, [1, 2, 3])
    outer_grids = pd.read_csv(out_dirs["metrics"] / "pytest_xgboost_outer_grid_results.csv")
    assert outer_grids.groupby("outer_fold").size().to_dict() == {1: 2, 2: 2, 3: 2}
    assert (out_dirs["figures"] / "pytest_oof_pr_roc.png").is_file()
