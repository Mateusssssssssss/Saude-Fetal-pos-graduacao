import numpy as np
import pandas as pd
import pytest

from saude_fetal.config import load_config, resolve_path
from saude_fetal.data import SchemaError, load_split, make_split, split_train_test, validate_schema

from .conftest import FEATURES, TARGET


def test_validate_schema_converts_target_to_int(synthetic_df):
    out = validate_schema(synthetic_df, FEATURES, TARGET, [1, 2, 3])
    assert out[TARGET].dtype.kind == "i"
    assert set(out[TARGET].unique()) <= {1, 2, 3}


def test_validate_schema_rejects_missing_column(synthetic_df):
    with pytest.raises(SchemaError, match="ausentes"):
        validate_schema(synthetic_df.drop(columns="accelerations"), FEATURES, TARGET, [1, 2, 3])


@pytest.mark.parametrize("bad", [-1.0, np.inf])
def test_validate_schema_rejects_invalid_values(synthetic_df, bad):
    df = synthetic_df.copy()
    df.loc[0, "accelerations"] = bad
    with pytest.raises(SchemaError):
        validate_schema(df, FEATURES, TARGET, [1, 2, 3])


def test_validate_schema_rejects_unknown_class(synthetic_df):
    df = synthetic_df.copy()
    df.loc[0, TARGET] = 4
    with pytest.raises(SchemaError, match="desconhecidas"):
        validate_schema(df, FEATURES, TARGET, [1, 2, 3])


def test_make_split_is_disjoint_stratified_and_not_overwritten(raw_csv, data_cfg):
    manifest = make_split(data_cfg)
    train = pd.read_csv(data_cfg["split"]["train_path"])
    test = pd.read_csv(data_cfg["split"]["test_path"])

    assert set(train["row_id"]).isdisjoint(test["row_id"])
    assert len(train) + len(test) == manifest["n_rows"]["raw"]
    # estratificação: prevalência parecida entre TRAIN e TEST
    p_train = train[TARGET].value_counts(normalize=True)
    p_test = test[TARGET].value_counts(normalize=True)
    assert (p_train - p_test).abs().max() < 0.05

    # segunda chamada não recria o split (protege o TEST)
    first_hash = manifest["test_sha256"]
    assert make_split(data_cfg)["test_sha256"] == first_hash


def test_load_split_returns_features_and_target(raw_csv, data_cfg):
    make_split(data_cfg)
    X, y = load_split("train", data_cfg)
    assert list(X.columns) == FEATURES
    assert "row_id" not in X.columns
    assert len(X) == len(y)


def test_grouped_split_keeps_test_and_removes_overlapping_train_features(synthetic_df, data_cfg):
    split_cfg = data_cfg["split"]
    split_cfg["group_duplicate_features"] = True
    data_cfg["signed_features"] = [FEATURES[0]]
    synthetic_df[FEATURES[0]] = -1.0
    train_before, test_before = split_train_test(
        synthetic_df, TARGET, split_cfg["test_size"], split_cfg["random_state"]
    )
    synthetic_df.loc[train_before.index[0], FEATURES] = test_before.iloc[0][FEATURES]
    synthetic_df.to_csv(data_cfg["dataset"]["raw_path"], index=False)

    manifest = make_split(data_cfg)
    train_features, _ = load_split("train", data_cfg)
    test_disk = pd.read_csv(split_cfg["test_path"])
    assert test_disk["row_id"].tolist() == test_before["row_id"].tolist()
    assert manifest["excluded_train_feature_overlap"] >= 1
    assert len(train_features) + manifest["excluded_train_feature_overlap"] == len(train_before)
    assert (train_features[FEATURES[0]] == -1).all()
    assert train_features.merge(test_disk[FEATURES], on=FEATURES).empty


def test_real_raw_dataset_schema_if_present():
    """Valida o CSV real quando ele existe no repositório."""
    cfg = load_config("data")
    path = resolve_path(cfg["dataset"]["raw_path"])
    if not path.exists():
        pytest.skip("Dataset bruto não encontrado")
    df = validate_schema(pd.read_csv(path), cfg["features"], cfg["target"]["column"], [1, 2, 3])
    assert len(df) > 0
