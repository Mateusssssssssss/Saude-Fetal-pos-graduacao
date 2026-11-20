import numpy as np
import pandas as pd

from saude_fetal.analysis import (
    association_table,
    duplicate_report,
    learn_bins,
    rate_by_bins,
)

from .conftest import FEATURES, TARGET

CLASSES = {1: "Normal", 2: "Suspeito", 3: "Patologico"}


def test_learn_bins_low_cardinality_returns_none():
    assert learn_bins(pd.Series([0, 1, 0, 1, 2])) is None


def test_learn_bins_extends_to_infinity():
    edges = learn_bins(pd.Series(np.arange(100, dtype=float)), max_bins=4)
    assert edges[0] == -np.inf and edges[-1] == np.inf


def test_rate_by_bins_rates_sum_to_one(synthetic_df):
    X, y = synthetic_df[FEATURES], synthetic_df[TARGET].astype(int)
    table = rate_by_bins(X, y, "accelerations", CLASSES)
    rates = table[[f"taxa_{c}" for c in CLASSES.values()]].sum(axis=1)
    assert np.allclose(rates, 1.0)
    assert table["n_total"].sum() == len(X)


def test_association_table_columns(synthetic_df):
    X, y = synthetic_df[FEATURES], synthetic_df[TARGET].astype(int)
    tab = association_table(X, y)
    assert set(tab.columns) == {"spearman", "abs_spearman", "mutual_info"}
    assert set(tab.index) == set(FEATURES)


def test_duplicate_report_detects_conflicts():
    df = pd.DataFrame({"a": [1, 1, 2], "b": [0, 0, 0], "t": [1, 2, 1]})
    rep = duplicate_report(df, ["a", "b"], "t")
    assert rep["duplicadas_somente_features"] == 1
    assert rep["combinacoes_com_rotulos_conflitantes"] == 1
