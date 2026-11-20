import numpy as np
import pandas as pd

from saude_fetal.preprocess import build_preprocessor

from .conftest import FEATURES


def test_scaler_learns_only_from_fit_data():
    """O scaler usa média/desvio do conjunto de ajuste, não do conjunto transformado."""
    train = pd.DataFrame({f: [0.0, 2.0, 4.0] for f in FEATURES})
    other = pd.DataFrame({f: [100.0, 200.0] for f in FEATURES})
    pre = build_preprocessor(FEATURES, scale=True).fit(train)
    out = pre.transform(other)
    expected = (100.0 - 2.0) / np.std([0.0, 2.0, 4.0])
    assert np.isclose(out[0, 0], expected)


def test_imputer_uses_train_median():
    train = pd.DataFrame({f: [1.0, 3.0, 5.0] for f in FEATURES})
    new = pd.DataFrame({f: [np.nan] for f in FEATURES})
    pre = build_preprocessor(FEATURES, scale=False).fit(train)
    assert np.allclose(pre.transform(new), 3.0)


def test_extra_columns_are_dropped():
    train = pd.DataFrame({**{f: [1.0, 2.0] for f in FEATURES}, "extra": [9, 9]})
    out = build_preprocessor(FEATURES, scale=False).fit_transform(train)
    assert out.shape == (2, len(FEATURES))
