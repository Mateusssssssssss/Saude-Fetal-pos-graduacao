"""Fixtures com dados SINTÉTICOS (não dependem do CSV real)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

FEATURES = ["severe_decelerations", "accelerations", "fetal_movement", "uterine_contractions"]
TARGET = "fetal_health"


@pytest.fixture
def data_cfg(tmp_path):
    return {
        "dataset": {"name": "synthetic", "raw_path": str(tmp_path / "raw.csv")},
        "target": {
            "column": TARGET,
            "classes": {1: "Normal", 2: "Suspeito", 3: "Patologico"},
            "critical_class": 3,
        },
        "features": FEATURES,
        "split": {
            "test_size": 0.2,
            "random_state": 0,
            "stratify": True,
            "train_path": str(tmp_path / "processed" / "train.csv"),
            "test_path": str(tmp_path / "processed" / "test.csv"),
            "manifest_path": str(tmp_path / "split_manifest.json"),
        },
    }


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    """300 exames sintéticos com sinal fraco ligado à classe."""
    rng = np.random.default_rng(0)
    n = 300
    y = rng.choice([1, 2, 3], size=n, p=[0.7, 0.2, 0.1])
    df = pd.DataFrame(
        {
            "severe_decelerations": (rng.random(n) < 0.05 * y).astype(float),
            "accelerations": rng.poisson(np.where(y == 1, 4, 1)).astype(float),
            "fetal_movement": rng.poisson(2, n).astype(float),
            "uterine_contractions": rng.poisson(np.where(y == 3, 2, 5)).astype(float),
            TARGET: y.astype(float),
        }
    )
    return df


@pytest.fixture
def raw_csv(synthetic_df, data_cfg):
    synthetic_df.to_csv(data_cfg["dataset"]["raw_path"], index=False)
    return data_cfg["dataset"]["raw_path"]
