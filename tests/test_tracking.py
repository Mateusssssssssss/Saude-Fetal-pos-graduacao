import importlib
import os
from types import SimpleNamespace
from unittest.mock import Mock

import mlflow
import pytest

from saude_fetal.config import load_config
from saude_fetal.tracking import setup_mlflow, tracking_session


def test_dagshub_session_uses_remote_artifacts_and_restores_credentials(tmp_path, monkeypatch):
    dagshub = pytest.importorskip("dagshub")
    sdk_init = importlib.import_module("dagshub.common.init")
    from dagshub.common import config as sdk_config

    prior_sdk_token = sdk_config.token
    cfg = load_config("experiment")["mlflow"]
    local_uri = (tmp_path / "mlruns").as_uri()
    mlflow.set_tracking_uri(local_uri)
    monkeypatch.delenv("DAGSHUB_USER_TOKEN", raising=False)
    monkeypatch.setenv("MLFLOW_TRACKING_PASSWORD", "previous-password")
    monkeypatch.setenv("MLFLOW_TRACKING_TOKEN", "previous-bearer")
    monkeypatch.setattr("saude_fetal.tracking.PROJECT_ROOT", tmp_path)
    (tmp_path / ".env").write_text("DAGSHUB_USER_TOKEN=test-token\n", encoding="utf-8")
    monkeypatch.setattr(sdk_init.RepoAPI, "get_repo_info", Mock())
    monkeypatch.setattr(dagshub.auth, "get_authenticator", Mock())
    monkeypatch.setattr(sdk_init, "get_token", lambda **_: sdk_config.token)
    monkeypatch.setattr(
        sdk_init.UserAPI, "get_user_from_token", lambda *_, **__: SimpleNamespace(username="test")
    )

    client = Mock()
    client.get_experiment_by_name.return_value = None
    client.create_experiment.return_value = "42"
    client.get_experiment.return_value.experiment_id = "42"
    monkeypatch.setattr(mlflow, "MlflowClient", lambda: client)

    with tracking_session(cfg, backend="dagshub"):
        assert mlflow.get_tracking_uri() == (
            "https://dagshub.com/mateusssssss/Saude-Fetal-pos-graduacao.mlflow"
        )
        assert os.environ["MLFLOW_TRACKING_PASSWORD"] == "test-token"
        assert "MLFLOW_TRACKING_TOKEN" not in os.environ
        assert setup_mlflow(cfg) == "42"
        client.create_experiment.assert_called_once_with(
            cfg["experiment_name"], artifact_location=None
        )

    assert mlflow.get_tracking_uri() == local_uri
    assert os.environ["MLFLOW_TRACKING_PASSWORD"] == "previous-password"
    assert os.environ["MLFLOW_TRACKING_TOKEN"] == "previous-bearer"
    assert "DAGSHUB_USER_TOKEN" not in os.environ
    assert sdk_config.token == prior_sdk_token
