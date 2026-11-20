"""Baixa o pipeline do DagsHub e inicia o servidor de inferência do MLflow."""

from __future__ import annotations

import os
from tempfile import mkdtemp

import mlflow

from saude_fetal.config import load_config
from saude_fetal.tracking import tracking_session


def main() -> None:
    if not os.environ.get("DAGSHUB_USER_TOKEN"):
        raise SystemExit("Defina DAGSHUB_USER_TOKEN para baixar o modelo do DagsHub.")

    model_uri = os.environ.get("MODEL_URI", "models:/saude-fetal-xgboost@champion")
    cfg = load_config("experiment")["mlflow"]
    with tracking_session(cfg, backend="dagshub"):
        model_dir = mlflow.artifacts.download_artifacts(
            artifact_uri=model_uri, dst_path=mkdtemp(prefix="saude-fetal-")
        )

    environment = os.environ.copy()
    environment.pop("DAGSHUB_USER_TOKEN", None)
    args = [
        "mlflow", "models", "serve", "--model-uri", model_dir,
        "--env-manager", "local", "--host", "0.0.0.0", "--port", "8080",
    ]
    os.execvpe(args[0], args, environment)


if __name__ == "__main__":
    main()
