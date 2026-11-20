"""Tracking e log de modelos no MLflow local ou hospedado pelo DagsHub."""

from __future__ import annotations

import logging
import os
import platform
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from urllib.parse import urlsplit

import mlflow
from mlflow.exceptions import MlflowException

from saude_fetal.config import PROJECT_ROOT, resolve_path

logger = logging.getLogger(__name__)


def resolve_tracking_uri(uri: str) -> str:
    """Resolve tracking URI relativo (``sqlite:///...`` ou diretório local como ``mlruns``)
    a partir da raiz do projeto.

    Evita criar stores diferentes conforme o diretório de trabalho (ex.: notebooks/).
    """
    prefix = "sqlite:///"
    if uri.startswith(prefix):
        db_path = uri[len(prefix) :]
        if not (db_path.startswith("/") or ":" in db_path):
            return prefix + (PROJECT_ROOT / db_path).as_posix()
        return uri
    if "://" in uri:
        return uri
    path = resolve_path(uri)
    path.mkdir(parents=True, exist_ok=True)
    return path.as_uri()


@contextmanager
def tracking_session(
    mlflow_cfg: dict[str, Any], backend: str | None = None, tracking_uri: str | None = None
) -> Iterator[str]:
    """Seleciona o destino e limita as credenciais DagsHub à operação atual."""
    backend = backend or mlflow_cfg.get("backend", "local")
    if backend == "local":
        uri = resolve_tracking_uri(tracking_uri or mlflow_cfg["tracking_uri"])
        if urlsplit(uri).scheme == "file":
            # O projeto já tem runs e um registro local em mlruns/.
            os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        mlflow.set_tracking_uri(uri)
        mlflow.set_registry_uri(uri)
        yield backend
        return
    if backend != "dagshub":
        raise ValueError(f"Backend de tracking desconhecido: {backend}")
    if tracking_uri is not None:
        raise ValueError("Use --tracking-uri com --tracking-backend local.")
    if mlflow.active_run() is not None:
        raise RuntimeError("Encerre o run ativo antes de trocar o destino para DagsHub.")

    try:
        import dagshub
        from dagshub.common import config as dagshub_config
        from dagshub.common.api import RepoAPI
        from dotenv import dotenv_values
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            'Instale a integração com python -m pip install -e ".[dagshub]".'
        ) from exc

    # O SDK configura a autenticação do MLflow em variáveis de processo.
    keys = (
        "MLFLOW_TRACKING_URI",
        "MLFLOW_TRACKING_USERNAME",
        "MLFLOW_TRACKING_PASSWORD",
        "MLFLOW_TRACKING_TOKEN",
        "MLFLOW_REGISTRY_URI",
    )
    previous_env = {key: os.environ.get(key) for key in keys}
    previous_uri = mlflow.get_tracking_uri()
    previous_token = dagshub_config.token
    previous_host = dagshub_config.host
    try:
        # O cliente DagsHub lê o token no import. Sincronize também quando já
        # foi importado, por exemplo num kernel Jupyter reutilizado.
        token = os.environ.get("DAGSHUB_USER_TOKEN") or dotenv_values(PROJECT_ROOT / ".env").get(
            "DAGSHUB_USER_TOKEN"
        )
        dagshub_config.token = token or None
        repo = mlflow_cfg["dagshub"]
        # Verifica o repositório existente antes do init, que também pode criá-lo.
        host = "https://dagshub.com"
        dagshub_config.set_host(host)
        RepoAPI(f"{repo['repo_owner']}/{repo['repo_name']}", host=host).get_repo_info()
        dagshub.init(
            repo_owner=repo["repo_owner"], repo_name=repo["repo_name"], host=host, mlflow=True
        )
        os.environ.pop("MLFLOW_TRACKING_TOKEN", None)
        # O setter tem precedência sobre o ambiente em sessões reutilizadas.
        uri = os.environ["MLFLOW_TRACKING_URI"]
        mlflow.set_tracking_uri(uri)
        mlflow.set_registry_uri(uri)
        yield backend
    finally:
        dagshub_config.token = previous_token
        dagshub_config.set_host(previous_host)
        mlflow.set_tracking_uri(previous_uri)
        mlflow.set_registry_uri(previous_env["MLFLOW_REGISTRY_URI"] or "")
        for key, value in previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def setup_mlflow(mlflow_cfg: dict[str, Any]) -> str:
    """Seleciona o experimento no backend ativo e devolve seu ID."""
    uri = mlflow.get_tracking_uri()
    name = mlflow_cfg["experiment_name"]
    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name(name)
    if experiment is None:
        artifact_location = None
        if urlsplit(uri).scheme in {"file", "sqlite"}:
            artifact_dir = resolve_path(mlflow_cfg.get("artifact_dir", "mlartifacts"))
            artifact_dir.mkdir(parents=True, exist_ok=True)
            artifact_location = artifact_dir.as_uri()
        try:
            experiment_id = client.create_experiment(name, artifact_location=artifact_location)
            experiment = client.get_experiment(experiment_id)
        except MlflowException as exc:
            # Outro processo pode ter criado o mesmo experimento em paralelo.
            if exc.error_code != "RESOURCE_ALREADY_EXISTS":
                raise
            experiment = client.get_experiment_by_name(name)
            if experiment is None:
                raise
    logger.info("Experimento MLflow %s (%s)", name, experiment.experiment_id)
    return experiment.experiment_id


def flatten_dict(d: dict[str, Any], parent: str = "", sep: str = ".") -> dict[str, Any]:
    """Achata dicionários aninhados para ``mlflow.log_params``."""
    items: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{parent}{sep}{k}" if parent else str(k)
        if isinstance(v, dict):
            items.update(flatten_dict(v, key, sep))
        else:
            items[key] = v
    return items


def environment_tags() -> dict[str, str]:
    """Versões das bibliotecas principais (reprodutibilidade)."""
    import matplotlib
    import numpy
    import pandas
    import sklearn

    return {
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "pandas_version": pandas.__version__,
        "numpy_version": numpy.__version__,
        "matplotlib_version": matplotlib.__version__,
        "mlflow_version": mlflow.__version__,
    }
