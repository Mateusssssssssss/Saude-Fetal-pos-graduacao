"""Publica o pipeline selecionado no DagsHub a partir do pacote congelado."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature

from saude_fetal.config import PROJECT_ROOT, load_config
from saude_fetal.data import file_sha256
from saude_fetal.tracking import setup_mlflow, tracking_session


def publish_bundle(bundle: Path, experiment_id: str) -> str:
    """Registra o pacote carregado no backend ativo e atribui o alias após verificar."""
    metadata = json.loads((bundle / "metadata.json").read_text(encoding="utf-8"))
    binary = bundle / metadata["artifact"]
    if file_sha256(binary) != metadata["sha256"]:
        raise ValueError("O SHA-256 do pipeline diverge dos metadados do pacote.")
    example = json.loads((bundle / "request_example.json").read_text(encoding="utf-8"))
    sample = pd.DataFrame([example], columns=metadata["features"], dtype="float64")
    pipeline = joblib.load(binary)
    if pipeline.classes_.tolist() != metadata["class_order"]:
        raise ValueError("A ordem das classes diverge dos metadados do pacote.")

    source = metadata["mlflow"]
    name, alias = source["registered_model"], source["alias"]
    source_tags = {
        "source.joblib_sha256": metadata["sha256"],
        "source.run_id": source["run_id"],
        "dataset.name": metadata["dataset"]["name"],
    }
    client = mlflow.MlflowClient()
    published = next(
        (
            item
            for item in client.search_model_versions(f"name = '{name}'")
            if item.tags.get("source.joblib_sha256") == metadata["sha256"]
        ),
        None,
    )
    if published is None:
        requirements = (bundle / "requirements.txt").read_text(encoding="utf-8").splitlines()
        requirements = [
            f"mlflow=={mlflow.__version__}" if line.startswith("mlflow==") else line
            for line in requirements
        ]
        with mlflow.start_run(run_name=f"publicacao_{name}", experiment_id=experiment_id):
            mlflow.set_tags({"stage": "publish_frozen_model", **source_tags})
            mlflow.log_params(metadata["hyperparameters"])
            mlflow.log_metrics(metadata["metrics"])
            mlflow.log_dict(metadata, "source_metadata.json")
            model_info = mlflow.sklearn.log_model(
                pipeline,
                name="model",
                signature=infer_signature(sample, pipeline.predict(sample)),
                input_example=sample,
                serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
                code_paths=[str(bundle / "code" / "saude_fetal")],
                pip_requirements=requirements,
            )
            published = mlflow.register_model(model_info.model_uri, name, tags=source_tags)

    uri = f"models:/{name}/{published.version}"
    registered = mlflow.sklearn.load_model(uri)
    np.testing.assert_array_equal(registered.predict(sample), pipeline.predict(sample))
    np.testing.assert_allclose(registered.predict_proba(sample), pipeline.predict_proba(sample))
    client.set_registered_model_alias(name, alias, published.version)
    return uri


def main() -> None:
    cfg = load_config("experiment")["mlflow"]
    with tracking_session(cfg, backend="dagshub"):
        uri = publish_bundle(PROJECT_ROOT / "best_model", setup_mlflow(cfg))
        print(f"DagsHub: {uri}")


if __name__ == "__main__":
    main()
