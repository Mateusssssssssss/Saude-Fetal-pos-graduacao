"""Caminhos do projeto e leitura das configurações YAML."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def _find_project_root() -> Path:
    """Localiza a raiz do projeto (pasta que contém o pyproject.toml).

    Pode ser sobrescrita pela variável de ambiente ``SAUDE_FETAL_ROOT``.
    """
    env_root = os.environ.get("SAUDE_FETAL_ROOT")
    if env_root:
        return Path(env_root).resolve()
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd().resolve()


PROJECT_ROOT = _find_project_root()
CONFIG_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"
EXPERIMENTS_DIR = REPORTS_DIR / "experiments"
MODELS_DIR = PROJECT_ROOT / "models"


def resolve_path(path: str | Path) -> Path:
    """Converte caminho relativo (à raiz do projeto) em absoluto."""
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Lê um arquivo YAML e devolve um dicionário."""
    with open(resolve_path(path), encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_config(name: str, config_dir: str | Path | None = None) -> dict[str, Any]:
    """Carrega ``configs/<name>.yaml`` (ex.: ``data``, ``model``, ``experiment``)."""
    base = Path(config_dir) if config_dir else CONFIG_DIR
    return load_yaml(base / f"{name}.yaml")


def ensure_dirs() -> None:
    """Cria as pastas de saída, caso não existam."""
    for d in (FIGURES_DIR, METRICS_DIR, EXPERIMENTS_DIR, MODELS_DIR, DATA_DIR / "processed"):
        d.mkdir(parents=True, exist_ok=True)
