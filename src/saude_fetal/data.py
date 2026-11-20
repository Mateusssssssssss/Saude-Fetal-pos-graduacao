"""Carregamento, validação de schema e split TRAIN/TEST dos dados.

O split é feito UMA vez e persistido em ``data/processed``. O TEST fica
isolado: nenhuma etapa de EDA orientada a decisões, pré-processamento,
seleção, CV ou threshold deve usá-lo.

Uso (pela raiz do projeto)::

    python -m saude_fetal.data            # cria o split se ainda não existir
    python -m saude_fetal.data --force    # recria (sobrescreve) o split
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from collections.abc import Collection
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from saude_fetal.config import load_config, resolve_path

logger = logging.getLogger(__name__)


class SchemaError(ValueError):
    """Erro de schema ou de domínio nos dados."""


def file_sha256(path: str | Path) -> str:
    """Hash SHA256 do arquivo (identifica a versão exata dos dados)."""
    h = hashlib.sha256()
    with open(resolve_path(path), "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_schema(
    df: pd.DataFrame,
    features: list[str],
    target: str | None,
    classes: list[int] | None,
    *,
    signed_features: Collection[str] = (),
) -> pd.DataFrame:
    """Valida colunas, tipos e domínio; devolve cópia com o target como inteiro.

    Regras:
    - todas as features (e o target, se informado) presentes;
    - features numéricas e finitas, com valores negativos só nas colunas declaradas;
    - target sem ausentes e somente com as classes conhecidas.
    """
    required = list(features) + ([target] if target else [])
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SchemaError(f"Colunas ausentes: {missing}")

    out = df.copy()
    for col in features:
        if not pd.api.types.is_numeric_dtype(out[col]):
            raise SchemaError(f"Feature não numérica: {col}")
        values = out[col].to_numpy(dtype=float)
        finite = values[~np.isnan(values)]
        if not np.isfinite(finite).all():
            raise SchemaError(f"Valores infinitos em {col}")
        if col not in signed_features and (finite < 0).any():
            raise SchemaError(f"Valores negativos em {col}")

    if target:
        if out[target].isna().any():
            raise SchemaError("Target com valores ausentes")
        y = out[target]
        if not np.allclose(y, np.round(y)):
            raise SchemaError("Target com valores não inteiros")
        out[target] = y.round().astype(int)
        if classes is not None:
            unknown = sorted(set(out[target].unique()) - set(classes))
            if unknown:
                raise SchemaError(f"Classes desconhecidas no target: {unknown}")
    return out


def load_raw(data_cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    """Lê o CSV bruto (imutável) e valida o schema."""
    cfg = data_cfg or load_config("data")
    df = pd.read_csv(resolve_path(cfg["dataset"]["raw_path"]))
    return validate_schema(
        df,
        features=cfg["features"],
        target=cfg["target"]["column"],
        classes=[int(k) for k in cfg["target"]["classes"]],
        signed_features=cfg.get("signed_features", ()),
    )


def split_train_test(
    df: pd.DataFrame, target: str, test_size: float, random_state: int, stratify: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split estratificado TRAIN/TEST, preservando o índice original como coluna ``row_id``."""
    df = df.reset_index(drop=True).copy()
    df.insert(0, "row_id", df.index)
    train, test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[target] if stratify else None,
    )
    return train.sort_values("row_id"), test.sort_values("row_id")


def make_split(data_cfg: dict[str, Any] | None = None, force: bool = False) -> dict[str, Any]:
    """Cria e persiste o split TRAIN/TEST + manifesto com hashes e contagens.

    Não sobrescreve um split existente sem ``force=True``: recriar o split
    pode misturar TEST já consultado com o TRAIN.
    """
    cfg = data_cfg or load_config("data")
    split_cfg = cfg["split"]
    train_path = resolve_path(split_cfg["train_path"])
    test_path = resolve_path(split_cfg["test_path"])
    manifest_path = resolve_path(split_cfg["manifest_path"])

    if train_path.exists() and test_path.exists() and not force:
        logger.info("Split já existe em %s; use force=True para recriar.", train_path.parent)
        return (
            json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        )

    target = cfg["target"]["column"]
    df = load_raw(cfg)
    train, test = split_train_test(
        df,
        target=target,
        test_size=split_cfg["test_size"],
        random_state=split_cfg["random_state"],
        stratify=split_cfg.get("stratify", True),
    )
    excluded_overlap = 0
    if split_cfg.get("group_duplicate_features", False):
        features = cfg["features"]
        # Mantém o TEST original e exclui do treino vetores idênticos a ele.
        overlap = pd.MultiIndex.from_frame(train[features]).isin(
            pd.MultiIndex.from_frame(test[features])
        )
        excluded_overlap = int(overlap.sum())
        train = train.loc[~overlap]
    train_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "raw_path": cfg["dataset"]["raw_path"],
        "raw_sha256": file_sha256(cfg["dataset"]["raw_path"]),
        "train_path": split_cfg["train_path"],
        "train_sha256": file_sha256(train_path),
        "test_path": split_cfg["test_path"],
        "test_sha256": file_sha256(test_path),
        "test_size": split_cfg["test_size"],
        "random_state": split_cfg["random_state"],
        "stratify": split_cfg.get("stratify", True),
        "excluded_train_feature_overlap": excluded_overlap,
        "n_rows": {"raw": len(df), "train": len(train), "test": len(test)},
        "class_counts": {
            "train": {str(k): int(v) for k, v in train[target].value_counts().sort_index().items()},
            "test": {str(k): int(v) for k, v in test[target].value_counts().sort_index().items()},
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Split salvo: train=%d, test=%d", len(train), len(test))
    return manifest


def load_split(
    which: str = "train", data_cfg: dict[str, Any] | None = None
) -> tuple[pd.DataFrame, pd.Series]:
    """Carrega X, y de ``train`` (padrão) ou ``test``.

    Atenção: ``which="test"`` só deve ser usado na avaliação final do modelo
    escolhido, com receita congelada.
    """
    if which not in {"train", "test"}:
        raise ValueError("which deve ser 'train' ou 'test'")
    cfg = data_cfg or load_config("data")
    path = resolve_path(cfg["split"][f"{which}_path"])
    if not path.exists():
        raise FileNotFoundError(f"{path} não encontrado. Rode: python -m saude_fetal.data")
    df = validate_schema(
        pd.read_csv(path),
        features=cfg["features"],
        target=cfg["target"]["column"],
        classes=[int(k) for k in cfg["target"]["classes"]],
        signed_features=cfg.get("signed_features", ()),
    )
    return df[cfg["features"]], df[cfg["target"]["column"]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Cria o split TRAIN/TEST.")
    parser.add_argument("--force", action="store_true", help="Recria o split existente.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    manifest = make_split(force=args.force)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
