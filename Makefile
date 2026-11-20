# Atalhos (opcional; no Windows use os comandos equivalentes do README).
.PHONY: install data train test mlflow format

install:
	pip install -e ".[notebook,dev]"

data:
	python -m saude_fetal.data

train:
	python -m saude_fetal.train

test:
	python -m pytest

mlflow:
	mlflow ui --backend-store-uri ./mlruns

format:
	black src tests scripts
