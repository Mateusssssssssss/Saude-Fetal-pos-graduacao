# Saúde Fetal — Classificação de Cardiotocografias (CTG)

Projeto da pós-graduação em IA e Aprendizado de Máquina para classificação da saúde fetal a partir de exames de cardiotocografia (CTG). (baseado em Cookiecutter Data Science e Made With ML).

## 1. Problema, população e dados

- **Objetivo:** classificar exames de cardiotocografia (CTG) em três classes atribuídas por especialistas: `1 = Normal`, `2 = Suspeito`, `3 = Patológico`.
- **Unidade de observação:** um exame de CTG por linha (2.126 linhas em `data/raw/fetal_health_reduced.csv`). Não há identificador de paciente disponível na base.
- **Target:** `fetal_health` (multiclasse, ordinal em gravidade).
- **Classe crítica:** `Patológico (3)` — falsos negativos (não sinalizar um caso patológico) têm maior custo clínico; a avaliação acompanha `F1 macro` como métrica principal e `Recall`/`Precision` por classe (com atenção especial à classe Patológico).
- **Features disponíveis (`fetal_health_reduced.csv`):** `severe_decelerations`, `accelerations`, `fetal_movement`, `uterine_contractions`.
  - *Nota:* trata-se de uma versão reduzida (4 features) do dataset público *Fetal Health Classification*; a origem exata da redução e a unidade/escala exata dos valores devem ser confirmadas durante a EDA.

> **Aviso de saúde e ética:** base pública e sem identificadores pessoais. O projeto tem finalidade acadêmica e **não** representa prontidão clínica, validação prospectiva ou recomendação assistencial.

## 2. Estrutura do projeto

```text
Saude-Fetal-pos-graduacao/
├── AGENTS.md                        # Padrão metodológico e de engenharia do projeto
├── README.md
├── pyproject.toml                   # Dependências e configuração do pacote/pytest/black
├── environment.yml                  # Ambiente conda (saude_fetal)
├── requirements.txt                 # Dependências pip complementares
├── Makefile                         # Atalhos opcionais
├── configs/
│   ├── data.yaml                    # Schema, classes, features e split TRAIN/TEST
│   ├── model.yaml                   # Random Forest, XGBoost e grades de busca
│   └── experiment.yaml              # Validação cruzada e configuração do MLflow
├── data/
│   ├── raw/
│   │   └── fetal_health_reduced.csv # Origem imutável
│   ├── interim/
│   ├── processed/                   # train.csv e test.csv gerados pelo split
│   └── external/
├── notebooks/
│   ├── 01_eda.ipynb                 # EDA (integridade na base completa; análises no TRAIN)
│   ├── 02_model_training.ipynb      # Disparo controlado (RUN_TRAINING=False) do treino + MLflow
│   └── 03_model_metrics.ipynb       # Análise das métricas de treino/CV/OOF (lê artefatos)
├── src/
│   └── saude_fetal/
│       ├── __init__.py
│       ├── config.py                # Caminhos a partir da raiz e leitura de YAMLs
│       ├── data.py                  # Validação de schema e split TRAIN/TEST persistido
│       ├── preprocess.py            # ColumnTransformer + imputação + scaling opcional
│       ├── model.py                 # Construção de pipelines a partir de model.yaml
│       ├── validation.py            # StratifiedKFold, métricas por fold e OOF
│       ├── evaluation.py            # Scorers, métricas multiclasse e gráficos
│       ├── analysis.py              # Funções de apoio à EDA (somente TRAIN)
│       ├── tracking.py              # MLflow local ou hospedado pelo DagsHub
│       ├── train.py                 # Fluxo de treino com CV no TRAIN + log no MLflow
│       └── predict.py               # Inferência a partir de pipeline salvo
├── scripts/
│   ├── prepare_data.py              # Wrapper CLI para criar o split TRAIN/TEST
│   ├── train.py                     # Wrapper CLI para treinar e logar no MLflow
│   └── mlflow_ui.ps1                # Abre a interface do MLflow (PowerShell)
├── tests/                           # Testes automatizados com pytest (dados sintéticos)
├── models/
├── reports/
│   ├── figures/
│   ├── metrics/
│   └── experiments/
└── docs/
```

## 3. Instalação e ambiente

Com o ambiente conda `saude_fetal` ativo:

```bash
conda activate saude_fetal
pip install -e ".[notebook,dev]"
```

Se precisar criar ou atualizar o ambiente a partir do `environment.yml`:

```bash
conda env update -n saude_fetal -f environment.yml --prune
pip install -e ".[notebook,dev]"
```

## 4. Como executar (fluxo reproduzível)

### 4.1. Preparar o split TRAIN/TEST (uma única vez)

```bash
python -m saude_fetal.data
# ou: python scripts/prepare_data.py
```

Gera `data/processed/train.csv`, `data/processed/test.csv` e o manifesto `reports/experiments/split_manifest.json` (com hashes SHA256 e contagens por classe). Por segurança, o split não é sobrescrito sem `--force`.

### 4.2. Executar a EDA

Abra e execute `notebooks/01_eda.ipynb` no JupyterLab/VS Code usando o kernel `saude_fetal`.

- Checagens de integridade (schema, ausentes, duplicatas) rodam na base completa.
- Todas as análises que orientam modelagem (distribuições, faixas, correlações, associação vs permutation importance) rodam **exclusivamente no TRAIN**.
- Tabelas e figuras são salvas em `reports/metrics/eda_*.csv` e `reports/figures/eda_*.png`.

### 4.3. Buscar hiperparâmetros e registrar os modelos no MLflow

```bash
python -m saude_fetal.train
# ou: python scripts/train.py
```

Para treinar Random Forest e XGBoost com as 21 features de `data/raw/fetal_health.csv`, use `python -m saude_fetal.train --data-config data_full`. A base completa usa os mesmos folds externos da CNN e mantém vetores idênticos juntos também nos folds internos da busca.

Também é possível disparar pelo notebook `notebooks/02_model_training.ipynb` alterando `RUN_TRAINING = True`.

Por padrão, o destino é o FileStore local em `mlruns/`, com artefatos em `mlartifacts/`. O fluxo compara Random Forest e XGBoost em cinco folds externos estratificados no TRAIN. Dentro de cada fold externo, uma `GridSearchCV` de três folds seleciona por F1 macro e reajusta a melhor configuração. Os pipelines dos folds externos geram os rótulos e scores OOF. Depois, uma busca de três folds no TRAIN completo seleciona e reajusta o pipeline final de cada família.

Cada grade tem oito combinações. A Random Forest explora profundidade, tamanho da folha e ponderação de classes com 150 árvores. O XGBoost explora profundidade, quantidade de árvores e taxa de aprendizado. São 150 ajustes por família, incluindo os seis refits. O desempate usa o primeiro candidato na ordem da grade.

O run pai registra configuração, hashes, versões, métricas comparativas, manifesto e figuras. Os dois runs filhos registram as grades completas, parâmetros escolhidos, métricas por fold, diagnósticos OOF e o pipeline final, quando habilitado. O TEST fica reservado para a avaliação da receita congelada.

#### Registrar no DagsHub

O `environment.yml` inclui a integração DagsHub. Com um ambiente `saude_fetal` já existente, instale o extra pela raiz do projeto.

```powershell
conda activate saude_fetal
python -m pip install -e ".[dagshub]"
```

Para autenticação por token, copie `.env.example` para `.env` na raiz e preencha `DAGSHUB_USER_TOKEN` com um token de uma conta com acesso de escrita ao repositório. A variável de ambiente tem precedência sobre o arquivo, que é ignorado pelo Git. Em sessão interativa, o SDK também aceita autenticação via `dagshub login`. [Autenticação do cliente DagsHub](https://dagshub.com/docs/client/reference/auth/)

```powershell
python -m saude_fetal.train --tracking-backend dagshub
```

No notebook de treino, defina `TRACKING_BACKEND = "dagshub"`. Para alterar o padrão do projeto, defina `mlflow.backend: dagshub` em `configs/experiment.yaml`. Use `--tracking-backend local` para selecionar o FileStore do projeto.

O treino chama `dagshub.init(..., mlflow=True)` com o proprietário `mateusssssss` e o repositório `Saude-Fetal-pos-graduacao`. Acompanhe os runs na aba **Experiments** do repositório. O DagsHub hospeda métricas, configurações, gráficos e pipelines, conforme `mlflow.log_models`. O CSV de treino é representado apenas por metadados no tracking. O pipeline aparece como artefato do run. [Repositório DagsHub](https://dagshub.com/mateusssssss/Saude-Fetal-pos-graduacao) · [Integração DagsHub com MLflow](https://dagshub.com/docs/integration_guide/mlflow_tracking/)

Os relatórios de cada execução ficam em `reports/runs/<parent_run_id>/`. O manifesto identifica o destino, o experimento e os runs filhos. Para carregar um modelo remoto, use `load_model` de `saude_fetal.predict`.

```python
from saude_fetal.predict import load_model

pipeline = load_model("runs:/<run_id_do_modelo>/model", tracking_backend="dagshub")
```

### 4.4. Abrir a interface do MLflow

```powershell
.\scripts\mlflow_ui.ps1
# ou diretamente:
mlflow ui --backend-store-uri ./mlruns
```

Acesse `http://127.0.0.1:5000` e abra o experimento `saude-fetal-classificacao`.

### 4.5. Analisar métricas de treino/CV/OOF

Abra [03_model_metrics.ipynb](notebooks/03_model_metrics.ipynb) para consultar o run completo mais recente com os dois modelos. Preencha `RUN_ID` para abrir um run específico. O notebook lê os artefatos gerados pelo treino e apresenta barras de F1, Recall e Precision macro, comparação por fold, diferenças entre treino e validação e as matrizes OOF. Também inclui curvas PR e ROC por classe, calibração e distribuição do score patológico, resultados da grade e os parâmetros finais.

### 4.6. Rodar os testes

```bash
python -m pytest
```

### 4.7. Treinar a CNN 1D

A CNN usa TensorFlow/Keras e recebe as quatro features na ordem de `configs/data.yaml`. Como a base contém medidas agregadas, a convolução percorre colunas tabulares. A primeira receita usa 60 épocas por fold e no refit. Imputação e normalização são aprendidas em cada bloco de treino e fazem parte do modelo salvo.

```powershell
conda activate saude_fetal
python -m pip install -e ".[dl,notebook]"
python -m saude_fetal.train --workflow cnn
```

Para treinar a CNN com as 21 features de `data/raw/fetal_health.csv`, use `python -m saude_fetal.train --workflow cnn --data-config data_full`. Esse perfil grava um split separado, preserva as linhas reservadas para TEST e exclui do TRAIN vetores de features idênticos aos do TEST. A validação da base completa mantém vetores idênticos no mesmo fold.

Use `--workflow all` para executar também a busca das árvores, ou acrescente `--tracking-backend dagshub` para registrar os runs remotamente. O fluxo padrão continua sendo `trees`. O [notebook de métricas da CNN](notebooks/notebooks_metrics_dl.ipynb) lê o último run concluído e mostra curvas por época, métricas por fold, matriz OOF e PR/ROC patológica.

O modelo Keras é salvo em `runs:/<run_id>/model`. Para carregá-lo, abra uma sessão de tracking com `tracking_session` de `saude_fetal.tracking` e chame `mlflow.keras.load_model`. Use uma matriz `float32` com as colunas na ordem de `input_schema.json`. A coluna de maior score indica a posição da classe em `class_order`.

### 4.8. Carregar o XGBoost selecionado

O XGBoost selecionado pela CV externa da base completa está no Model Registry local como `saude-fetal-xgboost@champion`. O diretório `best_model/` reúne o pipeline Joblib, o SHA-256, o schema de entrada e a referência da versão no MLflow. O arquivo `requirements.txt` dessa pasta registra as dependências do pipeline exportado.

```python
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

bundle = Path("best_model").resolve()
sys.path.insert(0, str(bundle / "code"))
metadata = json.loads((bundle / "metadata.json").read_text(encoding="utf-8"))
example = json.loads((bundle / "request_example.json").read_text(encoding="utf-8"))
X = pd.DataFrame([example], columns=metadata["features"], dtype="float64")
pipeline = joblib.load(bundle / "best_model.joblib")
print(pipeline.predict(X))
print(pipeline.predict_proba(X))  # Colunas na ordem de metadata["class_order"].
```

### 4.9. Consultar a avaliação no TEST

O [notebook de métricas no TEST](notebooks/notebook_metrics_teste.ipynb) compara o XGBoost versão 1, com 21 features, ao XGBoost congelado de quatro features nos mesmos exames reservados. O comando `python -m saude_fetal.evaluate_test` gera os relatórios agregados em `reports/test/xgboost_full_vs_reduced_v1/` e reaproveita os resultados quando já existem.

As métricas do TEST registram a avaliação das receitas congeladas. Mudanças guiadas por esses resultados exigem novos dados reservados para confirmação independente.

## 5. Metodologia atual e limitações

- **Split:** 80% TRAIN / 20% TEST estratificado (`random_state=42`).
- **Validação:** CV aninhada estratificada no TRAIN com seed 42 e divisão externa compartilhada. F1 macro orienta a busca. Confira também o comportamento da classe Patológico e a dispersão entre folds.
- **Pré-processamento e decisão:** imputação pela mediana ajustada dentro de cada fold. O XGBoost codifica os rótulos internamente e devolve as classes originais 1, 2 e 3. A decisão usa a classe de maior score.
- **Relatórios publicados:** o Git inclui os gráficos da EDA, os manifestos dos splits, as execuções oficiais de árvores e da CNN na base completa e a comparação no TEST. As tabelas geradas pela EDA e o histórico exploratório ficam no disco local.
- **Limitações conhecidas:** apenas 4 features discretas (possíveis combinações idênticas com rótulos distintos); ausência de identificador de paciente impede `GroupKFold`; unidades originais da versão reduzida pendentes de confirmação.
