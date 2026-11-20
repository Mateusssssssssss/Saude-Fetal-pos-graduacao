# Saúde Fetal — Classificação de Cardiotocografias (CTG)

Projeto de pós-graduação em IA e Aprendizado de Máquina para classificar exames de cardiotocografia (CTG) com 21 medidas agregadas por exame. A organização segue Cookiecutter Data Science e Made With ML.

## 1. Problema, população e dados

- **Objetivo:** classificar exames de cardiotocografia (CTG) em três classes atribuídas por especialistas: `1 = Normal`, `2 = Suspeito`, `3 = Patológico`.
- **Unidade de observação:** um exame de CTG por linha. O arquivo `data/raw/fetal_health.csv` contém **2.126 exames e 21 features**, além da coluna-alvo. O perfil está em [configs/data_full.yaml](configs/data_full.yaml).
- **Target:** `fetal_health` (multiclasse, ordinal em gravidade).
- **Classe crítica:** `Patológico (3)` — falsos negativos (não sinalizar um caso patológico) têm maior custo clínico; a avaliação acompanha `F1 macro` como métrica principal e `Recall`/`Precision` por classe (com atenção especial à classe Patológico).
- **Features disponíveis** incluem frequência cardíaca fetal basal, acelerações, movimentos fetais, contrações uterinas, três tipos de desaceleração, quatro medidas de variabilidade e dez estatísticas do histograma da frequência cardíaca. O YAML registra os nomes e a ordem usados pelos modelos.

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
│   ├── data_full.yaml               # Schema das 21 features e split TRAIN/TEST
│   ├── model.yaml                   # Random Forest, XGBoost e grades de busca
│   └── experiment.yaml              # Validação cruzada e configuração do MLflow
├── data/
│   ├── raw/
│   │   └── fetal_health.csv         # Base de 21 features, mantida localmente
│   ├── interim/
│   ├── processed/full/              # TRAIN e TEST da base completa
│   └── external/
├── notebooks/
│   ├── 01_eda.ipynb                 # EDA histórica
│   ├── 02_model_training.ipynb      # Disparo controlado (RUN_TRAINING=False) do treino + MLflow
│   ├── 03_model_metrics.ipynb       # Métricas de Random Forest e XGBoost
│   ├── notebooks_metrics_dl.ipynb   # Métricas da CNN
│   └── notebook_metrics_teste.ipynb # Avaliação final no TEST
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
│       ├── train_dl.py              # Treino e avaliação da CNN
│       ├── evaluate_test.py         # Geração dos resultados finais no TEST
│       └── predict.py               # Inferência a partir de pipeline salvo
├── scripts/
│   ├── prepare_data.py              # Wrapper CLI para criar o split TRAIN/TEST
│   ├── train.py                     # Wrapper CLI para treinar e logar no MLflow
│   └── mlflow_ui.ps1                # Abre a interface do MLflow (PowerShell)
├── tests/                           # Testes automatizados com pytest (dados sintéticos)
├── best_model/                      # Metadados e pipeline Joblib do XGBoost selecionado
├── reports/
│   ├── experiments/                 # Manifestos dos splits
│   ├── runs/                        # Métricas de validação, gráficos e grades
│   └── test/                        # Avaliação final no TEST
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

Coloque `fetal_health.csv` em `data/raw/`. Os dados brutos e as partições processadas são locais.

```bash
python -c "from saude_fetal.config import load_config; from saude_fetal.data import make_split; make_split(load_config('data_full'))"
```

O perfil grava `data/processed/full/train.csv` e `data/processed/full/test.csv`. O [manifesto do split](reports/experiments/split_full_manifest.json) registra **1.695 exames no TRAIN** e **426 no TEST**, além dos hashes SHA-256. O particionamento inicial reservou 20% para TEST com seed 42. Cinco linhas com vetores de features idênticos aos de exames do TEST foram excluídas do TRAIN.

### 4.2. Inspecionar as 21 features no TRAIN

```bash
python -c "from saude_fetal.config import load_config; from saude_fetal.data import load_split; X, y = load_split('train', load_config('data_full')); print(X.describe().T); print(y.value_counts().sort_index())"
```

Use o TRAIN para análises que orientem a modelagem. O [notebook de EDA](notebooks/01_eda.ipynb) reúne a análise histórica anterior ao experimento de 21 features.

### 4.3. Buscar hiperparâmetros e registrar os modelos no MLflow

```bash
python -m saude_fetal.train --workflow trees --data-config data_full
```

As árvores usam os mesmos cinco folds externos agrupados da CNN. A busca interna também mantém exames com vetores de features idênticos juntos.

Para usar o [notebook de treino](notebooks/02_model_training.ipynb), defina `RUN_TRAINING = True` e acrescente `data_config="data_full"` à chamada de `run_training`.

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
python -m saude_fetal.train --workflow trees --data-config data_full --tracking-backend dagshub
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

A CNN usa TensorFlow/Keras e recebe as 21 features na ordem de `configs/data_full.yaml`. A convolução percorre as colunas de medidas agregadas por exame. O treino usa 60 épocas por fold e no refit. A imputação e a normalização são aprendidas em cada bloco de treino e fazem parte do modelo salvo.

```powershell
conda activate saude_fetal
python -m pip install -e ".[dl,notebook]"
python -m saude_fetal.train --workflow cnn --data-config data_full
```

Use `--workflow all --data-config data_full` para treinar as árvores e a CNN na base completa. O [notebook de métricas da CNN](notebooks/notebooks_metrics_dl.ipynb) mostra curvas por época, métricas por fold, matriz OOF e curvas PR e ROC da classe Patológico.

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

O [notebook de métricas no TEST](notebooks/notebook_metrics_teste.ipynb) apresenta a avaliação do XGBoost versão 1 nos 426 exames reservados. A linha `XGBoost · 21 features` do [CSV da avaliação](reports/test/xgboost_full_vs_reduced_v1/metrics/test_model_metrics.csv) corresponde ao modelo selecionado.

As métricas do TEST registram a avaliação das receitas congeladas. Mudanças guiadas por esses resultados exigem novos dados reservados para confirmação independente.

## 5. Resultados da base de 21 features

Médias dos cinco folds externos agrupados no TRAIN. A AP patológica é calculada sobre as predições OOF.

| Modelo | F1 macro (média ± desvio-padrão) | Acurácia | Recall patológico | Precisão patológica | AP patológica OOF |
|---|---:|---:|---:|---:|---:|
| XGBoost | 92,2% ± 2,1 p.p. | 95,5% | 93,6% | 94,4% | 0,972 |
| Random Forest | 88,9% ± 2,1 p.p. | 93,5% | 90,0% | 92,7% | 0,974 |
| CNN 1D | 86,3% ± 2,8 p.p. | 92,6% | 82,9% | 89,2% | 0,932 |

O **XGBoost** foi selecionado pelo F1 macro médio no TRAIN. A configuração escolhida usa 200 árvores, profundidade máxima 2 e taxa de aprendizado 0,15.

No **TEST**, o XGBoost versão 1 obteve **89,1% de F1 macro**, **94,1% de acurácia**, **91,4% de recall patológico**, **88,9% de precisão patológica** e **0,881 de AP patológica**. Detectou 32 dos 35 casos patológicos e emitiu 4 alertas patológicos incorretos.

## 6. Metodologia e limitações

- **Validação:** cinco folds externos estratificados com agrupamento por vetores de features idênticos e seed 42. As árvores usam três folds internos para seleção por F1 macro. A CNN usa uma receita fixa.
- **Pré-processamento e decisão:** imputação pela mediana ajustada dentro de cada fold. A CNN também normaliza as features usando seu bloco de treino. O XGBoost devolve os rótulos originais 1, 2 e 3. A decisão usa a classe de maior score.
- **Relatórios publicados:** o Git inclui os gráficos históricos da EDA, os manifestos dos splits, as execuções oficiais na base completa e a avaliação no TEST. As tabelas da EDA e os runs exploratórios são mantidos localmente.
- **Limitações:** a base contém agregados por exame e carece de identificadores de pacientes. O agrupamento por features evita separar vetores idênticos e a independência entre pacientes permanece sem verificação.
