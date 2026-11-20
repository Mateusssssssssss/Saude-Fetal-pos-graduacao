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
│   └── experiment.yaml              # Validação cruzada e destino DagsHub
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
│       ├── publish_best_model.py    # Publicação do pipeline congelado no DagsHub
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

O projeto usa **MLflow 3.18.0** no ambiente Conda `saude_fetal`. O extra `dagshub` instala o SDK e mantém a faixa de `graphql-core` compatível com o MLflow.

```bash
conda activate saude_fetal
python -m pip install -e ".[notebook,dev,dagshub]"
```

Se precisar criar ou atualizar o ambiente a partir do `environment.yml`:

```bash
conda env update -n saude_fetal -f environment.yml --prune
conda activate saude_fetal
python -m pip install -e ".[notebook,dev,dagshub]"
```

### Autenticar no DagsHub

Os novos runs usam o MLflow hospedado em `mateusssssss/Saude-Fetal-pos-graduacao`, conforme [configs/experiment.yaml](configs/experiment.yaml). Na primeira configuração, copie `.env.example` para `.env` na raiz e preencha `DAGSHUB_USER_TOKEN` com um token com acesso de escrita ao repositório. O Git ignora o `.env`. O SDK também aceita a sessão criada por `dagshub login`. [Repositório no DagsHub](https://dagshub.com/mateusssssss/Saude-Fetal-pos-graduacao) · [Autenticação do cliente DagsHub](https://dagshub.com/docs/client/reference/auth/)

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

### 4.3. Treinar as árvores e registrar os experimentos no DagsHub

```bash
python -m saude_fetal.train --workflow trees --data-config data_full
```

As árvores usam os mesmos cinco folds externos agrupados da CNN. A busca interna também mantém exames com vetores de features idênticos juntos.

Para usar o [notebook de treino](notebooks/02_model_training.ipynb), defina `RUN_TRAINING = True` e acrescente `data_config="data_full"` à chamada de `run_training`.

Dentro de cada fold externo, uma `GridSearchCV` de três folds seleciona por F1 macro e reajusta a melhor configuração. Os pipelines dos folds externos geram os rótulos e scores OOF. Depois, uma busca de três folds no TRAIN completo seleciona e reajusta o pipeline final de cada família.

Cada grade tem oito combinações. A Random Forest explora profundidade, tamanho da folha e ponderação de classes com 150 árvores. O XGBoost explora profundidade, quantidade de árvores e taxa de aprendizado. São 150 ajustes por família, incluindo os seis refits. O desempate usa o primeiro candidato na ordem da grade.

A aba **Experiments** do DagsHub mostra o run pai e um run filho por família. O tracking guarda métricas agregadas, configurações, hashes, grades em CSV e gráficos. Com `mlflow.log_models` ativo, o pipeline ajustado também fica associado ao run. O comando de publicação da seção 4.8 cria a versão no Model Registry. Os relatórios para os notebooks ficam em `reports/runs/<parent_run_id>/`. [MLflow no DagsHub](https://dagshub.com/mateusssssss/Saude-Fetal-pos-graduacao.mlflow/#/experiments/0)

### 4.4. Consultar o histórico local do MLflow

O ambiente usa MLflow 3.18.0. Para abrir os runs e versões já armazenados em `mlruns/`, use o script abaixo.

```powershell
.\scripts\mlflow_ui.ps1
# ou diretamente:
$env:MLFLOW_ALLOW_FILE_STORE = 'true'
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
python -m pip install -e ".[dl,notebook,dagshub]"
python -m saude_fetal.train --workflow cnn --data-config data_full
```

Use `--workflow all --data-config data_full` para treinar as árvores e a CNN na base completa. O [notebook de métricas da CNN](notebooks/notebooks_metrics_dl.ipynb) mostra curvas por época, métricas por fold, matriz OOF e curvas PR e ROC da classe Patológico.

O modelo Keras é salvo em `runs:/<run_id>/model`. Para carregá-lo, abra uma sessão de tracking com `tracking_session` de `saude_fetal.tracking` e chame `mlflow.keras.load_model`. Use uma matriz `float32` com as colunas na ordem de `input_schema.json`. A coluna de maior score indica a posição da classe em `class_order`.

### 4.8. Carregar e publicar o XGBoost selecionado

O XGBoost selecionado pela CV externa da base completa foi publicado no Model Registry do DagsHub como `models:/saude-fetal-xgboost@champion` (versão 1). O diretório `best_model/` reúne o pipeline Joblib, o SHA-256, o schema de entrada e a proveniência do ajuste original. O arquivo `requirements.txt` dessa pasta registra as dependências do pipeline exportado. [Run de publicação no DagsHub](https://dagshub.com/mateusssssss/Saude-Fetal-pos-graduacao.mlflow/#/experiments/0/runs/04fa1edf388341ca96c0fbead7a848e6)

```python
from saude_fetal.predict import load_model

pipeline = load_model("models:/saude-fetal-xgboost@champion", tracking_backend="dagshub")
```

Use `models:/saude-fetal-xgboost/1` para fixar a versão avaliada no TEST.

Para repetir a publicação, configure a autenticação e execute pela raiz do projeto. O comando reutiliza a versão registrada com o mesmo SHA-256 e valida a inferência com um exemplo sintético.

```powershell
python -m saude_fetal.publish_best_model
```

Para usar a cópia Joblib da pasta `best_model/`, carregue o pacote local.

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
