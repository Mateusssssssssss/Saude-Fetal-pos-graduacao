# Abre a interface do MLflow apontando para o backend local do projeto.
# Uso (PowerShell, com o ambiente conda "saude_fetal" ativo):
#   .\scripts\mlflow_ui.ps1            # porta 5000
#   .\scripts\mlflow_ui.ps1 -Port 5001
param([int]$Port = 5000)

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:MLFLOW_ALLOW_FILE_STORE = 'true'
mlflow ui --backend-store-uri "./mlruns" --port $Port
