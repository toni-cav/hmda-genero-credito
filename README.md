# Gênero e Crédito Imobiliário — HMDA 2025

Disparidade de gênero na negação de financiamento imobiliário nos EUA, com estatística (regressão logística) e ML fairness (Gradient Boosting, métricas de fairness e proxies).

- Protocolo de hipóteses, métricas e filtros: [`docs/protocolo.md`](docs/protocolo.md)
- Versão e integridade dos dados: [`docs/dados/`](docs/dados/)

## Reproduzindo

```powershell
python -m venv <caminho-do-venv>
<caminho-do-venv>\Scripts\Activate.ps1
pip install -r requirements.txt

# Depois, rodar os notebooks em ordem:
#   notebooks/01_dados.ipynb    -> baixa o HMDA 2025, checa integridade e gera o Parquet
#   notebooks/02_amostra.ipynb  -> filtros, fluxograma e tratamento de faltantes
```

Os dados ficam em `data/`, que está no `.gitignore` (não sobem para o GitHub).

Fonte: [FFIEC HMDA Snapshot National Loan-Level Dataset](https://ffiec.cfpb.gov/data-publication/snapshot-national-loan-level-dataset).
