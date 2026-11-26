# Gênero e Crédito Imobiliário — HMDA 2025

Projeto final de SME0828 (Introdução à Ciência de Dados, Prof. Francisco Rodrigues, ICMC/USP).

**Pergunta:** mulheres têm mais pedidos de financiamento imobiliário negados que homens com o mesmo perfil de risco? E um modelo de análise de crédito treinado nas decisões dos bancos reproduz essa diferença, mesmo sem usar a variável gênero?

- Protocolo de hipóteses, métricas e filtros: [`docs/protocolo.md`](docs/protocolo.md)
- Versão e integridade dos dados: [`docs/dados/`](docs/dados/)

## Notebooks

Rodar em ordem, a partir da pasta `notebooks/`:

| Notebook | Etapa | O que faz |
|---|---|---|
| `01_dados.ipynb` | Issue #2 | Baixa o HMDA 2025, confere a integridade e gera o Parquet |
| `02_amostra.ipynb` | Issue #3 | Filtros, fluxograma da amostra e tratamento de faltantes |
| `03_analisedescritiva.ipynb` | Etapa A (#4) | Diferenças brutas entre homens e mulheres |
| `04_regressao_logistica.ipynb` | Etapa B (#5) | Regressão logística com controles e efeitos fixos (H1) |
| `05_gradient_boosting.ipynb` | Etapa D (#6) | Modelos de crédito com e sem gênero (H2) |
| `06_fairness.ipynb` | Etapa E (#7) | Métricas de fairness: paridade demográfica, disparate impact e diferença condicional |

## Estrutura

```
data/         dados brutos e amostras (não vão para o GitHub)
docs/         protocolo e documentação dos dados
figuras/      gráficos gerados pelos notebooks
notebooks/    análises, em ordem
resultados/   tabelas de resultados (.md); previsões e modelos ficam só no local
```

## Ambiente

```powershell
python -m venv <caminho-do-venv>
<caminho-do-venv>\Scripts\Activate.ps1
pip install -r requirements.txt
```

Fonte: [FFIEC HMDA Snapshot National Loan-Level Dataset](https://ffiec.cfpb.gov/data-publication/snapshot-national-loan-level-dataset).
