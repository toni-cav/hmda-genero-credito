# Gênero e crédito imobiliário — HMDA 2025

Projeto final da disciplina SME0828 — Introdução à Ciência de Dados, ICMC/USP.

O projeto compara as decisões de crédito para mulheres e homens com perfis observáveis semelhantes. Também avalia se modelos sem a variável gênero reproduzem diferenças nas decisões dos bancos.

- Protocolo, hipóteses e filtros: [docs/protocolo.md](docs/protocolo.md)
- Integridade e versão dos dados: [docs/dados/integridade_2025.md](docs/dados/integridade_2025.md)
- Resumo dos resultados: [docs/relatorio_final.md](docs/relatorio_final.md)

## Estrutura

```text
code/
  notebooks/     análises numeradas
  scripts/       preparação e execução das etapas
data/            dados originais e arquivos Parquet locais
docs/            protocolo, dados e relatório
figuras/         gráficos
resultados/      tabelas e métricas
tests/           testes do preparo dos dados
```

## Ambiente e dados

Execute os comandos na raiz do repositório:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
jupyter lab code/notebooks/01_dados.ipynb
```

Execute `01_dados.ipynb` e `02_amostra.ipynb` primeiro. Eles baixam e verificam o HMDA 2025, salvam `data/hmda_2025.parquet` e criam as amostras usadas nas análises. Os dados e modelos grandes ficam fora do Git. Fonte: [FFIEC HMDA Snapshot National Loan-Level Dataset](https://ffiec.cfpb.gov/data-publication/snapshot-national-loan-level-dataset).

## Análises em notebook

Execute os notebooks na ordem da tabela. Cada um configura o diretório de trabalho para a raiz do projeto.

| Notebook | Etapa | Conteúdo e dependências |
|---|---|---|
| `01_dados.ipynb` | Dados | Download, integridade e Parquet completo |
| `02_amostra.ipynb` | Amostra | Filtros, fluxograma, faltantes e amostras |
| `03_analise_descritiva.ipynb` | A | Diferenças brutas por gênero |
| `04_regressao_logistica_h1.ipynb` | B / H1 | Regressão logística, efeitos fixos e MPL |
| `05_modelagem_h2.ipynb` | D / H2 | Modelos com e sem gênero; gera artefatos H2 |
| `06_equidade.ipynb` | E | Fairness; requer as predições da Etapa D |
| `07_proxies_genero.ipynb` | F | Classificador de gênero e SHAP |
| `08_robustez.ipynb` | Robustez | Controles adicionais, subgrupos e faltantes |

As etapas seguintes usam o modelo e as previsões da Etapa D. Eles não são incluídos no Git; execute novamente a Etapa D para gerá-los.

Os notebooks 04 e 05 usam os mesmos pipelines dos scripts H1 e H2; não mantêm cópias próprias do preparo e do ajuste dos modelos. A lógica reutilizável fica em `regressao_logistica_h1.py` e `modelagem_h2.py`; os scripts numerados são entradas curtas para execução pela linha de comando. Execute o notebook para acompanhar a análise e os gráficos, ou o script para rodar o pipeline. A comparação adicional com regressão logística fica no notebook H2 e é salva em `resultados/etapa_05_h2_comparacao_modelos.md`.

## Semente e resultados salvos

O código usa semente `444`. Os resultados, figuras e outputs já salvos vieram de execuções anteriores. Reexecute as etapas para gerar resultados com a nova seed.

## Scripts de execução

Os scripts podem ser executados a partir da raiz:

```powershell
python code/scripts/04_regressao_logistica_h1.py
python code/scripts/05_modelagem_h2.py
python code/scripts/09_precificacao_h3.py
python code/scripts/10_mitigacao.py
```

Execute a Etapa H depois de gerar `resultados/etapa_05_h2_modelos.joblib` na Etapa D. Os resultados e as figuras são salvos em suas respectivas pastas.

## Validação

```powershell
python -m unittest discover -s tests -v
python -m compileall -q code/scripts
```
