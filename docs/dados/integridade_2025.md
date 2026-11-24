# Integridade dos dados — HMDA 2025

Fonte: HMDA Snapshot National Loan-Level Dataset 2025 (FFIEC), arquivo `2025_public_lar_csv.zip` publicado em 23/06/2026, baixado em 23/11/2026 pelo notebook `notebooks/01_dados.ipynb`.

## Verificações

| Verificação | Resultado | |
|---|---|---|
| Linhas no LAR | 13,543,606 | |
| Soma de `lar_count` no Transmittal Sheet | 13,543,606 | ✅ |
| Colunas no LAR (esperado 99) | 99 | ✅ |
| Instituições (LEI distintos) no LAR | 4,782 | |
| Instituições no Transmittal Sheet | 4,782 | ✅ |
| LEIs do LAR ausentes do TS | 0 | ✅ |
| LEIs do TS sem linhas no LAR | 0 | ✅ |
| Instituições com `lar_count` ≠ linhas no LAR | 0 | ✅ |
| `activity_year` presentes | 2025 | ✅ |
| Linhas no Parquet = linhas no CSV | 13,543,606 | ✅ |

Valores inesperados ao converter códigos para inteiro: nenhum ✅

## Parquet gerado

- Arquivo: `data/hmda_2025.parquet` (ignorado pelo Git)
- 13,543,606 linhas × 75 colunas, 472 MB (zstd)
- Colunas descartadas: campos de múltipla resposta `*_2` a `*_5` e `multifamily_affordable_units`

## Distribuições brutas (antes dos filtros da issue #3)

**`action_taken`**

| Código | Linhas | % |
|---|---|---|
| 1 | 6,827,891 | 50.4% |
| 2 | 417,961 | 3.1% |
| 3 | 2,115,090 | 15.6% |
| 4 | 1,710,723 | 12.6% |
| 5 | 695,006 | 5.1% |
| 6 | 1,578,064 | 11.7% |
| 7 | 50,499 | 0.4% |
| 8 | 148,372 | 1.1% |

**`derived_sex`**

| Valor | Linhas | % |
|---|---|---|
| Male | 4,173,097 | 30.8% |
| Joint | 4,065,993 | 30.0% |
| Female | 2,662,580 | 19.7% |
| Sex Not Available | 2,641,936 | 19.5% |
