# Estrutura e Regras do Projeto

## Diretórios
- `data/`: Datasets brutos e processados (NUNCA altere os arquivos originais nem imprima tabelas inteiras).
- `code/notebooks/`: Notebooks numerados que documentam as etapas acadêmicas e exploratórias.
- `code/scripts/`: Scripts executáveis e funções compartilhadas de limpeza, features, treino e avaliação.
- `figuras/`: Gráficos exportados em PNG/PDF (300 dpi) para o relatório.
- `resultados/`: Tabelas e métricas versionáveis; modelos e predições grandes são artefatos locais.
- `docs/`: Protocolo, dados e documentação acadêmica.

## Convenções de Execução e Caminhos
- Execute scripts a partir da **raiz do projeto** (ex.: `python code/scripts/09_precificacao_h3.py`), usando caminhos relativos como `data/arquivo.parquet` e `figuras/grafico.png`.
- Abra os notebooks em `code/notebooks/`; eles identificam a raiz do repositório e normalizam o diretório de trabalho.
- Extraia lógica reutilizável de notebooks para `code/scripts/` e importe-a nos notebooks, evitando cópias divergentes.
- Preserve os contratos dos artefatos entre etapas. A Etapa D grava `resultados/etapa_05_h2_modelos.joblib` e `resultados/etapa_05_h2_predicoes_teste.parquet`; as etapas posteriores dependem desses nomes e do formato versionado do modelo.
- Não altere dados originais nem registre datasets, modelos ou predições grandes no Git.