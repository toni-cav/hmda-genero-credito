# Estrutura e Regras do Projeto

## Diretórios
- `data/`: Datasets brutos e processados (NUNCA altere os arquivos originais nem imprima tabelas inteiras).
- `notebooks/`: Rascunhos e Análise Exploratória de Dados (EDA) em `.ipynb`.
- `code/scripts/`: Todos os scripts `.py` modulares do projeto (limpeza, engenharia de features, treino e avaliação). Não coloque scripts `.py` soltos direto em `code/`.
- `figuras/`: Gráficos exportados em PNG/PDF (300 dpi) para o relatório.
- `resultados/`: Métricas salvas (CSV/JSON) e predições dos modelos.
- `docs/`: Relatório final e documentação acadêmica.

## Convenções de Execução e Caminhos
- Sempre assuma que os comandos são executados a partir da **raiz do projeto** (ex.: `python code/scripts/treino.py`), usando caminhos relativos como `data/arquivo.csv` e `figuras/grafico.png`.
- Ao finalizar uma análise em `notebooks/`, extraia as funções reutilizáveis para `code/scripts/`.