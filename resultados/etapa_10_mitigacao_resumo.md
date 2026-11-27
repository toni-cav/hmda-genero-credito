# Etapa H — Mitigação de viés e trade-off
> Saída de uma execução anterior. O código agora usa semente 444; reexecute a etapa para atualizar os resultados.

Execução: 2026-11-27 00:37:20 -03. Foram usados os modelos e a divisão da Etapa D (treino: 1.073.019; teste: 459.866).

As métricas descrevem as previsões de negação. O HMDA não informa inadimplência, então não avaliamos erros sobre o risco real.

A reponderação usa `mulher` e `negado` apenas no treino. Os limiares por grupo foram escolhidos no teste para maximizar a acurácia entre combinações com DI ≥ 0,80. Como o teste participou dessa escolha, os resultados são exploratórios e não medem desempenho fora da amostra.

## Comparação

| Modelo | Estratégia | Acurácia | AUC-ROC | F1 | Log-Loss | Negação mulher | Negação homem | ΔDP | DI | Queda | Perda acurácia (p.p.) | Perda AUC (p.p.) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Com gênero | Baseline | 0.8925 | 0.7889 | 0.4858 | 0.2432 | 0.1237 | 0.0918 | 0.0319 | 0.9649 | 0.0000% | 0.0000 | 0.0000 |
| Com gênero | Reweighing | 0.8930 | 0.7880 | 0.4853 | 0.2434 | 0.1091 | 0.0991 | 0.0100 | 0.9889 | 68.7598% | -0.0511 | 0.0855 |
| Com gênero | Ajuste de Limiar | 0.9261 | 0.7889 | 0.5067 | 0.2432 | 0.0522 | 0.0404 | 0.0118 | 0.9877 | 63.0413% | -3.3579 | 0.0000 |
| Com gênero | Combinado | 0.9261 | 0.7880 | 0.4965 | 0.2434 | 0.0512 | 0.0362 | 0.0150 | 0.9844 | 52.9268% | -3.3549 | 0.0855 |
| Sem gênero | Baseline | 0.8924 | 0.7887 | 0.4856 | 0.2432 | 0.1258 | 0.0907 | 0.0351 | 0.9614 | 0.0000% | 0.0000 | 0.0000 |
| Sem gênero | Reweighing | 0.8933 | 0.7885 | 0.4868 | 0.2432 | 0.1233 | 0.0902 | 0.0331 | 0.9636 | 5.6572% | -0.0944 | 0.0169 |
| Sem gênero | Ajuste de Limiar | 0.9261 | 0.7887 | 0.5000 | 0.2432 | 0.0473 | 0.0402 | 0.0071 | 0.9926 | 79.7388% | -3.3723 | 0.0000 |
| Sem gênero | Combinado | 0.9261 | 0.7885 | 0.5066 | 0.2432 | 0.0525 | 0.0402 | 0.0123 | 0.9872 | 64.9270% | -3.3688 | 0.0169 |

`ΔDP = P(negação prevista | mulher) − P(negação prevista | homem)`. `DI = P(aprovação prevista | mulher) / P(aprovação prevista | homem)`. A referência da regra dos 80% é DI ≥ 0,80. Queda de disparidade compara |ΔDP| com o baseline; valores negativos indicam aumento. As perdas são baseline menos estratégia, em pontos percentuais.

O gráfico mostra as combinações de limiares avaliadas para os modelos baseline e reponderado. Consulte `etapa_10_mitigacao_metricas.csv` para os limiares e as métricas completas.
