# Etapa H — Mitigação de viés e trade-off

Execução: 2026-11-27 00:37:20 -03. Foram reutilizados os índices do split salvo em `h2_modelos.joblib` (treino n=1,073,019; teste n=459,866), com os modelos e hiperparâmetros da Etapa D.

As taxas e métricas de fairness descrevem apenas a decisão prevista de negação. O HMDA não contém inadimplência observada, portanto não se estimam equalized odds, equal opportunity ou erro de risco.

O reweighing usa exclusivamente a tabela conjunta `mulher` × `negado` do treino. O ajuste de limiar escolhe um limiar por grupo no teste, maximizando acurácia entre combinações com DI ≥ 0,80; em empate favorece menor |DP|. Essa escolha no teste é pós-hoc e deve ser lida como análise exploratória do trade-off, não como validação fora da amostra.

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

`ΔDP = P(negação prevista | mulher) − P(negação prevista | homem)`. `DI = P(aprovação prevista | mulher) / P(aprovação prevista | homem)`; referência da regra dos 80%: DI ≥ 0,80. Queda de disparidade é a redução percentual de |ΔDP| frente ao baseline; quedas negativas indicam aumento. Perdas são baseline menos estratégia, em pontos percentuais.

O gráfico também mostra as fronteiras exploradas pelos limiares específicos por grupo nas probabilidades do baseline e do modelo reponderado; o ponto quadrado representa Reweighing. Consulte `etapa_h_mitigacao_metricas.csv` para limiares e valores completos.
