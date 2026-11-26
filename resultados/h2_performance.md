| Modelo | AUC-ROC | PR-AUC | Brier Score | Limiar (CV treino) | Taxa prev. teste | DP prévia (p.p.) |
|---|---|---|---|---|---|---|
| Com gênero | 0,7889 | 0,5481 | 0,0646 | 0,1441 | 10,42% | 3,19 |
| Sem gênero | 0,7887 | 0,5481 | 0,0646 | 0,1437 | 10,44% | 3,51 |

Notas: Divisão 70/30 estratificada conjuntamente por `negado` e `mulher` (`random_state=42`). Limiar escolhido via CV 5-fold no treino para igualar a taxa prevista à observada. DP prévia = P(ŷ=1 | Mulher) − P(ŷ=1 | Homem) no teste (IC por bootstrap na Etapa E).
