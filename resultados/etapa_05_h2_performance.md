| Modelo | AUC-ROC | PR-AUC | Brier Score | Limiar (CV treino) | Taxa prev. teste | DP prévia (p.p.) |
|---|---|---|---|---|---|---|
| Com gênero | 0,7859 | 0,5448 | 0,0649 | 0,1439 | 10,31% | 3,52 |
| Sem gênero | 0,7861 | 0,5454 | 0,0648 | 0,1437 | 10,37% | 3,77 |

Notas: Divisão 70/30 estratificada conjuntamente por `negado` e `mulher` (`random_state=444`). Limiar escolhido via CV 5-fold no treino para igualar a taxa prevista à observada. DP prévia = P(ŷ=1 | Mulher) − P(ŷ=1 | Homem) no teste (IC por bootstrap na Etapa E).
