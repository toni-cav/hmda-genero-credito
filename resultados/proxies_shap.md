Classificador de gênero: AUC = 0.6642 (IC 95%: 0.6627 a 0.6658), teste com 459,866 pedidos.

|                    |   Importância para prever o gênero (SHAP) |   Contribuição para a diferença na negação (log-odds) |
|:-------------------|------------------------------------------:|------------------------------------------------------:|
| Renda (log)        |                                    0.2546 |                                                0.0729 |
| Idade              |                                    0.2468 |                                                0.0207 |
| Tipo de empréstimo |                                    0.1986 |                                                0.0419 |
| Estado             |                                    0.0763 |                                                0.0002 |
| LTV                |                                    0.0559 |                                               -0.0129 |
| DTI                |                                    0.0495 |                                                0.0174 |

Importância: média do |SHAP| do classificador de gênero (amostra de 50 mil pedidos do teste). Contribuição: média do SHAP do modelo de crédito sem gênero entre mulheres menos entre homens.
