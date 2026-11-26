|                      |   AUC-ROC |   PR-AUC |   Brier |
|:---------------------|----------:|---------:|--------:|
| GB com gênero        |    0.7868 |   0.5444 |  0.0649 |
| GB sem gênero        |    0.7867 |   0.5444 |  0.0649 |
| Logística sem gênero |    0.7401 |   0.4723 |  0.0721 |

Teste: 30% da base (459,866 pedidos). Hiperparâmetros (GridSearchCV): {'learning_rate': 0.05, 'max_leaf_nodes': 15}. Limiares: com gênero = 0.1432, sem gênero = 0.1433.
