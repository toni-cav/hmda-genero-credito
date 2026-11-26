| Especificação | Tipo | N | OR do gênero (IC 95%) | p | AME (p.p.) (IC 95%) | Verificação MPL (p.p.) (IC 95%) |
|---|---|---|---|---|---|---|
| (1) Só gênero | principal | 1.532.885 | 1,160 (1,098; 1,225) | <0,001 | 1,41 (0,78; 2,04) | 1,41 (0,78; 2,04) |
| (2) + controles | principal | 1.532.885 | 1,022 (0,988; 1,057) | 0,205 | 0,16 (-0,09; 0,41) | 0,20 (-0,07; 0,47) |
| (3) + EF estado | principal | 1.532.885 | 1,010 (0,976; 1,045) | 0,561 | 0,07 (-0,18; 0,33) | 0,12 (-0,15; 0,40) |
| (4) + EF instituição | principal | 1.514.115 | 1,015 (0,989; 1,041) | 0,264 | 0,09 (-0,09; 0,27) | 0,08 (-0,10; 0,26) |
| (5) + EF instituição e estado [H1] | principal | 1.514.115 | 1,005 (0,980; 1,032) | 0,678 | 0,04 (-0,14; 0,21) | 0,03 (-0,15; 0,21) |
| S1: (4) sem exigir state_code/idade | exploratória | 1.521.774 | 1,017 (0,992; 1,044) | 0,184 | 0,11 (-0,07; 0,30) | 0,09 (-0,09; 0,27) |
| S2: (5) todas as linhas | exploratória | 1.529.269 | 1,006 (0,981; 1,032) | 0,658 | 0,04 (-0,13; 0,20) | 0,01 (-0,17; 0,19) |
| E1: (5) + faixa etária | exploratória | 1.514.115 | 0,974 (0,948; 1,000) | 0,050 | -0,17 (-0,32; -0,02) | -0,20 (-0,39; -0,01) |

Notas: OR > 1 indica maior chance de negação para mulheres em relação a homens. Erros-padrão clusterizados por instituição (`lei`). AME = efeito marginal médio de ser mulher na probabilidade de negação. Nas especificações com efeitos fixos, o IC do AME usa o método delta tratando os efeitos fixos como conhecidos (aproximado); a coluna MPL (modelo linear de probabilidade, mesmos controles e efeitos fixos) traz o efeito em p.p. com inferência exata como verificação. N varia entre especificações porque o logit com efeitos fixos descarta instituições/estados sem variação no desfecho (ver `removidas_singleton` e `removidas_separacao` no CSV). S1 e S2 são sensibilidades à exclusão de linhas com campos faltantes: S1 não exige `state_code`/`applicant_age` (spec. 4 não usa estado) e S2 mantém todas as linhas, com estado ausente como categoria própria do efeito fixo e renda <= 0 com indicador. S1, S2 e E1 são exploratórias. Associação estatística não é prova de discriminação (protocolo, seção 2).
