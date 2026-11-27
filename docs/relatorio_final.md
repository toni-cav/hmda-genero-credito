# Gênero e crédito imobiliário — síntese dos resultados

> Os resultados abaixo são de execuções anteriores. O código agora usa semente 444; reexecute as etapas para atualizar métricas, figuras e outputs.

## Pergunta e escopo

Analisamos pedidos individuais para compra de residência principal no HMDA 2025. Comparamos decisões entre mulheres e homens, avaliamos modelos que reproduzem essas decisões e investigamos variáveis associadas ao gênero. A análise é observacional: não estima causalidade nem risco de inadimplência.

O protocolo, os filtros e as decisões metodológicas estão em [protocolo.md](protocolo.md). A origem e a integridade dos dados estão em [dados/integridade_2025.md](dados/integridade_2025.md).

## Resultados principais

### H1 — associação com negação

Na especificação principal, com controles de renda, DTI, LTV, tipo de empréstimo e efeitos fixos de instituição e estado, o odds ratio estimado para mulheres foi **1,005** (IC 95%: **0,980 a 1,032**; p = **0,678**). O efeito do MPL foi **0,03 p.p.** (IC 95%: **−0,15 a 0,21**). A análise não rejeita a hipótese nula de ausência de associação condicional nessa especificação. O contraste bruto entre grupos não deve ser interpretado como diferença ajustada.

### H2 — previsões e disparidade

O Gradient Boosting sem gênero obteve AUC-ROC **0,7867**, PR-AUC **0,5444** e Brier **0,0649**, praticamente igual ao modelo com gênero. Ainda assim, no teste, a diferença entre as taxas previstas de negação foi **3,66 p.p.** (IC 95%: **3,46 a 3,84**); o disparate impact foi **0,960** (IC 95%: **0,958 a 0,962**). O resultado mostra disparidade nas decisões previstas, mas não ultrapassa o limiar de 0,80 da regra dos 80%.

Um classificador de gênero baseado nas demais variáveis obteve AUC **0,6642** (IC 95%: **0,6627 a 0,6658**), evidência de que os preditores carregam informação associada ao gênero. Isso não demonstra, por si só, discriminação.

### H3 — preço do crédito

Entre os empréstimos originados, a associação ajustada para mulheres foi de **−0,0356** em `rate_spread` (IC 95%: **−0,0502 a −0,0210**; p < 0,001), **−0,0351** em `interest_rate` (IC 95%: **−0,0504 a −0,0199**; p < 0,001) e **+95,37** em `total_loan_costs` (IC 95%: **67,22 a 123,52**; p < 0,001). Essas associações têm populações e observações completas distintas; não indicam causalidade e devem ser interpretadas junto às unidades e definições das variáveis publicadas pelo HMDA.

### Robustez e mitigação

Na especificação exploratória com raça e etnia, o OR foi **0,963** (IC 95%: **0,942 a 0,985**; p = 0,001); na base imputada, foi **1,023** (p = 0,223). A conclusão depende, portanto, da especificação e do tratamento dos dados faltantes. Nos subgrupos avaliados, resultados isolados devem ser tratados como exploratórios e sujeitos a comparações múltiplas.

O reweighing e o ajuste de limiares da Etapa H são exploratórios. Como os limiares por grupo foram escolhidos usando os rótulos do teste, essas métricas não avaliam desempenho fora da amostra.

## Limitações e itens pendentes

- O HMDA público não fornece credit score, histórico de crédito nem inadimplência observada. O modelo aprende a decisão histórica dos credores, não o risco real de crédito.
- Gênero é analisado em categorias binárias publicadas como `Male` e `Female`; outras identidades não são representadas.
- Os resultados são associações observacionais e podem refletir fatores não observados.
- A comparação planejada com HMDA 2024 não foi realizada.
- A Etapa H deve ser lida como demonstração metodológica; para uma comparação confirmatória, os limiares precisam ser selecionados em validação separada, nunca no teste final.
- Uma versão acadêmica para entrega ainda pode exigir revisão bibliográfica, referências externas e formatação conforme as orientações da disciplina.

## Como reproduzir

Consulte o [README.md](../README.md) para instalar o ambiente e executar o pipeline. Dados, modelos e previsões grandes ficam localmente; gere-os antes de executar as etapas dependentes.
