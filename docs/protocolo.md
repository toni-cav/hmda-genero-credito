# Protocolo de Análise — Gênero e Crédito Imobiliário (HMDA 2025)

Este documento fixa, **antes de qualquer análise dos dados**, a pergunta de pesquisa, as hipóteses, as métricas e os critérios de amostra. Toda alteração posterior deve ser registrada na seção [Registro de alterações](#7-registro-de-alterações), com data e justificativa.

---

## 1. Pergunta central

Entre solicitantes individuais de financiamento imobiliário para compra de residência principal nos EUA (HMDA 2025), mulheres têm probabilidade diferente de ter o crédito negado em relação a homens com perfil de risco observável semelhante? E modelos preditivos que não utilizam a variável de gênero reproduzem essa diferença por meio de variáveis correlacionadas (proxies)?

## 2. Hipóteses

Todas as hipóteses são **bidirecionais** (testam diferença, não direção), dado que a literatura apresenta resultados mistos.

| Hipótese | Enunciado | Etapa | Teste |
|---|---|---|---|
| **H1** | Após controlar por renda, DTI, LTV, tipo de empréstimo e efeitos fixos de instituição e estado, o gênero do solicitante está associado à probabilidade de negação. | B | H0: coeficiente de gênero = 0 na regressão logística. Rejeição se p < 0,05. |
| **H2** | Um modelo de Gradient Boosting treinado **sem** a variável de gênero produz taxas previstas de negação diferentes entre homens e mulheres, porque outras variáveis carregam informação de gênero. | D, E, F | (a) Diferença de paridade demográfica do modelo sem gênero com IC 95% que exclui zero; **e** (b) classificador de gênero a partir das demais variáveis com AUC significativamente > 0,5. |
| **H3** | Entre empréstimos originados, há diferença de custo do crédito entre gêneros após controles. | G | H0: coeficiente de gênero = 0 na regressão de `rate_spread` (principal) e `interest_rate` / `total_loan_costs` (secundárias). |

**Interpretação:** associação estatística **não** é tratada como prova de discriminação. Variáveis não observadas no HMDA público (por exemplo, credit score e histórico de crédito) podem explicar parte da diferença (ver seção 6).

## 3. Métricas

### 3.1 Inferência estatística (Etapas B e G)

- **Odds ratio** do gênero (`exp(β_gênero)`) com IC 95%.
- **Efeito marginal médio (AME)** do gênero, em pontos percentuais de probabilidade de negação.
- **Erros-padrão clusterizados por instituição** (`lei`).
- Nível de significância: **α = 0,05**. Resultados são reportados com IC, não apenas com p-valor.

### 3.2 Performance dos modelos preditivos (Etapa D)

- Divisão treino/teste **70/30, estratificada pelo alvo e pelo gênero**, com `random_state = 42`, **idêntica** para os modelos com e sem gênero.
- Métricas no conjunto de teste: **AUC-ROC** (principal), **PR-AUC** (o alvo é desbalanceado) e **Brier score**.
- Limiar de classificação: escolhido **no treino** (validação cruzada) para que a taxa prevista de negação seja igual à taxa observada. O mesmo critério é usado nos dois modelos.

### 3.3 Fairness (Etapa E)

Notação: ŷ = 1 indica negação prevista; F = feminino; M = masculino.

| Métrica | Definição | Referência |
|---|---|---|
| **Diferença de paridade demográfica** | DP = P(ŷ=1 \| F) − P(ŷ=1 \| M) | 0 = paridade |
| **Disparate impact** | DI = P(ŷ=0 \| F) / P(ŷ=0 \| M) (razão das taxas de aprovação) | DI < 0,8 indica impacto adverso (regra dos 80%) |
| **Diferença condicional ao risco** | DP calculada dentro de cada estrato de DTI × LTV e agregada pela média ponderada pelo tamanho dos estratos | 0 = sem diferença entre perfis comparáveis |

Os ICs das métricas de fairness são obtidos por **bootstrap (1.000 reamostragens)** do conjunto de teste.

**Métricas que não serão calculadas:** *equalized odds*, *equal opportunity* e *calibração por grupo* exigem o desfecho real de crédito (inadimplência), que **não existe no HMDA público**. O alvo disponível (negação) é a própria decisão da instituição, não o risco verdadeiro.

### 3.4 Proxies (Etapa F)

- Classificador (Gradient Boosting) que prevê `derived_sex` a partir das mesmas variáveis do modelo sem gênero. Métrica: **AUC-ROC** com IC por bootstrap.
- Importância das variáveis via **SHAP** (média do |valor SHAP|). Os proxies relevantes são as 5 variáveis de maior importância.

### 3.5 Variáveis excluídas dos modelos (vazamento)

Variáveis definidas **depois** da decisão não entram como preditoras na negação: `denial_reason-*`, `interest_rate`, `rate_spread`, `total_loan_costs`, `origination_charges`, `discount_points`, `lender_credits`, `loan_term` (ausente para negados) e `purchaser_type`.

## 4. Critérios de filtragem da amostra

Cada filtro é aplicado em sequência. O número de linhas removidas em cada passo é registrado no fluxograma (issue #3).

| # | Filtro | Código HMDA | Justificativa |
|---|---|---|---|
| 1 | Decisão final de aprovação ou negação | `action_taken` ∈ {1, 3}; alvo `negado = 1` se 3 | Exclui desistências, pedidos incompletos e compras de carteira, que não refletem uma decisão de crédito. |
| 2 | Compra de imóvel | `loan_purpose = 1` | Refinanciamento e home equity têm outra dinâmica de risco. |
| 3 | Residência principal | `occupancy_type = 1` | Investimento e segunda moradia têm critérios de crédito diferentes. |
| 4 | Imóvel de 1 a 4 unidades | `total_units` ∈ {1, 2, 3, 4} | Padrão do crédito residencial (exclui multifamiliar). |
| 5 | Primeira hipoteca | `lien_status = 1` | Segunda hipoteca tem risco e precificação diferentes. |
| 6 | Sem finalidade comercial | `business_or_commercial_purpose = 2` | Mantém apenas crédito ao consumidor. |
| 7 | Sem linha de crédito aberta | `open_end_line_of_credit = 2` | Produto diferente do financiamento tradicional. |
| 8 | Sem hipoteca reversa | `reverse_mortgage = 2` | Produto específico para idosos, sem decisão comparável. |
| 9 | Sem co-candidato | `co_applicant_sex = 5` | Com co-candidato, o gênero do pedido fica ambíguo. |
| 10 | Gênero identificado | `derived_sex` ∈ {Male, Female} | Exclui `Joint` e `Sex Not Available`. |

**Valores faltantes** (renda, `debt_to_income_ratio`, `combined_loan_to_value_ratio`, incluindo `NA` e `Exempt`):

- **Base principal:** apenas casos completos.
- **Base de robustez:** imputação (mediana por estrato) **e** indicador de faltante, para avaliar se a exclusão afeta os resultados (issue #12).
- A taxa de ausência por gênero e por `action_taken` será reportada, pois a ausência provavelmente **não é aleatória** (por exemplo, DTI pode faltar mais em pedidos negados).

**Codificação de variáveis:** `debt_to_income_ratio` é publicado em faixas no HMDA e será tratado como **categórico**. `income` está em milhares de US$ e será usado em log.

## 5. Compromisso de transparência

- Os resultados serão **reportados integralmente, inclusive se não confirmarem as hipóteses** ou se forem nulos.
- As especificações descritas aqui são as **principais**. Análises adicionais serão identificadas como **exploratórias**.
- Todo o código, a versão dos dados e a data de download ficarão registrados no repositório.

## 6. Limitações conhecidas a priori

- O HMDA público **não contém credit score**, histórico de crédito, patrimônio nem inadimplência. Por isso, viés de variável omitida é possível.
- `derived_sex` é **autodeclarado ou registrado pela instituição** e é binário na análise. Outras identidades de gênero não são representadas.
- Os resultados são **associações observacionais**, sem interpretação causal.
- O alvo (negação) reflete a decisão da instituição, então o modelo preditivo aprende o comportamento histórico, não o risco real.

---

## 7. Registro de alterações

Decisões tomadas durante a Etapa B (H1) que o protocolo original não previa ou que especificam algo que ele deixou em aberto. Todas foram tomadas **depois** de ver os dados e, portanto, são **pós-hoc**. Quando há impacto sobre a amostra, os números vêm de `resultados/h1_fluxo_amostra.csv`.

| # | Data | Alteração | Justificativa | Impacto |
|---|---|---|---|---|
| 1 | 2026-11-26 | Linhas com `income` <= 0 saem das especificações principais e das sensibilidades S1 e E1. | O protocolo (seção 4) manda usar `income` em log, que é indefinido para renda <= 0. | 7.111 linhas (6.644 com renda zero e 467 negativas); 65,1% negadas. Em S2 elas são mantidas, com indicador `renda_nao_pos` e log(renda) = 0. |
| 2 | 2026-11-26 | `combined_loan_to_value_ratio` é winsorizado nos percentis 0,1 e 99,9 (limites [13,99; 115,52]), calculados após a exclusão do item 1. Entra no modelo de forma linear. | Há valores incompatíveis com razão empréstimo/valor (máximo de 29,5 milhões de %; 168 linhas acima de 200%, quase todas negadas), que dominariam um termo linear. O protocolo não define o tratamento do LTV. | Nenhuma linha removida. |
| 3 | 2026-11-26 | Linhas com `state_code` ou `applicant_age` faltante saem das especificações principais. | Os efeitos fixos de estado exigem `state_code`. A idade só é usada na exploratória E1, e a exclusão mantém a mesma amostra entre as especificações. O protocolo trata apenas renda, DTI e LTV como casos completos. | `state_code`: 7.167 linhas (7.195 antes do item 1), das quais 96,6% negadas e 50,8% mulheres. `applicant_age`: 83 linhas (130 antes dos itens anteriores), 80,7% negadas. **A ausência não é aleatória**; ver S1 e S2. |
| 4 | 2026-11-26 | Categorias de referência: DTI `<20%`, tipo de empréstimo 1 (convencional) e, em E1, faixa etária 25-34. | Escolha de parametrização, que o protocolo não definiu. | Não altera o coeficiente de gênero, só a leitura dos coeficientes dos controles. |
| 5 | 2026-11-26 | O AME é calculado pelo método dos contrafactuais, com IC por método delta e vcov clusterizada. Nas especificações com efeitos fixos, os efeitos fixos entram como conhecidos (IC aproximado). Foi adicionado um modelo linear de probabilidade (MPL) com os mesmos controles e efeitos fixos como verificação. | O protocolo (seção 3.1) pede AME com erros clusterizados, mas não define o cálculo com efeitos fixos. | Na validação sem efeitos fixos, o resultado é idêntico ao do `statsmodels`. Com efeitos fixos, o MPL difere do AME do logit em no máximo 0,05 p.p. no ponto estimado. |
| 6 | 2026-11-26 | Estimação com `pyfixest` 0.70.0 (efeitos fixos absorvidos). O estimador descarta singletons e instituições/estados sem variação no desfecho (separação). | Inviável incluir cerca de 2.400 dummies de instituição em 1,5 milhão de linhas. | Especificações (4), (5) e E1: 89 singletons e 18.681 linhas por separação; permanecem 1.817 das 2.435 instituições (N = 1.514.115). As especificações (1) a (3) usam 1.532.885 linhas. |
| 7 | 2026-11-26 | Análises **exploratórias** adicionadas e identificadas como tais na tabela: **S1** (especificação 4 sem exigir `state_code`/`applicant_age`), **S2** (especificação 5 com todas as linhas: estado ausente como categoria própria e indicador de renda <= 0) e **E1** (especificação 5 + faixa etária). | S1 e S2 avaliam se as exclusões dos itens 1 e 3 afetam a conclusão. E1 avalia a sensibilidade a um controle que não está em H1. S2 é diferente da base de robustez com imputação da seção 4 (issue #12), que continua pendente. | S1: OR 1,017 (0,992; 1,044). S2: OR 1,006 (0,981; 1,032). E1: OR 0,974 (0,948; 1,000), p = 0,050. Especificação principal (5): OR 1,005 (0,980; 1,032), p = 0,678. |

| 8 | 2026-11-26 | Os scripts das Etapas B e D foram convertidos para os notebooks `04_regressao_logistica.ipynb` e `05_gradient_boosting.ipynb`, seguindo o formato das aulas. Na Etapa B, o efeito em p.p. passa a ser reportado pelo MPL (com IC clusterizado), sem o cálculo do AME por método delta; as sensibilidades S1 e S2 saem do notebook e passam para a issue #12 (robustez). Na Etapa D, os hiperparâmetros passam a ser escolhidos por `GridSearchCV` (`scoring='roc_auc'`, 3 folds, amostra de 100 mil pedidos do treino), e foi adicionada uma regressão logística como modelo de comparação. | Padronizar o projeto no formato dos notebooks da disciplina. O MPL difere do AME em no máximo 0,05 p.p. (item 5). | Etapa B: resultados idênticos aos do script (especificação 5: OR 1,005; p = 0,678). Etapa D: `learning_rate = 0.05`, `max_leaf_nodes = 15`; AUC 0,787 (antes 0,789). |
| 9 | 2026-11-26 | Robustez (`08_robustez.ipynb`, issue #12): especificação (5) com controles extras (raça, etnia, valor do empréstimo, casa pré-fabricada, bairro), por subgrupo (tipo de empréstimo, terço de renda, região do censo) e na base com faltantes preenchidos; modelo de crédito sem gênero com as variáveis extras (sem raça). Comparação com HMDA 2024 não realizada. | Verificar se as conclusões dependem de variáveis omitidas, de subgrupos ou do tratamento de faltantes. Análises **exploratórias**; nos subgrupos, considerar correção para comparações múltiplas (Bonferroni, α = 0,005). | Com raça e etnia: OR 0,963 (0,942; 0,985), p = 0,001. Subgrupos significativos a 5%: FHA (1,054), VA (1,088), renda alta (1,044); só FHA passa em Bonferroni. Base com faltantes preenchidos: OR 1,023, p = 0,22. Modelo ampliado: AUC 0,808; diferença nas decisões 3,04 p.p. (antes 3,66). |

**Reprodutibilidade:** código nos notebooks da pasta `notebooks/` (sementes fixadas em 42) e tabelas de resultados em `resultados/`. Os dados foram baixados em 23/11/2026 (ver `docs/dados/integridade_2025.md`).
