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
