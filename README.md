# Customer Churn Prediction

Este projeto busca apresentar minha abordagem para um problema de predição de ‘churn’. Churn é utilizado em empresas para calcular a taxa de cancelamento ou desistência de clientes em um determinado período, o que leva a perda de receita. Sendo assim, fica claro que a capacidade de prever quais clientes possuem maior potenciais de cancelamento é uma informação importante para o negócio.

Irei segmentar meu estudo em três partes:

→ Análise Exploratória

Nesta etapa farei uso de conceitos e ferramentas estatísticas descritiva e inferência estatística para identificar informações que possam ser úteis para a compreensão dos grupos mais sucetíveis ao *churn.*

→ Modelagem e Machine Learning

Em seguida, utilizaremos os conhecimentos obtidos sobre os dados até então para buscar um modelo capaz de prever os clientes que irão realizar o cancelamento do plano.

→ Conclusões:

Por fim, o modelo obtido é apresentado com seus parâmetros de performance e recomendações.

# Dos Dados *(Training set)*

---

Os dados utilizados possuem 4250 observações. Estão disponíveis no Kaggle (Customer Churn Prediction 2020).

### Atributos (*features)*

**state**, string. 2-letter code of the US state of customer residence

**account_length**, numerical. Number of months the customer has been with the current telco provider

**area_code**, string="area_code_AAA" where AAA = 3 digit area code.

**international_plan**, (yes/no). The customer has international plan.

**voice_mail_plan**, (yes/no). The customer has voice mail plan.

**number_vmail_messages**, numerical. Number of voice-mail messages.

**total_day_minutes**, numerical. Total minutes of day calls.

**total_day_calls**, numerical. Total minutes of day calls.

**total_day_charge**, numerical. Total charge of day calls.

**total_eve_minutes**, numerical. Total minutes of evening calls.

**total_eve_calls**, numerical. Total number of evening calls.

**total_eve_charge,** numerical. Total charge of evening calls.

**total_night_minutes**, numerical. Total minutes of night calls.

**total_night_calls**, numerical. Total number of night calls.

**total_night_charge**, numerical. Total charge of night calls.

**total_intl_minutes**, numerical. Total minutes of international calls.

**total_intl_calls,** numerical. Total number of international calls.

**total_intl_charge**, numerical. Total charge of international calls

**number_customer_service_calls**, numerical. Number of calls to customer service

**churn**, (yes/no). Customer churn - target variable.

---

# → Análise Exploratória

---

## **→ Estatística Descritiva**

O método  `train_dt.describe().T` retorna algumas estatísticas descritivas a respeito das variáveis contínuas presentes em nossos dados.

| --- | --- | --- | --- | --- | --- | --- | --- | --- |

**Inferências a partir da tabela:**

- Comparando as colunas **std**(desvio padrão), **75%**(3º Quartil) e **Max** podemos esperar a presença de outliers em todas variáveis explanatórias. Um dos parâmetros usuais é definir 3 desvios padrões como *trashold* para identificar os outliers. Utilizarei ele neste momento pois não requer cálculos adicionais de maior complexidade além da comparação dos valores já apresentados na tabela.
- Comparando as colunas **mean** (média) e **50%** (mediana) podemos identificar se a distribuição dos dados estão deslocados em relação a média. Chama a atenção as variáveis: *number_vmail_messages, number_customer_service_calls.*
- A coluna **count** nos leva a crer que não há valores faltantes em nossos dados. `train_dt.info()` pode nos dar maior certeza sobre isso.

## → Variáveis Categóricas

Os dados apresentam 4 variáveis categóricas: *states, area_code, international_plan, voice_mail_plan*.

Irei investigar a proporção de ‘churn’ em cada grupo limitado pela uma das variáveis categóricas.

→ **Gráficos de Barra - Proporção de Churn**

Quantidade de planos cancelados (churn = yes) em cada grupo de variável categórica.

![Untitled](Customer%20Churn%20Prediction%20c252cc84d52e4780b4ffa2116642caad/Untitled.png)

**→** Pela visualização dos gráficos podemos inferir:

- Os assinantes do plano internacional têm maior propensão de cancelamento comparado com os não assinantes.
- Essa propensão ao cancelamento se mostra inversa quando se trata dos adeptos do plano de caixa postal.
- O código de área 415 possui a maior concentração de clientes, logo um maior número de cancelamentos. Apesar disso, a primeira vista não parece haver maior proporção de churn positivo ao compararmos com a área 408 e 510.

**→ Churn por estados**

![Untitled](Customer%20Churn%20Prediction%20c252cc84d52e4780b4ffa2116642caad/Untitled%201.png)

→ Utilizei a divisão proposta pela agência *Bureau of Economic Analysis* para agrupar os estados de acordo com os dados econômicos de cada estado:

![Untitled](Customer%20Churn%20Prediction%20c252cc84d52e4780b4ffa2116642caad/Untitled%202.png)

**→ Proporção total de churn**

![Untitled](Customer%20Churn%20Prediction%20c252cc84d52e4780b4ffa2116642caad/Untitled%203.png)

A proporção total de churn em nosso banco de dados é de 14,07%.

Adotando um nível de significância de 5% temos um intervalo de confiança para a proporção de churn entre: **13% a 15%.**

Como o que nos preocupa é o limite superior, ou seja, cenários onde o churn é mais elevado devemos dar maior atenção aos estados ou grupos onde a proporção é superior a 15%.

→ Podemos inferir que o grupo econômico Mideast definido pelo *Bureau of Economic Analysis* possui maior proporção de churn que a estimativa da proporção populacional.

→ Podemos seguir o mesmo raciocínio e realizar uma avaliação por estado.  Observamos que 17 dos 52 estados apresentam proporção de churn superior a estimativa populacional.

## → Variáveis Contínuas

### **→ Boxplot**

Com a análise das estatísticas descritivas pudemos inferir a existência de outliers em nossos dados. Vamos investigar um pouco mais por meio dos gráficos boxplot.

![Untitled](Customer%20Churn%20Prediction%20c252cc84d52e4780b4ffa2116642caad/Untitled%204.png)

Já conseguimos observar a presença de pontos além do terceiro quartil. Porém, devido os atributos possuirem escalas diferentes a análise pode ser comprometida. Irei então, padronizar os dados (em função da média e desvio padrão de cada coluna). 

![Untitled](Customer%20Churn%20Prediction%20c252cc84d52e4780b4ffa2116642caad/Untitled%205.png)

`Index(['account_length', 'number_vmail_messages', 'total_day_minutes','total_day_calls', 'total_day_charge', 'total_eve_minutes','total_eve_calls', 'total_eve_charge', 'total_night_minutes','total_night_calls', 'total_night_charge', 'total_intl_minutes','total_intl_calls', 'total_intl_charge','number_customer_service_calls'],dtype='object')`

Conforme esperado, pudemos verificar a existência de observações consideravelmente distantes do intervalo inter quartil (menores que Q1 e superiores a Q3).

→ Assumiremos que os valores que encontrarem-se acima  do terceiro quartil ou abaixo do primeiro quartil em 1,5 vezes o desvio padrão são outliers e serão removidos. 

![Untitled](Customer%20Churn%20Prediction%20c252cc84d52e4780b4ffa2116642caad/Untitled%206.png)

### → Análise de Correlações

Segue mapa de calor das correlações entre as variáveis, podemos ver a existência de uma correlação muito forte entre minutos e cobranças.

![Untitled](Customer%20Churn%20Prediction%20c252cc84d52e4780b4ffa2116642caad/Untitled%207.png)

**→ Scatterplot** gráfico de dispersão entre variáveis com forte correlação.

![Untitled](Customer%20Churn%20Prediction%20c252cc84d52e4780b4ffa2116642caad/Untitled%208.png)

## → Verificação de Hipóteses

**→** sobre o plano internacional:

$H_0$→ Hipótese nula: Não há diferença entre a proporção de churn entre assinantes e não assinantes do plano internacional.

$\alpha$ → Nível de significância de 5%

→ Assumindo dois grupos independentes (assinantes do plano internacional e não assinantes)

→ Quantidade de amostras suficientemente grandes para assumir normalidade.

**Estatística de Teste:**

`p_intl: 0.422
p_no_intl: 0.112
p_d: 0.31
n_intl: 396
n_no_intl: 3854
Z statistic: 12.698395237305308
p-value: 0.0`

→ Z statistic = 12.7 significa que o valor da diferença entre a proporção das amostras das duas populações está 12,7 std acima da valor esperado (zero) considerando a hipótese nula como verdadeira.

→ Conclusão do teste de hipótese: Rejeitamos a hipótese nula. P-value < 5%

---

# →Modelagem e Machine Learning

---

## Engenharia de variáveis

Transformaremos as variáveis categóricas em numéricas. Os outliers serão removidos pelo critério apresentado anteriormente.

Criei novas variáveis para os valores totais de minutos e cobranças (total_minutes e total_charge). Temos uma correlação moderada entre essas novas variáveis e a variável alvo! Abaixo temos uma régua de valores absolutos de correlação.

![Untitled](Customer%20Churn%20Prediction%20c252cc84d52e4780b4ffa2116642caad/Untitled%209.png)

Reduzirei a quantidade de variáveis, mantendo apenas as correlações mais relevantes.

## → Critério de Avaliação do Modelo

→ Definir os critérios de avaliação do modelo. Utilizarei algumas métricas de avaliação de classificação. 

- Recall (sensibilidade): Porcentagem de churn positivos que foram identificados corretamente.
- Precision: Porcentagem de clientes identificados como churn positivo estão corretas.
- F1: Média harmônica entre Precision e Recall
- Accuracy: Porcentagem de acertos na previsão.
- Matriz de confusão: Não se trata de um parâmetro, mas uma forma de representar quantidade de error tipo I e II.

→ Definir Modelo Base: servirá como referência para os modelos criados.

- Este modelo irá prever churn de todos os assinantes do plano internacional. Embora esse modelo seja muito simplório, ele produzirá uma acuracidade próxima próxima a 85% pois acertará todas as previsões quando não houver churn nem adesão do plano internacional e quando houver churn de clientes do plano internacional.

```python
# Modelo Base
y_pred = X_train.iloc[:,2]
```

![Untitled](Customer%20Churn%20Prediction%20c252cc84d52e4780b4ffa2116642caad/Untitled%2010.png)

## Modelo 1: Regressão Logística

---

O primeiro modelo a ser testado para realizar a previsão de churn é a Regressão Logística.

→ Neste novo modelo tivemos uma maior acurácia, ou seja, acertarmos mais previsões. Porém isso pode não ser muito significativo se estivermos acertando mais vezes em que não ocorrem o churn.

→ A métrica Recall (sensibilidade) caiu pela metade. Isso significa que o modelo está deixando de detectar churn’s reais.

→ Por outro lado, a precisão do modelo está mais elevada. Ou seja, 64.7% das vezes que o modelo acusou Churn ele previu corretamente.

→ F1 pende para o menor valor, como o modelo está mais desbalanceado entre Recall e Precisão essa métrica diminuiu.

**Matriz de Confusão - Churn**

![Untitled](Customer%20Churn%20Prediction%20c252cc84d52e4780b4ffa2116642caad/Untitled%2011.png)

![Untitled](Customer%20Churn%20Prediction%20c252cc84d52e4780b4ffa2116642caad/Untitled%2012.png)

---

## Modelo 2: K-Neighbors Classifier

---

Neste modelo, KNN, conseguimos melhorar um pouco nossas previsões.

→ A melhora da métrica de acurácia do modelo foi muito baixa. Embora o valor final seja superior a 90%. Relembrando que este parâmetro isoladamente não é suficiente para avaliar esse tipo de situação onde os dados estão desbalanceados.

→ O Recall foi a métrica que apresentou uma melhora relevante em relação ao modelo de regressão linear. Ou seja, este modelo é capaz de identificar 40% dos churn reais.

→ A precisão teve melhora, passou para o patamar de 68% de acerto em suas indicações de churn.

→ A forte evolução de Recall aproximou os valores das métricas (precisão x recall), isso refletiu numa melhora em F1.

**Matriz de Confusão - Churn**

![Untitled](Customer%20Churn%20Prediction%20c252cc84d52e4780b4ffa2116642caad/Untitled%2013.png)

![Untitled](Customer%20Churn%20Prediction%20c252cc84d52e4780b4ffa2116642caad/Untitled%2014.png)

---

### Modelo 3: Random Forest Classifier

---

O terceiro modelo utilizado foi o Random Forest. Com excelentes resultados.

→ Acurácia deste modelo é de aproximadamente 97%.

→ A métrica Recall indica que o modelo detecta 71% dos churns.

→ A métrica de precisão indica que o modelo acerta em 99% das vezes que prevê churn.

→ F1 mostra que os dois parâmetros estão elevados.

**Matriz de Confusão - Churn**

![Untitled](Customer%20Churn%20Prediction%20c252cc84d52e4780b4ffa2116642caad/Untitled%2015.png)

![Untitled](Customer%20Churn%20Prediction%20c252cc84d52e4780b4ffa2116642caad/Untitled%2016.png)

---

# → Conclusões# Churn_Prediction
