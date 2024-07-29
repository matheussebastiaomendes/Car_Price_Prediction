
# Modelo Preditivo para preços de veículos 🚗


Neste projeto será feito uma análise exploratoria dos dados e a elaboração de um modelo preditivo feito com LGBMODEL e RANDOM FOREST, os dados podem ser encontrados no [Kaggle](https://www.kaggle.com/datasets/hellbuoy/car-price-prediction/data) e foram disponibilizados por [Manish Kumar](https://www.kaggle.com/hellbuoy).


### Problema de negócio


A empresa Toad Motors , especializada na venda de veículos, está buscando entender melhor o mercado de carros que comercializa. A empresa possui um vasto conjunto de dados sobre diferentes marcas e modelos de carros, incluindo informações como preço, potência, consumo urbano, comprimento, largura e tipo de carroceria. O objetivo é usar esses dados para otimizar suas estratégias de precificação e melhorar a oferta de produtos.

![](https://github.com/matheussebastiaomendes/ml_lgbmodel/blob/main/imagens/carros.png)

### Objetivos e resultados
O primeiro objetivo é responder as seguintes perguntas sobre o dataset:

- Qual é a quantidade de carros por marca?
- Quais as marcas que possuem a maior media de preço de carros?
- Como o preço se comporta em relação a potencia, consumo urbano, comprimento e largura do carro?
- O tipo de carroceria influencia no preço do veiculo?

Já a predição dos preços tem por objetivo desenvolver e utilizar um modelo de previsão de preço para estimar o valor de mercado dos carros com base nas suas características. Isso permitirá a empresa ajustar os preços dos veículos de forma mais precisa e competitiva, além de ajudar na avaliação de novos modelos e marcas.


### 🛠️ Ferramentas utilizadas
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)

## A estrutura do dataset

A colunas do dataset estão organizadas da seguinte forma:

| Coluna                       | Descrição                                                       |
|------------------------------|-----------------------------------------------------------------|
| nome_do_Carro                | Nome completo do carro                                         |
| marca                        | Nome da marca do carro                                         |
| tipo_de_combustível           | Tipo de combustível usado pelo carro                            |
| aspiração                    | Tipo de aspiração do motor (por exemplo, aspirado, turbo)        |
| número_de_portas             | Número de portas do carro                                       |
| tipo_de_carroceria           | Tipo de carroceria (por exemplo, sedan, SUV, hatchback)         |
| rodas_motrizes               | Tipo de tração (por exemplo, tração dianteira, tração integral) |
| localização_do_motor         | Localização do motor no carro (por exemplo, dianteiro, traseiro) |
| distância_entre_eixos        | Distância entre os eixos do carro                               |
| comprimento_do_carro         | Comprimento total do carro                                      |
| largura_do_carro             | Largura total do carro                                          |
| altura_do_carro              | Altura total do carro                                           |
| peso_em_ordem_de_marcha      | Peso do carro em ordem de marcha                                |
| tipo_de_motor                | Tipo de motor (por exemplo, V6, V8)                             |
| número_de_cilindros          | Número de cilindros no motor                                    |
| tamanho_do_motor             | Tamanho do motor                                                |
| sistema_de_combustível        | Sistema de fornecimento de combustível (por exemplo, injeção)   |
| diâmetro_do_cilindro         | Diâmetro dos cilindros do motor                                 |
| curso_do_pistão              | Curso dos pistões do motor                                      |
| taxa_de_compressão           | Taxa de compressão do motor                                     |
| potência                     | Potência do motor (em HP ou kW)                                 |
| rpm_máximo                   | Número máximo de rotações por minuto do motor                   |
| consumo_urbano               | Consumo de combustível em ambiente urbano (km/litro)            |
| consumo_rodoviário           | Consumo de combustível em estrada (km/litro)                    |
| preço                        | Preço do carro   


## Bibliotecas utilizadas

#### EDA (Análise Exploratória de Dados)
- Pandas, Numpy.
#### Visualizações Gráficas
- Seaborn, Matplotlib.
#### Machine Learning
- sklearn (scikit-learn)
- lightgbm

# Análsie exploratória de dados 

## Qual é a quantidade de carros por marca?

![](https://github.com/matheussebastiaomendes/ml_lgbmodel/blob/main/imagens/qtd_por_marca_top_10.png)

A marca que lidera com certa vantagem quando comparada a segunda colocada é a Toyota com 32 veiculos, seguida da Nissan com 18 e a Mazda com 17, as demais marcas ficaram ente 13 e 9 veiculos.

## Quais as marcas que possuem a maior média de preço de carros?

![](https://github.com/matheussebastiaomendes/ml_lgbmodel/blob/main/imagens/media_preco_marca_top_10.png).

O top 3 das marcas que possuem maior média de preço dos veículos são Jaguar, Buick e Porsche, ficando na média de R$ 31.000 á R$ 34.600, outra marca que possui uma média de valor consideravel quando comparada as outras é a BMW que fica em torno de R$ 26.100, as demais marcas ficam em uma faixa de preço entre R$ 15.000 á 18.000 .

## Como o preço se comporta em relação a potencia, consumo urbano, comprimento e largura do carro?

![](https://github.com/matheussebastiaomendes/ml_lgbmodel/blob/main/imagens/comportamento.png)

Analisando o gráfico podemos chegar a conclusão que o preço diminui conforme um maior consumo urbano e aumenta se potência, largura do carro e comprimento aumentam.

## O tipo de carroceria influencia no preço do veiculo?

![](https://github.com/matheussebastiaomendes/ml_lgbmodel/blob/main/imagens/comportamento_carroceria.png)

No dataset encontramos 5 tipos de carroceria, conversivel, hatchback, sedan, wagon (hatckback alongado) e hardtop.
Podemos analisar que os carros conversiveis estão em um patamar de preço mais elevado que as carrocerias mais encontradas no dataset, que possuem uma faixa de preços mais acessíveis devido à maior disponibilidade e variação de modelos, a carroceria hardtop possui um intervalo de preco maior

# Modelo de Prediçao

## Pré-processamento dos dados
As colunas númericas e categóricas foram separadas e tratadas, as colunas numéricas receberam um imputer de mediana da coluna analisada nos valores faltantes, além disso tambem foi feito standar scaler para a padronizacao dos dados para ter média zero e variância unitária. Já nas colunas categóricas utilizamos o OneHotEncoder para sua transformação.
## Feature selection
Após o tratamento das colunas utilizamos o SelectKBest para a selecao das features que tem maior impacto no modelo, para este modelo selecionamos 10.

## LGBMODEL
## Métricas
As métricas do modelo 

|Métrica|Resultado|
|------------------|-------------|
|RMSE|3378.7426|
|R² Score|0.8352|
|MAE|2221.6420|
|MSE|11415901.6063|
|MAPE|0.1566|

![](https://github.com/matheussebastiaomendes/modelo_predicao/blob/main/imagens/comportamento_modelo_LGBMODEL.png)

As previsões ficam com um erro maior conforme aumentamos o valor dos carros, testaremos outro modelo para analisar se os valores se adequam melhor a reta.

## RANDOM FOREST 

Para este modelo mantivemos o tratamento das colunas numericas e categoricas e incluimos alguns parametros do modelo Random Forest

As métricas do modelo 

|Métrica|Resultado|
|------------------|-------------|
|RMSE|2180.4615|
|R² Score|0.9314|
|MAE|1480.7826|
|MSE|4754412.1706|
|MAPE|0.1097|

![](https://github.com/matheussebastiaomendes/modelo_predicao/blob/main/imagens/comportamento_modelo_RANDOM_FOREST.png)

Neste segundo modelo podemos ver uma melhora nas métricas com erros menores comparada a antiga, no grafico vemos que os valores da previsão estão muito mais ajustados a reta, ficando bem proximos dos valores reais sendo assim esse é o modelo escolhido para realizar as previsões dos preços dos carros.


# Conclusoes

Ao analisar o dataset, respondemos aos questionamentos sobre preços de veículos e construímos um modelo preditivo de alta precisão. Identificamos que os carros conversíveis têm preços mais elevados devido ao seu luxo e engenharia sofisticada, as marcas com mais carro e as que possuem maior media de preco. Utilizando técnicas de aprendizado de máquina, nosso modelo previu com precisão os valores dos carros com base em suas características.



