
# Modelo Preditivo para preços de veículos 🚗


Neste projeto será feito uma análise exploratoria dos dados e a elaboração de um modelo preditivo feito com LGBMODEL, os dados podem ser encontrados no [Kaggle](https://www.kaggle.com/datasets/hellbuoy/car-price-prediction/data) e foram disponibilizados por [Manish Kumar](https://www.kaggle.com/hellbuoy).


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
| ID_do_carro                  | Identificador único do carro                                    |
| simbolização                 | Código ou símbolo que representa o carro                        |
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

#### Manipulação de dados
- Pandas, Numpy.
#### EDA
- Seaborn, Matplotlib.
#### Machine Learning
- XGBoost, sklearn.

# Análsie exploratória de dados 

## Qual é a quantidade de carros por marca?

## Quais as marcas que possuem a maior media de preço de carros?

## Como o preço se comporta em relação a potencia, consumo urbano, comprimento e largura do carro?

## O tipo de carroceria influencia no preço do veiculo?

