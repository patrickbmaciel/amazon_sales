# Especificação do Projeto

## Objetivo

Construir um projeto Python de Data Science simples, eficiente, reprodutível e bem documentado para analisar produtos da Amazon com base em preço, desconto, categoria, rating, quantidade de avaliações e reviews.

## Contexto da base

A base utilizada será o Amazon Sales Dataset do Kaggle. Apesar do nome, ela não contém vendas reais, receita, quantidade vendida nem histórico temporal. Portanto, o projeto não deve tratar o dataset como uma base transacional de vendas.

`rating_count` será usado como proxy de popularidade, engajamento ou demanda aparente. `rating` será interpretado como proxy de satisfação média dos consumidores.

## Problema de negócio

O projeto busca responder:

1. Quais categorias concentram produtos mais populares?
2. Produtos com maior desconto tendem a ter maior `rating_count`?
3. Produtos mais caros recebem avaliações melhores ou piores?
4. Quais produtos combinam alto rating e alto número de avaliações?
5. Quais produtos têm alto rating, mas baixo `rating_count`?
6. Quais produtos têm alto `rating_count`, mas rating baixo?
7. É possível prever a popularidade aparente de um produto?
8. É possível classificar produtos como populares ou não populares?
9. Quais segmentos aparecem a partir de clusterização?
10. O conteúdo textual dos reviews ajuda a explicar satisfação ou popularidade?

## Inputs

- Arquivo CSV bruto baixado via Kaggle API em `data/raw/`.
- Campos esperados: produto, categoria, preços, desconto, rating, `rating_count`, descrição e reviews.

## Outputs esperados

- Base tratada: `data/processed/amazon_sales_processed.csv`.
- Tabelas analíticas e métricas em `outputs/tables/`.
- Gráficos em `outputs/figures/`.
- Base com labels de cluster em `data/processed/amazon_sales_clusters.csv`.

## Arquitetura

```text
.
|-- README.md
|-- docs/
|   |-- AGENTS.md
|   |-- SPEC.md
|   `-- TASKS.md
|-- data/
|   |-- raw/
|   `-- processed/
|-- src/
|   `-- functions/
`-- outputs/
    |-- figures/
    `-- tables/
```

## Descrição dos scripts

- `src/1_tratamento.py`: lê a base bruta, limpa tipos, cria variáveis derivadas e salva a base tratada.
- `src/2_analise_exploratoria.py`: cria tabelas e gráficos exploratórios.
- `src/3_predicao.py`: treina modelos de regressão para prever `log_rating_count`.
- `src/4_classificacao.py`: classifica produtos como populares ou não populares.
- `src/5_clusterizacao.py`: segmenta produtos com K-Means, Hierarchical Clustering, DBSCAN e PCA.
- `src/6_analise_nlp.py`: explora textos de reviews e descrições.

## Variáveis derivadas esperadas

- `discount_value = actual_price - discounted_price`
- `log_rating_count = np.log1p(rating_count)`
- `main_category`
- `sub_category_1`
- `sub_category_2`, se existir
- `product_name_length`
- `about_product_length`
- `review_title_length`
- `review_content_length`
- `high_rating`
- `popular_product`
- `price_range`
- `discount_range`

## Estratégia de análise exploratória

- Resumo geral da base.
- Missing values por coluna.
- Ranking de categorias por número de produtos.
- Ranking de categorias por rating médio.
- Ranking de categorias por média e mediana de `rating_count`.
- Top produtos por `rating_count`.
- Top produtos por desconto percentual.
- Distribuições de preço, desconto, rating e `rating_count`.
- Boxplot de preço por categoria principal.
- Scatterplots entre desconto/rating e `rating_count`.
- Heatmap de correlação.

## Estratégia de predição

- Alvo: `log_rating_count`.
- Separação treino/teste.
- Pipeline scikit-learn com preprocessamento numérico e categórico.
- Modelos: Regressão Linear, Random Forest Regressor, XGBoost Regressor opcional e LightGBM Regressor opcional.
- Métricas: R2, MAE, RMSE e MAPE.

## Estratégia de classificação

- Alvo: `popular_product`.
- `popular_product` será derivado do percentil 75 de `rating_count`.
- Modelos: Regressão Logística, Random Forest Classifier, XGBoost Classifier opcional e LightGBM Classifier opcional.
- Métricas: matriz de confusão, accuracy, precision, recall e F1-score.

## Estratégia de clusterização

- Variáveis principais: preços, desconto, rating, `rating_count` e `log_rating_count`.
- Normalização das variáveis numéricas.
- Métodos: K-Means, Hierarchical Clustering, DBSCAN e PCA 2D.
- Interpretação por perfis médios dos clusters, sem forçar nomes quando não houver evidência.

## Estratégia de NLP

- Campos: `review_title`, `review_content` e `about_product`.
- Limpeza básica de texto.
- Contagem de palavras.
- Frequência de termos.
- TF-IDF simples.
- Comparação entre alto/baixo rating e popular/não popular.
- Sentimento simples apenas se viável com biblioteca leve.

## Regras metodológicas

- Não interpretar `rating_count` como vendas reais.
- Não interpretar `rating` como demanda.
- Usar `np.log1p(rating_count)` para reduzir assimetria e evitar problemas com zero.
- Usar modelos opcionais somente quando as bibliotecas estiverem instaladas.
- Evitar tunagem complexa nesta primeira versão.

## Critérios de aceite

- Estrutura de pastas criada.
- Documentos iniciais completos.
- Scripts-base funcionais e executáveis individualmente.
- Funções auxiliares simples e reutilizáveis.
- `.gitignore` adequado.
- Nenhuma credencial versionada.
- Nenhum dado bruto/processado ou output pesado versionado.
- Comentários metodológicos claros sobre `rating_count` e `rating`.
