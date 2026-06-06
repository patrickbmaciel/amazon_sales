# Relatório Executivo - Amazon Sales Data Science

## Base de dados

Este relatório resume os resultados do pipeline de Data Science aplicado ao Amazon Sales Dataset. A base tratada contém 1.465 produtos e 29 colunas após limpeza, padronização de tipos e criação de variáveis derivadas.

A base não contém vendas reais, receita, quantidade vendida nem histórico temporal. Por isso, `rating_count` foi usado como proxy de popularidade, engajamento ou demanda aparente. Já `rating` foi interpretado como proxy de satisfação média dos consumidores.

| Indicador | Valor |
|---|---:|
| Produtos analisados | 1.465 |
| Categorias principais | 9 |
| Rating médio | 4,10 |
| Mediana de `rating_count` | 5.178 |
| Média de `rating_count` | 18.270,56 |
| Produtos classificados como populares | 25,1% |

## Análise exploratória

A análise exploratória gerou tabelas de resumo, valores ausentes, rankings de categorias/produtos, distribuições, boxplots, scatterplots e heatmap de correlação. O heatmap usa azul para correlações positivas fortes e vermelho para correlações negativas.

Categorias com mais produtos:

| Categoria | Produtos |
|---|---:|
| Electronics | 526 |
| Computers&Accessories | 453 |
| Home&Kitchen | 448 |
| OfficeProducts | 31 |
| MusicalInstruments | 2 |

Categorias com maior mediana de `rating_count`:

| Categoria | Mediana de `rating_count` | Participação de produtos populares |
|---|---:|---:|
| MusicalInstruments | 44.441 | 100,0% |
| Toys&Games | 15.867 | 0,0% |
| Electronics | 10.689 | 39,0% |
| Computers&Accessories | 7.732 | 27,4% |
| OfficeProducts | 4.426 | 0,0% |

Relações observadas:

- A correlação entre `discount_percentage` e `rating_count` foi praticamente nula (0,011), indicando que desconto maior, isoladamente, não explica popularidade aparente.
- A correlação entre `discount_percentage` e `log_rating_count` foi levemente negativa (-0,112), sugerindo que descontos mais altos não necessariamente acompanham maior engajamento.
- A correlação entre `rating` e `log_rating_count` foi positiva, mas moderada (0,233), indicando que satisfação média ajuda, mas não explica sozinha a popularidade aparente.
- Faixas premium apresentaram maior participação de produtos populares (45,9%), mas esse resultado deve ser lido como padrão da base, não como evidência de vendas.

## Predição

Objetivo: prever `log_rating_count`, usado como proxy de popularidade aparente.

Modelos usados:

- Regressão Linear.
- Random Forest Regressor.
- XGBoost Regressor.
- LightGBM Regressor.

Resultados:

| Modelo | R² | MAE | RMSE | MAPE |
|---|---:|---:|---:|---:|
| Random Forest Regressor | 0,516 | 1,074 | 1,450 | 0,165 |
| XGBoost Regressor | 0,460 | 1,200 | 1,532 | 0,179 |
| LightGBM Regressor | 0,459 | 1,123 | 1,534 | 0,168 |
| Regressão Linear | 0,348 | 1,298 | 1,683 | 0,189 |

O melhor modelo foi o Random Forest Regressor. Isso indica que relações não lineares entre preço, desconto, categoria, rating e atributos textuais simples ajudam a explicar parte da popularidade aparente.

## Classificação

Objetivo: classificar produtos como populares ou não populares. A variável `popular_product` foi definida a partir do percentil 75 de `rating_count`.

Modelos usados:

- Regressão Logística.
- Random Forest Classifier.
- XGBoost Classifier.
- LightGBM Classifier.

Resultados:

| Modelo | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Random Forest Classifier | 0,877 | 0,878 | 0,589 | 0,705 |
| LightGBM Classifier | 0,863 | 0,800 | 0,603 | 0,688 |
| XGBoost Classifier | 0,853 | 0,788 | 0,562 | 0,656 |
| Regressão Logística | 0,785 | 0,609 | 0,384 | 0,471 |

O melhor classificador foi o Random Forest Classifier. O LightGBM ficou próximo em F1-score e apresentou recall ligeiramente maior que o Random Forest, o que pode ser útil caso o objetivo futuro seja capturar mais produtos potencialmente populares, mesmo com alguma perda de precisão.

Variáveis importantes na classificação incluem tamanho do conteúdo da review, preço com desconto, tamanho do nome do produto, tamanho da descrição, valor do desconto, preço original, tamanho do título da review, percentual de desconto e rating.

## Clusterização

Objetivo: segmentar produtos em grupos interpretáveis usando preço, desconto, rating, `rating_count` e `log_rating_count`.

Métodos usados:

- K-Means.
- Hierarchical Clustering.
- DBSCAN.
- PCA 2D para visualização dos três métodos.

Foram geradas figuras específicas:

- `outputs/figures/cluster_pca_kmeans.png`
- `outputs/figures/cluster_pca_hierarchical.png`
- `outputs/figures/cluster_pca_dbscan.png`

Resumo dos clusters K-Means:

| Cluster | Produtos | Mediana de preço com desconto | Mediana de desconto (%) | Mediana de rating | Mediana de `rating_count` | Perfil sugerido |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 534 | 399,00 | 60 | 4,0 | 935 | niche_candidate |
| 1 | 33 | 32.999,00 | 38 | 4,3 | 3.837 | niche_candidate |
| 2 | 719 | 849,00 | 41 | 4,2 | 10.725 | mixed_profile |
| 3 | 132 | 13.244,50 | 33 | 4,2 | 11.591 | popular_candidate |
| 4 | 47 | 709,00 | 54 | 4,1 | 178.912 | popular_candidate |

O cluster 4 se destaca pela mediana extremamente alta de `rating_count`, sinalizando produtos com popularidade aparente muito forte. O cluster 0 tem alta mediana de desconto e baixa mediana de `rating_count`, sugerindo produtos que podem exigir revisão de posicionamento, conteúdo ou atratividade.

## NLP

Objetivo: extrair padrões textuais simples a partir de `review_title`, `review_content` e `about_product`.

Técnicas usadas:

- Limpeza básica de texto.
- Frequência de termos.
- Termos distintivos por diferença de frequência relativa.
- TF-IDF geral e por grupo.
- Comparação de textos por popularidade e por rating alto/baixo.
- Nuvem de palavras e boxplots de quantidade de palavras.

Termos mais frequentes:

| Termo | Contagem |
|---|---:|
| good | 10.451 |
| product | 6.818 |
| not | 4.658 |
| you | 4.343 |
| but | 4.314 |

Termos distintivos em produtos populares:

| Termo | Leitura |
|---|---|
| phone | Mais associado aos produtos populares. |
| watch | Mais associado aos produtos populares. |
| battery | Mais associado aos produtos populares. |
| camera | Mais associado aos produtos populares. |

Termos distintivos em produtos com rating baixo:

| Termo | Leitura |
|---|---|
| not | Indica maior presença de negação ou frustração. |
| remote | Aparece mais em produtos com rating baixo. |
| noise | Sinaliza possível fricção de experiência. |
| call | Pode indicar problemas em produtos de áudio/comunicação. |
| bass | Pode indicar expectativa frustrada em áudio. |

Os resultados de NLP indicam que produtos populares estão fortemente conectados a temas de eletrônicos e uso prático, enquanto produtos de rating mais baixo concentram termos com sinais de problemas ou fricções específicas.

## Respostas às perguntas de negócio

1. **Quais categorias concentram produtos mais populares?**  
   `Electronics` e `Computers&Accessories` concentram grande parte dos produtos e têm medianas relevantes de `rating_count`. `Electronics` se destaca também pela participação de produtos populares.

2. **Produtos com maior desconto tendem a ter maior `rating_count`?**  
   Não de forma clara. A correlação entre desconto percentual e `rating_count` foi praticamente nula, e com `log_rating_count` foi levemente negativa.

3. **Produtos mais caros recebem avaliações melhores ou piores?**  
   A relação é fraca, mas positiva. A correlação entre preço original e rating foi 0,122, sugerindo que preço mais alto não está associado à pior avaliação.

4. **Quais produtos combinam alto rating e alto número de avaliações?**  
   Há destaque para produtos como cabos Amazon Basics, SSDs, acessórios de computador e alguns itens de casa/cozinha com rating alto e `rating_count` elevado.

5. **Quais produtos têm alto rating, mas baixo `rating_count`?**  
   Produtos com rating 4,7 a 5,0 e baixo `rating_count`, como acessórios e pequenos eletrodomésticos, aparecem como candidatos a ativação, pois têm boa satisfação, mas pouca popularidade aparente.

6. **Quais produtos têm alto `rating_count`, mas rating baixo?**  
   Foram encontrados 51 produtos populares com rating abaixo de 4,0. Eles aparecem principalmente em eletrônicos e acessórios, incluindo fones, earbuds, teclados e impressoras.

7. **É possível prever a popularidade aparente de um produto?**  
   Sim, parcialmente. O melhor modelo de regressão, Random Forest Regressor, atingiu R² de 0,516 para `log_rating_count`.

8. **É possível classificar produtos como populares ou não populares?**  
   Sim. O Random Forest Classifier atingiu accuracy de 0,877 e F1-score de 0,705.

9. **Quais segmentos aparecem a partir de clusterização?**  
   Foram identificados grupos com perfis de produtos populares, nichados, mistos e produtos de maior preço. O cluster 4 concentrou os produtos de maior popularidade aparente.

10. **O conteúdo textual dos reviews ajuda a explicar padrões de satisfação ou popularidade?**  
   Sim. Termos como `phone`, `watch`, `battery` e `camera` aparecem associados a popularidade, enquanto `not`, `noise`, `call`, `remote` e `bass` aparecem mais ligados a rating baixo.

## Insights acionáveis

- **Priorizar análise comercial e curadoria em `Electronics` e `Computers&Accessories`.** Essas categorias combinam alto volume de produtos e maior popularidade aparente, sendo boas candidatas para ações de ranking, vitrine, campanhas e monitoramento competitivo.
- **Não usar desconto percentual como única alavanca de popularidade.** A relação entre desconto e `rating_count` foi fraca; descontos altos sem boa proposta de valor podem não gerar mais engajamento.
- **Criar lista de revisão para produtos populares com rating baixo.** Os 51 produtos populares com rating abaixo de 4,0 merecem investigação de qualidade, descrição, expectativa de uso, suporte ou problemas recorrentes em reviews.
- **Ativar produtos com alto rating e baixo `rating_count`.** Produtos bem avaliados, mas pouco populares, podem ser candidatos a melhoria de exposição, conteúdo, imagens, SEO interno ou campanhas pontuais.
- **Monitorar termos de fricção em reviews.** Termos como `not`, `noise`, `call`, `remote` e `bass` devem orientar análise qualitativa em produtos com rating baixo, especialmente em eletrônicos e áudio.
- **Usar modelos como triagem, não como decisão automática.** Random Forest teve melhor desempenho geral, mas o recall da classificação ainda mostra que parte dos produtos populares pode não ser capturada. O modelo deve apoiar priorização, não substituir análise humana.
- **Usar clusters para estratégias diferenciadas.** Produtos do cluster 4 podem ser tratados como populares consolidados; produtos do cluster 0 podem demandar revisão de posicionamento; produtos de preço mais alto devem ser analisados com critérios próprios de valor percebido.

## Principais resultados

- Random Forest Regressor foi o melhor modelo de predição.
- Random Forest Classifier foi o melhor modelo de classificação.
- XGBoost e LightGBM foram incorporados e executados nas etapas de predição e classificação.
- K-Means, Hierarchical Clustering e DBSCAN agora têm visualizações PCA separadas.
- A etapa de NLP passou a gerar análises comparativas por popularidade e satisfação.
- O projeto segue interpretando `rating_count` como proxy de popularidade aparente e `rating` como proxy de satisfação média, sem inferir vendas reais.
