# Pacotes utilizados no projeto

Este documento resume os principais pacotes usados no projeto Amazon Sales Data Science e a função de cada um no pipeline.

## Manipulação e processamento de dados

| Pacote | Função no projeto |
|---|---|
| `pandas` | Leitura de arquivos CSV, tratamento de tabelas, agregações, criação de variáveis derivadas e salvamento de outputs tabulares. |
| `numpy` | Operações numéricas, criação de `log_rating_count` com `np.log1p` e apoio a tratamento de valores ausentes. |

## Visualização

| Pacote | Função no projeto |
|---|---|
| `matplotlib` | Criação e salvamento das figuras em `outputs/figures/`. |
| `seaborn` | Gráficos estatísticos, como histogramas, boxplots, scatterplots, heatmap de correlação e matrizes de confusão. |
| `wordcloud` | Geração de nuvem de palavras para leitura visual dos termos mais recorrentes nos textos. |

## Modelagem e aprendizado de máquina

| Pacote | Função no projeto |
|---|---|
| `scikit-learn` | Pipelines de preprocessamento, imputação, normalização, One-Hot Encoding, separação treino/teste, métricas, Regressão Linear, Regressão Logística, Random Forest, K-Means, Hierarchical Clustering, DBSCAN e PCA. |
| `xgboost` | Modelos XGBoost Regressor e XGBoost Classifier para predição e classificação. |
| `lightgbm` | Modelos LightGBM Regressor e LightGBM Classifier para predição e classificação. |

## NLP

| Pacote | Função no projeto |
|---|---|
| `nltk` | Biblioteca disponível para evoluções futuras de NLP, como stopwords, tokenização e análise de sentimento leve quando necessário. |
| `scikit-learn` | Também é usado no NLP para geração de TF-IDF com `TfidfVectorizer`. |

## Coleta de dados

| Pacote | Função no projeto |
|---|---|
| `kaggle` | Download do Amazon Sales Dataset via Kaggle API, usando credenciais locais em `kaggle.json`. |

## Observações

- `xgboost` e `lightgbm` são necessários para a versão atual dos scripts de predição e classificação.
- `kaggle.json` deve ficar fora do repositório e nunca deve ser versionado.
- Dados brutos, dados processados e outputs pesados também permanecem fora do versionamento.
