# Tarefas do Projeto

## Setup do projeto

- [x] Criar estrutura de pastas.
- [x] Criar `.gitignore`.
- [x] Criar `requirements.txt`.
- [x] Criar `docs/AGENTS.md`, `README.md`, `docs/SPEC.md` e `docs/TASKS.md`.

Critério de conclusão: estrutura inicial existe e arquivos de documentação estão preenchidos.

## Download dos dados

- [x] Configurar `kaggle.json` localmente fora do repositório.
- [x] Executar download via Kaggle API.
- [x] Confirmar presença de arquivo `.csv` em `data/raw/`.

Critério de conclusão: arquivo bruto está disponível localmente e não versionado.

## Tratamento

- [x] Ler arquivo bruto automaticamente de `data/raw/`.
- [x] Padronizar nomes de colunas.
- [x] Converter preços, desconto, rating e `rating_count`.
- [x] Separar categorias hierárquicas.
- [x] Criar variáveis derivadas obrigatórias.
- [x] Salvar `data/processed/amazon_sales_processed.csv`.

Critério de conclusão: base tratada salva sem erro e com variáveis derivadas.

## Análise exploratória

- [x] Gerar resumo geral.
- [x] Gerar tabela de missing values.
- [x] Gerar rankings por categoria e produto.
- [x] Gerar distribuições e gráficos de relação.
- [x] Gerar heatmap de correlação.

Critério de conclusão: tabelas e figuras principais salvas em `outputs/`.

## Predição

- [x] Definir `log_rating_count` como alvo.
- [x] Criar pipeline de preprocessamento.
- [x] Treinar Regressão Linear e Random Forest.
- [x] Treinar XGBoost e LightGBM se instalados.
- [x] Salvar métricas comparativas.
- [x] Salvar gráfico observado vs previsto.

Critério de conclusão: `outputs/tables/prediction_metrics.csv` e figura do melhor modelo existem.

## Classificação

- [x] Definir `popular_product` como alvo.
- [x] Criar pipeline de preprocessamento.
- [x] Treinar Regressão Logística e Random Forest.
- [x] Treinar XGBoost e LightGBM se instalados.
- [x] Salvar métricas comparativas.
- [x] Salvar matriz de confusão do melhor modelo.
- [x] Salvar importância de variáveis quando disponível.

Critério de conclusão: métricas, matriz de confusão e, quando possível, importâncias estão salvas.

## Clusterização

- [x] Selecionar variáveis numéricas.
- [x] Normalizar variáveis.
- [x] Aplicar K-Means.
- [x] Aplicar Hierarchical Clustering.
- [x] Aplicar DBSCAN.
- [x] Gerar PCA 2D.
- [x] Salvar base com labels e tabela-resumo.

Critério de conclusão: base clusterizada, tabela-resumo e gráfico PCA estão salvos.

## NLP

- [x] Limpar textos de reviews e descrições.
- [x] Contar palavras.
- [x] Gerar frequência de termos.
- [x] Gerar TF-IDF simples.
- [x] Comparar alto/baixo rating.
- [x] Comparar popular/não popular.
- [x] Salvar tabelas e gráficos.

Critério de conclusão: outputs textuais estão salvos em `outputs/`.

## Documentação

- [x] Atualizar README com instruções reais de execução.
- [x] Atualizar SPEC caso o escopo mude.
- [x] Atualizar TASKS conforme conclusão das etapas.

Critério de conclusão: documentação reflete o estado atual do projeto.

## Revisão final

- [x] Checar sintaxe dos scripts.
- [x] Confirmar que dados e credenciais não estão versionados.
- [x] Confirmar que outputs vão para as pastas esperadas.
- [x] Revisar premissas metodológicas.

Critério de conclusão: projeto pronto para download da base e execução do pipeline inicial.
