# Amazon Sales

Projeto em Python para analisar produtos da Amazon usando o dataset público do Kaggle [Amazon Sales Dataset](https://www.kaggle.com/datasets/karkavelrajaj/amazon-sales-dataset).

Este projeto não deve ser tratado como uma base transacional de vendas. O dataset não contém vendas reais, receita, quantidade vendida nem histórico temporal. A variável `rating_count` será usada como proxy de popularidade, engajamento ou demanda aparente. A variável `rating` será usada como proxy de satisfação média dos consumidores.

## Objetivo

Construir uma base simples, eficiente, reprodutível e bem documentada para:

- analisar categorias, preços, descontos, ratings e reviews;
- prever `log_rating_count`;
- classificar produtos como populares ou não populares;
- segmentar produtos por clusterização;
- explorar textos de reviews e descrições.

## Configuração do ambiente

Crie e ative um ambiente virtual:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

`xgboost` e `lightgbm` são necessários para executar todos os modelos de predição e classificação da versão atual. `nltk` e `wordcloud` apoiam evoluções e visualizações de NLP.

A descrição dos pacotes usados e suas funções está em `docs/PACKAGES.md`.

## Configuração da API do Kaggle

1. Acesse sua conta no Kaggle.
2. Crie um token em `Account > API > Create New Token`.
3. Salve o arquivo `kaggle.json` localmente em:

```text
C:\Users\<seu_usuario>\.kaggle\kaggle.json
```

4. Nunca versione `kaggle.json`. Ele já está incluído no `.gitignore`.

## Download dos dados

Com a API configurada, execute:

```bash
kaggle datasets download -d karkavelrajaj/amazon-sales-dataset -p data/raw --unzip
```

O script de tratamento localiza automaticamente um arquivo `.csv` em `data/raw/`.

Se a Kaggle API retornar erro relacionado a `username` ou `key`, revise o arquivo `kaggle.json`. Ele deve conter as chaves `username` e `key`, deve estar salvo fora do repositório e não deve ser versionado.

## Ordem de execução

Execute os scripts nesta ordem:

```bash
python src/1_tratamento.py
python src/2_analise_exploratoria.py
python src/3_predicao.py
python src/4_classificacao.py
python src/5_clusterizacao.py
python src/6_analise_nlp.py
```

## Estrutura

```text
.
|-- README.md
|-- requirements.txt
|-- docs/
|   |-- AGENTS.md
|   |-- SPEC.md
|   `-- TASKS.md
|-- data/
|   |-- raw/
|   `-- processed/
|-- src/
|   |-- 1_tratamento.py
|   |-- 2_analise_exploratoria.py
|   |-- 3_predicao.py
|   |-- 4_classificacao.py
|   |-- 5_clusterizacao.py
|   |-- 6_analise_nlp.py
|   `-- functions/
`-- outputs/
    |-- figures/
    `-- tables/
```

## Observações metodológicas

- `rating_count` é a principal variável-alvo porque representa popularidade, engajamento ou demanda aparente.
- `rating` representa satisfação média e pode ser usado como variável explicativa ou dimensão complementar.
- A base não contém vendas reais, receita, quantidade vendida nem evolução temporal.
- Resultados devem ser interpretados como análise de catálogo/produtos, não como análise transacional de vendas.
