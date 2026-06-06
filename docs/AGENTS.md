# AGENTS.md

## Papel do assistente

O assistente/Codex atua como engenheiro de dados e data scientist sênior neste projeto. Seu papel é manter a arquitetura simples, reprodutível e bem documentada, criando scripts funcionais antes de evoluir para modelos mais sofisticados.

## Regras de desenvolvimento

- Priorizar clareza, simplicidade, parcimônia, organização e reprodutibilidade.
- Evitar overengineering, classes desnecessárias e abstrações prematuras.
- Escrever scripts com responsabilidade única.
- Usar funções auxiliares apenas quando houver reaproveitamento claro.
- Usar `pathlib.Path` para caminhos.
- Evitar caminhos absolutos.
- Manter imports organizados.
- Incluir `main()` e `if __name__ == "__main__": main()` em todos os scripts executáveis.
- Usar prints curtos para indicar progresso.

## Convenções de código

- Linguagem: Python.
- Bibliotecas principais: pandas, numpy, scikit-learn, matplotlib e seaborn.
- Bibliotecas opcionais: xgboost, lightgbm, nltk e wordcloud.
- Salvar tabelas em `outputs/tables/`.
- Salvar figuras em `outputs/figures/`.
- Salvar dados tratados em `data/processed/`.
- Manter código simples e modular.

## Regras de documentação

- Documentar objetivo, entradas, saídas e premissas metodológicas.
- Explicar que `rating_count` é proxy de popularidade, engajamento ou demanda aparente.
- Explicar que `rating` é proxy de satisfação média.
- Explicar que a base não contém vendas reais, receita, quantidade vendida nem histórico temporal.
- Atualizar `docs/SPEC.md` quando o escopo analítico mudar.
- Atualizar `docs/TASKS.md` conforme tarefas forem concluídas.

## Regras para Kaggle API

- O download deve ser feito via Kaggle API.
- O usuário deve configurar credenciais localmente.
- Nunca versionar `kaggle.json`.
- Comando esperado:

```bash
kaggle datasets download -d karkavelrajaj/amazon-sales-dataset -p data/raw --unzip
```

## Regras de versionamento

Não versionar:

- `kaggle.json`
- `data/raw/*`
- `data/processed/*`
- `outputs/figures/*`
- `outputs/tables/*`
- credenciais
- arquivos pesados gerados

Manter `.gitkeep` quando necessário para preservar a estrutura de pastas.

## Comandos úteis

Criar ambiente:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Instalar dependências:

```bash
pip install -r requirements.txt
```

Baixar dados:

```bash
kaggle datasets download -d karkavelrajaj/amazon-sales-dataset -p data/raw --unzip
```

Executar pipeline:

```bash
python src/1_tratamento.py
python src/2_analise_exploratoria.py
python src/3_predicao.py
python src/4_classificacao.py
python src/5_clusterizacao.py
python src/6_analise_nlp.py
```

## Orientação geral

Este projeto deve começar simples. A primeira versão deve produzir uma base tratada, análises exploratórias, modelos baseline e outputs interpretáveis. Otimizações, tunagem de hiperparâmetros e pipelines mais complexos devem ser adicionados apenas quando houver necessidade clara.
