# Funções simples para gráficos recorrentes.

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DISPLAY_LABELS = {
    "actual_price": "Preço original",
    "discounted_price": "Preço com desconto",
    "discount_percentage": "Percentual de desconto",
    "discount_value": "Valor do desconto",
    "rating": "Rating",
    "rating_count": "Quantidade de avaliações",
    "log_rating_count": "Log da quantidade de avaliações",
    "main_category": "Categoria principal",
    "product_name_length": "Tamanho do nome do produto",
    "about_product_length": "Tamanho da descrição",
    "review_title_length": "Tamanho do título da review",
    "review_content_length": "Tamanho do conteúdo da review",
}


# Retorna um rótulo amigável para gráficos sem alterar nomes técnicos nos dados.
def friendly_label(column: str) -> str:
    return DISPLAY_LABELS.get(column, column)


# Salva a figura atual e fecha o objeto.
def save_current_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


# Gera histograma simples para uma coluna numérica.
def plot_histogram(df: pd.DataFrame, column: str, path: Path, bins: int = 30) -> None:
    plt.figure(figsize=(8, 5))
    sns.histplot(df[column].dropna(), bins=bins, kde=True)
    plt.title(f"Distribuição de {friendly_label(column)}")
    plt.xlabel(friendly_label(column))
    plt.ylabel("Frequência")
    save_current_figure(path)


# Gera scatterplot simples.
def plot_scatter(df: pd.DataFrame, x: str, y: str, path: Path) -> None:
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x=x, y=y, alpha=0.6)
    plt.title(f"{friendly_label(x)} vs. {friendly_label(y)}")
    plt.xlabel(friendly_label(x))
    plt.ylabel(friendly_label(y))
    save_current_figure(path)


# Gera gráfico de barras horizontal.
def plot_bar(df: pd.DataFrame, x: str, y: str, path: Path, title: str) -> None:
    plt.figure(figsize=(9, 6))
    sns.barplot(data=df, x=x, y=y)
    plt.title(title)
    plt.xlabel(friendly_label(x))
    plt.ylabel(friendly_label(y))
    save_current_figure(path)
