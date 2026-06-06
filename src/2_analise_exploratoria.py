# Análise exploratória inicial do Amazon Sales Dataset tratado.

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from functions.io_utils import ensure_directories, read_csv, require_file, save_table
from functions.plots import DISPLAY_LABELS, plot_histogram, plot_scatter, save_current_figure


ROOT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_PATH = ROOT_DIR / "data" / "processed" / "amazon_sales_processed.csv"
FIGURES_DIR = ROOT_DIR / "outputs" / "figures"
TABLES_DIR = ROOT_DIR / "outputs" / "tables"


# Gera tabelas e gráficos exploratórios.
def main() -> None:
    print("Preparando diretorios...")
    ensure_directories([FIGURES_DIR, TABLES_DIR])

    print("Lendo base tratada...")
    require_file(PROCESSED_PATH, "Execute primeiro: python src/1_tratamento.py")
    df = read_csv(PROCESSED_PATH)
    if df.empty:
        raise ValueError("A base tratada esta vazia. Revise a etapa de tratamento.")

    print("Gerando tabelas...")
    summary = df.describe(include="all").transpose().reset_index().rename(columns={"index": "column"})
    save_table(summary, TABLES_DIR / "eda_summary.csv")

    missing = (
        df.isna()
        .sum()
        .reset_index()
        .rename(columns={"index": "column", 0: "missing_count"})
    )
    missing["missing_pct"] = missing["missing_count"] / len(df)
    save_table(missing, TABLES_DIR / "eda_missing_values.csv")

    top_categories_count = df["main_category"].value_counts().head(15).reset_index()
    top_categories_count.columns = ["main_category", "product_count"]
    save_table(top_categories_count, TABLES_DIR / "eda_top_categories_by_count.csv")

    category_rating = (
        df.groupby("main_category", dropna=False)["rating"]
        .mean()
        .sort_values(ascending=False)
        .head(15)
        .reset_index()
    )
    save_table(category_rating, TABLES_DIR / "eda_top_categories_by_rating.csv")

    category_popularity = (
        df.groupby("main_category", dropna=False)["rating_count"]
        .agg(["mean", "median", "count"])
        .sort_values("median", ascending=False)
        .head(15)
        .reset_index()
    )
    save_table(category_popularity, TABLES_DIR / "eda_top_categories_by_rating_count.csv")

    product_cols = [col for col in ["product_name", "main_category", "rating", "rating_count"] if col in df.columns]
    save_table(
        df.sort_values("rating_count", ascending=False)[product_cols].head(30),
        TABLES_DIR / "eda_top_products_by_rating_count.csv",
    )

    discount_cols = [col for col in ["product_name", "main_category", "discount_percentage", "rating_count"] if col in df.columns]
    save_table(
        df.sort_values("discount_percentage", ascending=False)[discount_cols].head(30),
        TABLES_DIR / "eda_top_products_by_discount.csv",
    )

    print("Gerando graficos...")
    for column in ["discounted_price", "actual_price", "discount_percentage", "rating", "rating_count"]:
        if column in df.columns:
            plot_histogram(df, column, FIGURES_DIR / f"eda_distribution_{column}.png")

    top_main_categories = df["main_category"].value_counts().head(10).index
    box_df = df[df["main_category"].isin(top_main_categories)]
    plt.figure(figsize=(11, 6))
    sns.boxplot(data=box_df, x="main_category", y="discounted_price")
    plt.xticks(rotation=45, ha="right")
    plt.title("Preço com desconto por categoria principal")
    plt.xlabel("Categoria principal")
    plt.ylabel("Preço com desconto")
    save_current_figure(FIGURES_DIR / "eda_boxplot_price_by_main_category.png")

    plot_scatter(df, "discount_percentage", "rating_count", FIGURES_DIR / "eda_scatter_discount_rating_count.png")
    plot_scatter(df, "rating", "rating_count", FIGURES_DIR / "eda_scatter_rating_rating_count.png")

    numeric_cols = df.select_dtypes(include="number").columns
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr()
        corr = corr.rename(index=DISPLAY_LABELS, columns=DISPLAY_LABELS)
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, cmap="coolwarm_r", center=0, vmin=-1, vmax=1)
        plt.title("Correlação entre variáveis numéricas")
        save_current_figure(FIGURES_DIR / "eda_correlation_heatmap.png")

    print("Analise exploratoria concluida.")


if __name__ == "__main__":
    main()
