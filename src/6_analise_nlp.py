# Análise textual simples de reviews e descrições.

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud

from functions.io_utils import ensure_directories, read_csv, require_file, save_table
from functions.plots import save_current_figure
from functions.text_utils import build_tfidf, clean_text, term_frequency, tokenize


ROOT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_PATH = ROOT_DIR / "data" / "processed" / "amazon_sales_processed.csv"
FIGURES_DIR = ROOT_DIR / "outputs" / "figures"
TABLES_DIR = ROOT_DIR / "outputs" / "tables"


# Converte campos booleanos que podem ter sido lidos como texto.
def as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


# Salva gráfico de termos frequentes.
def save_frequency_plot(freq_df: pd.DataFrame, path: Path, title: str) -> None:
    if freq_df.empty:
        return
    plt.figure(figsize=(9, 6))
    sns.barplot(data=freq_df.head(20), x="count", y="term")
    plt.title(title)
    plt.xlabel("Frequência")
    plt.ylabel("Termo")
    save_current_figure(path)


# Salva uma nuvem de palavras simples para leitura rápida de temas recorrentes.
def save_wordcloud(texts: pd.Series, path: Path, title: str) -> None:
    text = " ".join(texts.fillna("").astype(str))
    if not text.strip():
        return
    cloud = WordCloud(width=1200, height=700, background_color="white", collocations=False).generate(text)
    plt.figure(figsize=(10, 6))
    plt.imshow(cloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(title)
    save_current_figure(path)


# Calcula frequência absoluta e relativa de termos em um grupo.
def term_frequency_relative(texts: pd.Series, top_n: int = 100) -> pd.DataFrame:
    rows = []
    token_count = 0
    counts: dict[str, int] = {}
    for text in texts.fillna(""):
        tokens = tokenize(text)
        token_count += len(tokens)
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1
    for term, count in sorted(counts.items(), key=lambda item: item[1], reverse=True)[:top_n]:
        rows.append({"term": term, "count": count, "relative_freq": count / token_count if token_count else 0})
    return pd.DataFrame(rows)


# Compara termos mais distintivos entre dois grupos.
def distinctive_terms(
    df: pd.DataFrame,
    group_mask: pd.Series,
    text_col: str,
    left_name: str,
    right_name: str,
    top_n: int = 50,
) -> pd.DataFrame:
    left = term_frequency_relative(df.loc[group_mask, text_col], top_n=200)
    right = term_frequency_relative(df.loc[~group_mask, text_col], top_n=200)
    merged = left.merge(right, on="term", how="outer", suffixes=(f"_{left_name}", f"_{right_name}")).fillna(0)
    merged["relative_freq_diff"] = merged[f"relative_freq_{left_name}"] - merged[f"relative_freq_{right_name}"]
    return merged.sort_values("relative_freq_diff", key=lambda values: values.abs(), ascending=False).head(top_n)


# Resume os principais termos TF-IDF médios por grupo.
def tfidf_by_group(texts: pd.Series, groups: pd.Series, top_n: int = 30) -> pd.DataFrame:
    tfidf_df, _ = build_tfidf(texts, max_features=150)
    tfidf_df["group"] = groups.reset_index(drop=True)
    rows = []
    for group_name, group_df in tfidf_df.groupby("group"):
        means = group_df.drop(columns="group").mean().sort_values(ascending=False).head(top_n)
        for term, value in means.items():
            rows.append({"group": group_name, "term": term, "mean_tfidf": value})
    return pd.DataFrame(rows)


# Salva boxplot de contagem de palavras por grupo.
def save_word_count_boxplot(df: pd.DataFrame, group_col: str, path: Path, title: str) -> None:
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x=group_col, y="word_count")
    plt.title(title)
    plt.xlabel("Grupo")
    plt.ylabel("Quantidade de palavras")
    save_current_figure(path)


# Executa análise NLP simples com comparações por satisfação e popularidade.
def main() -> None:
    print("Preparando diretorios...")
    ensure_directories([FIGURES_DIR, TABLES_DIR])

    print("Lendo base tratada...")
    require_file(PROCESSED_PATH, "Execute primeiro: python src/1_tratamento.py")
    df = read_csv(PROCESSED_PATH)

    text_cols = [col for col in ["review_title", "review_content", "about_product"] if col in df.columns]
    if text_cols:
        df["combined_text"] = df[text_cols].fillna("").agg(" ".join, axis=1)
    else:
        df["combined_text"] = ""
    df["combined_text_clean"] = df["combined_text"].map(clean_text)
    df["word_count"] = df["combined_text_clean"].str.split().str.len()
    df["rating_group"] = as_bool(df["high_rating"]).map({True: "rating_alto", False: "rating_baixo"})
    df["popularity_group"] = as_bool(df["popular_product"]).map({True: "popular", False: "nao_popular"})

    print("Gerando frequencias de termos...")
    overall_terms = term_frequency(df["combined_text_clean"], top_n=50)
    save_table(overall_terms, TABLES_DIR / "nlp_terms_overall.csv")
    save_frequency_plot(overall_terms, FIGURES_DIR / "nlp_terms_overall.png", "Termos mais frequentes")
    save_wordcloud(df["combined_text_clean"], FIGURES_DIR / "nlp_wordcloud_overall.png", "Nuvem de palavras geral")

    high_rating_mask = df["rating_group"].eq("rating_alto")
    popular_mask = df["popularity_group"].eq("popular")

    high_rating_terms = term_frequency(df.loc[high_rating_mask, "combined_text_clean"], top_n=50)
    low_rating_terms = term_frequency(df.loc[~high_rating_mask, "combined_text_clean"], top_n=50)
    save_table(high_rating_terms, TABLES_DIR / "nlp_terms_high_rating.csv")
    save_table(low_rating_terms, TABLES_DIR / "nlp_terms_low_rating.csv")
    save_frequency_plot(high_rating_terms, FIGURES_DIR / "nlp_terms_high_rating.png", "Termos frequentes em produtos com rating alto")
    save_frequency_plot(low_rating_terms, FIGURES_DIR / "nlp_terms_low_rating.png", "Termos frequentes em produtos com rating baixo")

    popular_terms = term_frequency(df.loc[popular_mask, "combined_text_clean"], top_n=50)
    not_popular_terms = term_frequency(df.loc[~popular_mask, "combined_text_clean"], top_n=50)
    save_table(popular_terms, TABLES_DIR / "nlp_terms_popular.csv")
    save_table(not_popular_terms, TABLES_DIR / "nlp_terms_not_popular.csv")
    save_frequency_plot(popular_terms, FIGURES_DIR / "nlp_terms_popular.png", "Termos frequentes em produtos populares")
    save_frequency_plot(not_popular_terms, FIGURES_DIR / "nlp_terms_not_popular.png", "Termos frequentes em produtos não populares")

    rating_distinctive = distinctive_terms(
        df,
        high_rating_mask,
        "combined_text_clean",
        "rating_alto",
        "rating_baixo",
    )
    popularity_distinctive = distinctive_terms(
        df,
        popular_mask,
        "combined_text_clean",
        "popular",
        "nao_popular",
    )
    save_table(rating_distinctive, TABLES_DIR / "nlp_distinctive_terms_high_vs_low_rating.csv")
    save_table(popularity_distinctive, TABLES_DIR / "nlp_distinctive_terms_popular_vs_not_popular.csv")

    print("Gerando TF-IDF...")
    try:
        tfidf_df, _ = build_tfidf(df["combined_text_clean"], max_features=100)
        tfidf_mean = (
            tfidf_df.mean()
            .sort_values(ascending=False)
            .head(50)
            .reset_index()
            .rename(columns={"index": "term", 0: "mean_tfidf"})
        )
        save_table(tfidf_mean, TABLES_DIR / "nlp_tfidf_top_terms.csv")
        save_table(
            tfidf_by_group(df["combined_text_clean"], df["popularity_group"], top_n=30),
            TABLES_DIR / "nlp_tfidf_by_popularity.csv",
        )
        save_table(
            tfidf_by_group(df["combined_text_clean"], df["rating_group"], top_n=30),
            TABLES_DIR / "nlp_tfidf_by_rating_group.csv",
        )
    except ValueError as error:
        print(f"TF-IDF nao gerado: {error}")

    word_count_cols = [col for col in ["product_id", "product_name", "word_count"] if col in df.columns]
    word_counts = df[word_count_cols].copy()
    save_table(word_counts, TABLES_DIR / "nlp_word_counts.csv")
    save_word_count_boxplot(
        df,
        "popularity_group",
        FIGURES_DIR / "nlp_word_count_by_popularity.png",
        "Quantidade de palavras por popularidade",
    )
    save_word_count_boxplot(
        df,
        "rating_group",
        FIGURES_DIR / "nlp_word_count_by_rating_group.png",
        "Quantidade de palavras por grupo de rating",
    )

    print("Analise NLP concluida.")


if __name__ == "__main__":
    main()
