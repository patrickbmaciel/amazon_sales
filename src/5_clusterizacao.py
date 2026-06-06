# Clusterização baseline para segmentar produtos.

import os
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from functions.io_utils import ensure_directories, read_csv, require_file, save_table
from functions.plots import save_current_figure


ROOT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_PATH = ROOT_DIR / "data" / "processed" / "amazon_sales_processed.csv"
CLUSTERED_PATH = ROOT_DIR / "data" / "processed" / "amazon_sales_clusters.csv"
FIGURES_DIR = ROOT_DIR / "outputs" / "figures"
TABLES_DIR = ROOT_DIR / "outputs" / "tables"


# Sugere nomes descritivos em português com base em perfis médios simples.
# Classifica o preço do cluster usando a mediana geral da base como referência.
def price_suffix(row: pd.Series, overall_actual_price_median: float) -> str:
    if row.get("actual_price_median", 0) > overall_actual_price_median:
        return "caro"
    return "acessivel"


# Sugere nomes descritivos com base na tabela-resumo dos clusters.
def suggest_cluster_profiles(summary: pd.DataFrame, overall_actual_price_median: float) -> pd.DataFrame:
    summary = summary.copy()
    labels = []
    high_rating_count = summary["rating_count_median"].quantile(0.75)
    low_rating_count = summary["rating_count_median"].quantile(0.25)
    high_log_rating_count = summary["log_rating_count_median"].quantile(0.75)
    low_log_rating_count = summary["log_rating_count_median"].quantile(0.25)
    high_price = summary["actual_price_median"].quantile(0.75)
    low_rating = summary["rating_median"].quantile(0.25)

    for _, row in summary.iterrows():
        is_popular = (
            row.get("rating_count_median", 0) >= high_rating_count
            or row.get("log_rating_count_median", 0) >= high_log_rating_count
        )
        is_niche = (
            row.get("rating_count_median", 0) <= low_rating_count
            or row.get("log_rating_count_median", 0) <= low_log_rating_count
        )
        price_group = price_suffix(row, overall_actual_price_median)

        if is_popular:
            labels.append(f"popular_{price_group}")
        elif row.get("rating_median", 0) >= 4 and is_niche:
            labels.append(f"nichado_{price_group}")
        elif row.get("actual_price_median", 0) >= high_price:
            labels.append("premium")
        elif row.get("rating_median", 5) < low_rating:
            labels.append("risco")
        else:
            labels.append("misto")
    summary["suggested_profile"] = labels
    return summary


# Cria resumo de clusters para um método específico.
def summarize_clusters(
    df: pd.DataFrame,
    features: list[str],
    label_col: str,
    overall_actual_price_median: float,
) -> pd.DataFrame:
    summary = df.groupby(label_col)[features].agg(["mean", "median", "count"])
    summary.columns = ["_".join(col).strip("_") for col in summary.columns]
    summary = summary.reset_index()
    return suggest_cluster_profiles(summary, overall_actual_price_median)


# Adiciona rótulo de legenda no formato Cluster N - perfil.
def add_cluster_legend_label(df: pd.DataFrame, summary: pd.DataFrame, label_col: str) -> pd.DataFrame:
    plot_df = df.copy()
    profile_map = summary.set_index(label_col)["suggested_profile"].to_dict()
    plot_df["cluster_legend"] = plot_df[label_col].map(
        lambda value: f"Cluster {value} - {profile_map.get(value, 'misto')}"
    )
    return plot_df


# Salva gráfico PCA 2D colorido pelo método de clusterização informado.
def plot_pca_clusters(df: pd.DataFrame, summary: pd.DataFrame, label_col: str, title: str, path: Path) -> None:
    plot_df = add_cluster_legend_label(df, summary, label_col)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=plot_df,
        x="pca_1",
        y="pca_2",
        hue="cluster_legend",
        palette="tab10",
        alpha=0.75,
    )
    plt.title(title)
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.legend(title="Perfil sugerido", bbox_to_anchor=(1.05, 1), loc="upper left")
    save_current_figure(path)


# Executa métodos de clusterização e salva resultados.
def main() -> None:
    print("Preparando diretorios...")
    ensure_directories([FIGURES_DIR, TABLES_DIR, CLUSTERED_PATH.parent])

    print("Lendo base tratada...")
    require_file(PROCESSED_PATH, "Execute primeiro: python src/1_tratamento.py")
    df = read_csv(PROCESSED_PATH)
    if len(df) < 2:
        raise ValueError("A clusterizacao precisa de pelo menos 2 produtos.")

    features = [
        "discounted_price",
        "actual_price",
        "discount_percentage",
        "discount_value",
        "rating",
        "rating_count",
        "log_rating_count",
    ]
    features = [col for col in features if col in df.columns]
    if not features:
        raise ValueError("Nenhuma variavel numerica disponivel para clusterizacao.")

    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    X_scaled = pipeline.fit_transform(df[features])

    print("Aplicando clusterizacao...")
    n_clusters = min(5, len(df))
    min_samples = min(10, len(df))
    df["cluster_kmeans"] = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(X_scaled)
    df["cluster_hierarchical"] = AgglomerativeClustering(n_clusters=n_clusters).fit_predict(X_scaled)
    df["cluster_dbscan"] = DBSCAN(eps=1.5, min_samples=min_samples).fit_predict(X_scaled)

    pca = PCA(n_components=2, random_state=42)
    pca_values = pca.fit_transform(X_scaled)
    df["pca_1"] = pca_values[:, 0]
    df["pca_2"] = pca_values[:, 1]

    print("Salvando resultados...")
    save_table(df, CLUSTERED_PATH)

    overall_actual_price_median = df["actual_price"].median()
    kmeans_summary = summarize_clusters(df, features, "cluster_kmeans", overall_actual_price_median)
    hierarchical_summary = summarize_clusters(df, features, "cluster_hierarchical", overall_actual_price_median)
    dbscan_summary = summarize_clusters(df, features, "cluster_dbscan", overall_actual_price_median)
    save_table(kmeans_summary, TABLES_DIR / "cluster_summary.csv")
    save_table(kmeans_summary, TABLES_DIR / "cluster_summary_kmeans.csv")
    save_table(hierarchical_summary, TABLES_DIR / "cluster_summary_hierarchical.csv")
    save_table(dbscan_summary, TABLES_DIR / "cluster_summary_dbscan.csv")

    plot_pca_clusters(df, kmeans_summary, "cluster_kmeans", "PCA 2D por cluster K-Means", FIGURES_DIR / "cluster_pca_2d.png")
    plot_pca_clusters(df, kmeans_summary, "cluster_kmeans", "PCA 2D por cluster K-Means", FIGURES_DIR / "cluster_pca_kmeans.png")
    plot_pca_clusters(
        df,
        hierarchical_summary,
        "cluster_hierarchical",
        "PCA 2D por cluster hierárquico",
        FIGURES_DIR / "cluster_pca_hierarchical.png",
    )
    plot_pca_clusters(df, dbscan_summary, "cluster_dbscan", "PCA 2D por cluster DBSCAN", FIGURES_DIR / "cluster_pca_dbscan.png")

    print("Clusterizacao concluida.")


if __name__ == "__main__":
    main()
