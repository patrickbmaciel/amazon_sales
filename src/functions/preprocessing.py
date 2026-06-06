# Funcoes de limpeza e engenharia de atributos.

from __future__ import annotations

import re

import numpy as np
import pandas as pd


# Padroniza nomes de colunas para snake_case simples.
def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )
    return df


# Converte precos com simbolos e separadores para float.
def clean_price(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.replace(r"[^\d.]", "", regex=True)
        .replace({"": np.nan, "nan": np.nan})
    )
    return pd.to_numeric(cleaned, errors="coerce")


# Converte percentual de desconto para escala 0-100.
def clean_discount_percentage(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.replace("%", "", regex=False)
        .str.replace(r"[^\d.]", "", regex=True)
        .replace({"": np.nan, "nan": np.nan})
    )
    return pd.to_numeric(cleaned, errors="coerce")


# Converte rating para float.
def clean_rating(series: pd.Series) -> pd.Series:
    extracted = series.astype(str).str.extract(r"(\d+(?:\.\d+)?)", expand=False)
    return pd.to_numeric(extracted, errors="coerce")


# Converte contagem de avaliacoes para numero.
def clean_rating_count(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace(r"[^\d]", "", regex=True)
        .replace({"": np.nan, "nan": np.nan})
    )
    return pd.to_numeric(cleaned, errors="coerce")


# Separa categorias hierarquicas em ate tres niveis.
def split_categories(df: pd.DataFrame, category_col: str = "category") -> pd.DataFrame:
    df = df.copy()
    if category_col not in df.columns:
        df["main_category"] = np.nan
        df["sub_category_1"] = np.nan
        df["sub_category_2"] = np.nan
        return df

    parts = df[category_col].fillna("").astype(str).str.split("|", expand=True)
    df["main_category"] = parts[0].replace("", np.nan) if 0 in parts else np.nan
    df["sub_category_1"] = parts[1].replace("", np.nan) if 1 in parts else np.nan
    df["sub_category_2"] = parts[2].replace("", np.nan) if 2 in parts else np.nan
    return df


# Conta caracteres em textos tratando valores ausentes.
def text_length(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.len()


# Cria variaveis derivadas usadas nas analises e modelos.
def create_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in ["product_name", "about_product", "review_title", "review_content"]:
        if col not in df.columns:
            df[col] = ""

    df["discount_value"] = df["actual_price"] - df["discounted_price"]
    df["rating_count"] = df["rating_count"].fillna(0)
    df["log_rating_count"] = np.log1p(df["rating_count"])

    df["product_name_length"] = text_length(df["product_name"])
    df["about_product_length"] = text_length(df["about_product"])
    df["review_title_length"] = text_length(df["review_title"])
    df["review_content_length"] = text_length(df["review_content"])

    df["high_rating"] = df["rating"] >= 4.0
    popularity_cutoff = df["rating_count"].quantile(0.75)
    df["popular_product"] = df["rating_count"] >= popularity_cutoff if popularity_cutoff > 0 else False

    price_bins = [-np.inf, 500, 1000, 5000, 10000, np.inf]
    price_labels = ["very_low", "low", "medium", "high", "premium"]
    df["price_range"] = pd.cut(df["discounted_price"], bins=price_bins, labels=price_labels)

    discount_bins = [-np.inf, 10, 25, 50, 75, np.inf]
    discount_labels = ["very_low", "low", "medium", "high", "very_high"]
    df["discount_range"] = pd.cut(
        df["discount_percentage"], bins=discount_bins, labels=discount_labels
    )

    return df


# Executa limpeza principal da base Amazon Sales.
def clean_amazon_sales(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_column_names(df)

    if "discounted_price" in df.columns:
        df["discounted_price"] = clean_price(df["discounted_price"])
    if "actual_price" in df.columns:
        df["actual_price"] = clean_price(df["actual_price"])
    if "discount_percentage" in df.columns:
        df["discount_percentage"] = clean_discount_percentage(df["discount_percentage"])
    if "rating" in df.columns:
        df["rating"] = clean_rating(df["rating"])
    if "rating_count" in df.columns:
        df["rating_count"] = clean_rating_count(df["rating_count"])

    required_numeric = ["discounted_price", "actual_price", "discount_percentage", "rating", "rating_count"]
    for col in required_numeric:
        if col not in df.columns:
            df[col] = np.nan

    df = split_categories(df)
    df = create_derived_features(df)

    # Remove espacos duplicados em campos textuais sem alterar conteudo essencial.
    for col in ["product_name", "about_product", "review_title", "review_content"]:
        df[col] = df[col].fillna("").astype(str).apply(lambda value: re.sub(r"\s+", " ", value).strip())

    return df
