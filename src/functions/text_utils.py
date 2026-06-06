# Utilitarios simples para analise textual.

from __future__ import annotations

import re
from collections import Counter

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


DEFAULT_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "with",
}


# Limpa texto mantendo apenas letras, numeros e espacos.
def clean_text(value: str) -> str:
    value = "" if pd.isna(value) else str(value).lower()
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


# Tokeniza texto de forma simples.
def tokenize(text: str, stopwords: set[str] | None = None) -> list[str]:
    stopwords = DEFAULT_STOPWORDS if stopwords is None else stopwords
    return [token for token in clean_text(text).split() if token not in stopwords and len(token) > 2]


# Calcula termos mais frequentes em uma serie de textos.
def term_frequency(texts: pd.Series, top_n: int = 30) -> pd.DataFrame:
    counter: Counter[str] = Counter()
    for text in texts.fillna(""):
        counter.update(tokenize(text))
    return pd.DataFrame(counter.most_common(top_n), columns=["term", "count"])


# Gera matriz TF-IDF simples e retorna DataFrame com nomes de termos.
def build_tfidf(texts: pd.Series, max_features: int = 100) -> tuple[pd.DataFrame, TfidfVectorizer]:
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words="english",
        min_df=2,
        ngram_range=(1, 2),
    )
    matrix = vectorizer.fit_transform(texts.fillna("").astype(str).map(clean_text))
    tfidf_df = pd.DataFrame(matrix.toarray(), columns=vectorizer.get_feature_names_out())
    return tfidf_df, vectorizer
