# Metricas reutilizaveis para regressao e classificacao.

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)


# Calcula metricas basicas de regressao.
def regression_metrics(y_true, y_pred, model_name: str) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    non_zero = y_true != 0
    mape = np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) if non_zero.any() else np.nan

    return {
        "model": model_name,
        "r2": r2_score(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mape": mape,
    }


# Calcula metricas basicas de classificacao binaria.
def classification_metrics(y_true, y_pred, model_name: str) -> dict:
    return {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


# Converte uma lista de metricas em DataFrame ordenado por modelo.
def metrics_to_frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows).sort_values("model").reset_index(drop=True)
