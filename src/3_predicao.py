# Modelos baseline de regressão para prever log_rating_count.

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor
except ImportError as error:
    raise ImportError("Instale as dependências com: python -m pip install -r requirements.txt") from error

from functions.io_utils import ensure_directories, read_csv, require_file, save_table
from functions.metrics import metrics_to_frame, regression_metrics
from functions.plots import save_current_figure


warnings.filterwarnings("ignore", message="X does not have valid feature names")

ROOT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_PATH = ROOT_DIR / "data" / "processed" / "amazon_sales_processed.csv"
FIGURES_DIR = ROOT_DIR / "outputs" / "figures"
TABLES_DIR = ROOT_DIR / "outputs" / "tables"
TARGET = "log_rating_count"


# Cria preprocessador para variáveis numéricas e categóricas.
def build_preprocessor(df: pd.DataFrame, features: list[str]) -> ColumnTransformer:
    numeric_features = df[features].select_dtypes(include="number").columns.tolist()
    categorical_features = [col for col in features if col not in numeric_features]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )


# Treina modelos de regressão e salva métricas.
def main() -> None:
    print("Preparando diretorios...")
    ensure_directories([FIGURES_DIR, TABLES_DIR])

    print("Lendo base tratada...")
    require_file(PROCESSED_PATH, "Execute primeiro: python src/1_tratamento.py")
    df = read_csv(PROCESSED_PATH).dropna(subset=[TARGET])
    if len(df) < 10:
        raise ValueError("A base precisa de pelo menos 10 linhas validas para treino/teste.")

    features = [
        "discounted_price",
        "actual_price",
        "discount_percentage",
        "discount_value",
        "rating",
        "main_category",
        "sub_category_1",
        "price_range",
        "discount_range",
        "product_name_length",
        "about_product_length",
        "review_title_length",
        "review_content_length",
    ]
    features = [col for col in features if col in df.columns]
    if not features:
        raise ValueError("Nenhuma variavel explicativa disponivel para predicao.")

    X = df[features]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=1),
        "XGBoost Regressor": XGBRegressor(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.08,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=1,
        ),
        "LightGBM Regressor": LGBMRegressor(
            n_estimators=150,
            learning_rate=0.08,
            random_state=42,
            n_jobs=1,
            verbose=-1,
        ),
    }

    rows = []
    predictions = {}
    print("Treinando modelos...")
    for name, model in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocess", build_preprocessor(df, features)),
                ("model", model),
            ]
        )
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        rows.append(regression_metrics(y_test, y_pred, name))
        predictions[name] = y_pred

    metrics_df = metrics_to_frame(rows).sort_values("rmse")
    save_table(metrics_df, TABLES_DIR / "prediction_metrics.csv")

    best_model = metrics_df.iloc[0]["model"]
    plot_df = pd.DataFrame({"observed": y_test, "predicted": predictions[best_model]})
    plt.figure(figsize=(7, 6))
    sns.scatterplot(data=plot_df, x="observed", y="predicted", alpha=0.7)
    plt.title(f"Observado vs. previsto - {best_model}")
    plt.xlabel("log_rating_count observado")
    plt.ylabel("log_rating_count previsto")
    save_current_figure(FIGURES_DIR / "prediction_observed_vs_predicted.png")

    print("Predicao concluida.")


if __name__ == "__main__":
    main()
