# Modelos baseline para classificar produtos populares.

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
except ImportError as error:
    raise ImportError("Instale as dependências com: python -m pip install -r requirements.txt") from error

from functions.io_utils import ensure_directories, read_csv, require_file, save_table
from functions.metrics import classification_metrics, metrics_to_frame
from functions.plots import save_current_figure


warnings.filterwarnings("ignore", message="X does not have valid feature names")

ROOT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_PATH = ROOT_DIR / "data" / "processed" / "amazon_sales_processed.csv"
FIGURES_DIR = ROOT_DIR / "outputs" / "figures"
TABLES_DIR = ROOT_DIR / "outputs" / "tables"
TARGET = "popular_product"


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


# Salva importância de variáveis quando o modelo disponibiliza.
def save_feature_importance(pipeline: Pipeline, path: Path) -> None:
    model = pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return

    preprocessor = pipeline.named_steps["preprocess"]
    feature_names = preprocessor.get_feature_names_out()
    importances = pd.DataFrame(
        {"feature": feature_names, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
    save_table(importances.head(50), path)


# Treina classificadores e salva métricas.
def main() -> None:
    print("Preparando diretorios...")
    ensure_directories([FIGURES_DIR, TABLES_DIR])

    print("Lendo base tratada...")
    require_file(PROCESSED_PATH, "Execute primeiro: python src/1_tratamento.py")
    df = read_csv(PROCESSED_PATH).dropna(subset=[TARGET])
    df[TARGET] = df[TARGET].astype(str).str.lower().isin(["true", "1", "yes"])
    if len(df) < 10 or df[TARGET].nunique() < 2:
        raise ValueError("A classificacao precisa de pelo menos 10 linhas e duas classes no alvo.")

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
        raise ValueError("Nenhuma variavel explicativa disponivel para classificacao.")

    X = df[features]
    y = df[TARGET]
    stratify = y if y.nunique() == 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest Classifier": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=1),
        "XGBoost Classifier": XGBClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.08,
            eval_metric="logloss",
            random_state=42,
            n_jobs=1,
        ),
        "LightGBM Classifier": LGBMClassifier(
            n_estimators=150,
            learning_rate=0.08,
            random_state=42,
            n_jobs=1,
            verbose=-1,
        ),
    }

    rows = []
    fitted = {}
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
        rows.append(classification_metrics(y_test, y_pred, name))
        fitted[name] = pipeline
        predictions[name] = y_pred

    metrics_df = metrics_to_frame(rows).sort_values("f1", ascending=False)
    save_table(metrics_df, TABLES_DIR / "classification_metrics.csv")

    best_model = metrics_df.iloc[0]["model"]
    cm = confusion_matrix(y_test, predictions[best_model])
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Matriz de confusão - {best_model}")
    plt.xlabel("Previsto")
    plt.ylabel("Observado")
    save_current_figure(FIGURES_DIR / "classification_confusion_matrix.png")

    save_feature_importance(
        fitted[best_model],
        TABLES_DIR / "classification_feature_importance.csv",
    )

    print("Classificacao concluida.")


if __name__ == "__main__":
    main()
