# Utilitarios de entrada, saida e criacao de diretorios.

from pathlib import Path

import pandas as pd


# Cria diretorios quando ainda nao existem.
def ensure_directories(paths: list[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


# Localiza um arquivo CSV bruto em data/raw.
def find_raw_csv(raw_dir: Path) -> Path:
    csv_files = sorted(raw_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"Nenhum CSV encontrado em {raw_dir}. "
            "Baixe a base com: kaggle datasets download -d "
            "karkavelrajaj/amazon-sales-dataset -p data/raw --unzip"
        )
    if len(csv_files) > 1:
        print(f"Mais de um CSV encontrado. Usando: {csv_files[0].name}")
    return csv_files[0]


# Interrompe a execucao com mensagem clara quando um arquivo esperado nao existe.
def require_file(path: Path, hint: str | None = None) -> None:
    if path.exists():
        return

    message = f"Arquivo nao encontrado: {path}"
    if hint:
        message = f"{message}\n{hint}"
    raise FileNotFoundError(message)


# Le um CSV com configuracao simples.
def read_csv(path: Path) -> pd.DataFrame:
    require_file(path)
    return pd.read_csv(path)


# Salva uma tabela CSV criando o diretorio de destino.
def save_table(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)
