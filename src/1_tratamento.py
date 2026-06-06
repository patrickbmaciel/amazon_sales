# Tratamento inicial do Amazon Sales Dataset.
# Le o CSV bruto baixado via Kaggle, limpa tipos, cria variaveis derivadas
# e salva a base processada para as proximas etapas do pipeline.

from pathlib import Path

from functions.io_utils import ensure_directories, find_raw_csv, read_csv, save_table
from functions.preprocessing import clean_amazon_sales


ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
OUTPUT_PATH = PROCESSED_DIR / "amazon_sales_processed.csv"


# Executa tratamento da base.
def main() -> None:
    print("Criando diretorios...")
    ensure_directories([RAW_DIR, PROCESSED_DIR])

    print("Localizando CSV bruto...")
    raw_path = find_raw_csv(RAW_DIR)

    print(f"Lendo arquivo: {raw_path.name}")
    raw_df = read_csv(raw_path)

    print("Limpando e criando variaveis...")
    processed_df = clean_amazon_sales(raw_df)

    print(f"Salvando base tratada em: {OUTPUT_PATH}")
    save_table(processed_df, OUTPUT_PATH)

    print("Tratamento concluido.")


if __name__ == "__main__":
    main()
