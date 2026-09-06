from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "golden_sales.csv"
EXCEL_PATH = BASE_DIR / "golden_sales.xlsx"


def create_excel_copy() -> None:
    """Create an Excel version of the golden dataset."""
    dataframe = pd.read_csv(CSV_PATH)
    dataframe.to_excel(EXCEL_PATH, index=False)


if __name__ == "__main__":
    create_excel_copy()
    print(f"Created: {EXCEL_PATH}")