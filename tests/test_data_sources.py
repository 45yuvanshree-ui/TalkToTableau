from pathlib import Path

import pandas as pd

from data_sources.file_source import FileDataSource


GOLDEN_CSV = Path("database/golden_sales.csv")


def test_csv_loading():
    source = FileDataSource(GOLDEN_CSV)

    assert source.test_connection() is True

    dataframe = source.load()

    assert isinstance(dataframe, pd.DataFrame)
    assert len(dataframe) == 20
    assert "revenue" in dataframe.columns
    assert "region" in dataframe.columns


def test_csv_schema():
    source = FileDataSource(GOLDEN_CSV)

    schema = source.get_schema()

    assert "revenue" in schema
    assert "product" in schema
    assert "sale_date" in schema
    
GOLDEN_EXCEL = Path("database/golden_sales.xlsx")


def test_excel_loading():
    source = FileDataSource(GOLDEN_EXCEL)

    assert source.test_connection() is True

    dataframe = source.load()

    assert isinstance(dataframe, pd.DataFrame)
    assert len(dataframe) == 20
    assert "revenue" in dataframe.columns
    assert "region" in dataframe.columns