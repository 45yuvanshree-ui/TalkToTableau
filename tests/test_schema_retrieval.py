from pathlib import Path

from schema import SchemaRetriever


GOLDEN_CSV = Path("database/golden_sales.csv")
GOLDEN_EXCEL = Path("database/golden_sales.xlsx")


EXPECTED_COLUMNS = {
    "sale_id",
    "sale_date",
    "product",
    "category",
    "region",
    "customer_type",
    "quantity",
    "revenue",
    "cost",
}


def test_csv_schema_retrieval():
    retriever = SchemaRetriever()

    schema = retriever.from_file(
        GOLDEN_CSV,
        table_name="golden_sales",
    )

    assert schema.source_type == "file"
    assert len(schema.tables) == 1

    table = schema.tables[0]

    assert table.name == "golden_sales"
    assert table.source_type == "file"

    column_names = {column.name for column in table.columns}

    assert column_names == EXPECTED_COLUMNS


def test_excel_schema_retrieval():
    retriever = SchemaRetriever()

    schema = retriever.from_file(
        GOLDEN_EXCEL,
        table_name="golden_sales",
    )

    assert schema.source_type == "file"
    assert len(schema.tables) == 1

    table = schema.tables[0]

    column_names = {column.name for column in table.columns}

    assert column_names == EXPECTED_COLUMNS


def test_schema_contains_data_types():
    retriever = SchemaRetriever()

    schema = retriever.from_file(
        GOLDEN_CSV,
        table_name="golden_sales",
    )

    table = schema.tables[0]

    column_types = {
        column.name: column.data_type
        for column in table.columns
    }

    assert column_types["sale_id"]
    assert column_types["product"]
    assert column_types["revenue"]
    assert column_types["sale_date"]