
from pathlib import Path

import pandas as pd

from data_sources.file_source import FileDataSource
from data_sources.postgres_source import PostgresDataSource


GOLDEN_CSV = Path("database/golden_sales.csv")
GOLDEN_EXCEL = Path("database/golden_sales.xlsx")


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


def test_excel_loading():
    source = FileDataSource(GOLDEN_EXCEL)

    assert source.test_connection() is True

    dataframe = source.load()

    assert isinstance(dataframe, pd.DataFrame)
    assert len(dataframe) == 20
    assert "revenue" in dataframe.columns
    assert "region" in dataframe.columns


def test_postgres_connection_string(monkeypatch):
    connection_string = (
        "postgresql://user:password@localhost:5432/testdb"
    )

    source = PostgresDataSource(
        connection_string=connection_string
    )

    assert source.connection_string == connection_string


def test_postgres_test_connection_success(monkeypatch):
    class FakeConnection:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def execute(self, query):
            assert query == "SELECT 1"

    def fake_connect(*args, **kwargs):
        assert args[0] == (
            "postgresql://user:password@localhost:5432/testdb"
        )
        assert kwargs["connect_timeout"] == 5
        return FakeConnection()

    monkeypatch.setattr(
        "data_sources.postgres_source.psycopg.connect",
        fake_connect,
    )

    source = PostgresDataSource(
        "postgresql://user:password@localhost:5432/testdb"
    )

    assert source.test_connection() is True


def test_postgres_test_connection_failure(monkeypatch):
    class FakePsycopgError(Exception):
        pass

    def fake_connect(*args, **kwargs):
        raise FakePsycopgError("connection failed")

    monkeypatch.setattr(
        "data_sources.postgres_source.psycopg.Error",
        FakePsycopgError,
    )

    monkeypatch.setattr(
        "data_sources.postgres_source.psycopg.connect",
        fake_connect,
    )

    source = PostgresDataSource(
        "postgresql://user:password@localhost:5432/testdb"
    )

    assert source.test_connection() is False


def test_postgres_get_schema(monkeypatch):
    class FakeConnection:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def execute(self, query):
            assert isinstance(query, str)

            class Result:
                def __init__(self, rows):
                    self.rows = rows

                def fetchall(self):
                    return self.rows

            # Query 1: column metadata.
            if "information_schema.columns" in query:
                return Result(
                    [
                        (
                            "public",
                            "sales",
                            "region",
                            "character varying",
                            "YES",
                        ),
                        (
                            "public",
                            "sales",
                            "revenue",
                            "numeric",
                            "NO",
                        ),
                    ]
                )

            # Query 2: primary-key metadata.
            if "PRIMARY KEY" in query:
                return Result([])

            # Query 3: foreign-key metadata.
            if "FOREIGN KEY" in query:
                return Result([])

            raise AssertionError(
                f"Unexpected PostgreSQL schema query: {query}"
            )

    monkeypatch.setattr(
        "data_sources.postgres_source.psycopg.connect",
        lambda *args, **kwargs: FakeConnection(),
    )

    source = PostgresDataSource(
        "postgresql://user:password@localhost:5432/testdb"
    )

    schema = source.get_schema()

    assert len(schema) == 2

    assert schema[0]["table_schema"] == "public"
    assert schema[0]["table_name"] == "sales"
    assert schema[0]["column_name"] == "region"
    assert schema[0]["data_type"] == "character varying"
    assert schema[0]["nullable"] is True
    assert schema[0]["primary_key"] is False
    assert schema[0]["foreign_keys"] == []

    assert schema[1]["column_name"] == "revenue"
    assert schema[1]["data_type"] == "numeric"
    assert schema[1]["nullable"] is False


def test_postgres_execute_returns_dataframe(monkeypatch):
    class FakeCursor:
        description = [
            type("Column", (), {"name": "region"})(),
            type("Column", (), {"name": "revenue"})(),
        ]

        def execute(self, query):
            assert query == (
                "SELECT region, revenue "
                "FROM sales"
            )

        def fetchall(self):
            return [
                ("North", 60000),
                ("South", 55600),
            ]

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    class FakeConnection:
        def cursor(self):
            return FakeCursor()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    monkeypatch.setattr(
        "data_sources.postgres_source.psycopg.connect",
        lambda *args, **kwargs: FakeConnection(),
    )

    source = PostgresDataSource(
        "postgresql://user:password@localhost:5432/testdb"
    )

    dataframe = source.execute(
        "SELECT region, revenue FROM sales"
    )

    assert isinstance(dataframe, pd.DataFrame)
    assert list(dataframe.columns) == [
        "region",
        "revenue",
    ]

    assert len(dataframe) == 2
    assert dataframe.iloc[0]["region"] == "North"
    assert dataframe.iloc[0]["revenue"] == 60000
