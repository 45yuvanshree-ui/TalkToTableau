from pathlib import Path

from schema import SchemaRetriever
from sql import SQLExecutor


BIRD_DATABASE = (
    Path("minidev")
    / "MINIDEV"
    / "dev_databases"
    / "debit_card_specializing"
    / "debit_card_specializing.sqlite"
)


def test_sqlite_schema_retrieval():
    """Verify that a BIRD SQLite database exposes all tables."""

    retriever = SchemaRetriever()

    schema = retriever.from_sqlite(
        BIRD_DATABASE
    )

    assert schema.source_type == "sqlite"

    table_names = {
        table.name
        for table in schema.tables
    }

    assert "customers" in table_names
    assert "yearmonth" in table_names


def test_sqlite_execution():
    """Verify read-only SQL execution against a BIRD database."""

    executor = SQLExecutor()

    result = executor.execute_sqlite(
        database_path=BIRD_DATABASE,
        query=(
            "SELECT Currency, COUNT(*) AS customer_count "
            "FROM customers "
            "GROUP BY Currency "
            "ORDER BY Currency"
        ),
    )

    assert result.row_count > 0
    assert "Currency" in result.columns
    assert "customer_count" in result.columns


def test_sqlite_execution_rejects_write_query():
    """Verify that SQLite execution uses the SQL safety validator."""

    executor = SQLExecutor()

    try:
        executor.execute_sqlite(
            database_path=BIRD_DATABASE,
            query="DELETE FROM customers",
        )
    except ValueError as error:
        assert "Only SELECT statements are allowed." in str(error)
    else:
        raise AssertionError(
            "Unsafe SQL was not rejected."
        )