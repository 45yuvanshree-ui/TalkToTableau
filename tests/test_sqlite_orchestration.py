from pathlib import Path

from orchestration.workflow import TalkToTableauWorkflow


BIRD_DATABASE = (
    Path("minidev")
    / "MINIDEV"
    / "dev_databases"
    / "debit_card_specializing"
    / "debit_card_specializing.sqlite"
)


def test_workflow_detects_sqlite_source():
    """Verify that the workflow recognizes a SQLite database."""

    workflow = TalkToTableauWorkflow(
        data_file=BIRD_DATABASE
    )

    assert workflow._is_sqlite_source() is True


def test_workflow_retrieves_multi_table_sqlite_schema():
    """Verify that orchestration retrieves the complete SQLite schema."""

    workflow = TalkToTableauWorkflow(
        data_file=BIRD_DATABASE
    )

    schema = workflow._retrieve_schema()

    table_names = {
        table.name
        for table in schema.tables
    }

    assert schema.source_type == "sqlite"
    assert "customers" in table_names
    assert "yearmonth" in table_names


def test_workflow_executes_sqlite_query():
    """Verify that orchestration executes SQL through SQLite."""

    workflow = TalkToTableauWorkflow(
        data_file=BIRD_DATABASE
    )

    result = workflow._execute_sql(
        """
        SELECT
            Currency,
            COUNT(*) AS customer_count
        FROM customers
        GROUP BY Currency
        ORDER BY Currency
        """
    )

    assert result.row_count > 0
    assert result.columns == [
        "Currency",
        "customer_count",
    ]