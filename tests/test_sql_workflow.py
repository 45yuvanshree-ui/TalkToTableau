from pathlib import Path

import pytest

from sql import SQLExecutor, SQLWorkflow, validate_sql


GOLDEN_CSV = Path("database/golden_sales.csv")


def test_select_sql_is_allowed():
    result = validate_sql(
        "SELECT region, SUM(revenue) FROM data GROUP BY region"
    )

    assert result.is_valid is True
    assert "SELECT" in result.normalized_query.upper()


@pytest.mark.parametrize(
    "query",
    [
        "DROP TABLE data",
        "DELETE FROM data",
        "UPDATE data SET revenue = 0",
        "INSERT INTO data VALUES (1)",
        "CREATE TABLE test (id INTEGER)",
        "ALTER TABLE data ADD COLUMN test INTEGER",
    ],
)
def test_write_operations_are_rejected(query):
    result = validate_sql(query)

    assert result.is_valid is False


def test_multiple_statements_are_rejected():
    result = validate_sql(
        "SELECT * FROM data; DROP TABLE data"
    )

    assert result.is_valid is False
    assert "Multiple SQL statements" in result.reason


def test_empty_sql_is_rejected():
    result = validate_sql("")

    assert result.is_valid is False


def test_file_sql_execution():
    executor = SQLExecutor()

    result = executor.execute_file(
        GOLDEN_CSV,
        """
        SELECT
            region,
            SUM(revenue) AS total_revenue
        FROM data
        GROUP BY region
        ORDER BY region
        """,
    )

    assert result.row_count == 4
    assert result.columns == ["region", "total_revenue"]


def test_sql_workflow_with_generator():
    def generator(user_request: str) -> str:
        assert user_request == "Show revenue by region"

        return """
            SELECT
                region,
                SUM(revenue) AS total_revenue
            FROM data
            GROUP BY region
            ORDER BY region
        """

    workflow = SQLWorkflow(generator)

    result = workflow.run_file(
        GOLDEN_CSV,
        "Show revenue by region",
    )

    assert result.row_count == 4
    assert result.columns == ["region", "total_revenue"]


def test_sql_workflow_has_bounded_retries():
    attempts = []

    def failing_generator(user_request: str) -> str:
        attempts.append(user_request)
        return "DROP TABLE data"

    workflow = SQLWorkflow(
        sql_generator=failing_generator,
        max_retries=2,
    )

    with pytest.raises(RuntimeError):
        workflow.run_file(
            GOLDEN_CSV,
            "Do something",
        )

    assert len(attempts) == 3