import os

import pandas as pd
import psycopg

from .base import DataSource


class PostgresDataSource(DataSource):
    """PostgreSQL data source using Psycopg 3."""

    def __init__(self, connection_string: str | None = None):
        self.connection_string = connection_string or os.getenv(
            "POSTGRES_CONNECTION_STRING"
        )

        if not self.connection_string:
            raise ValueError(
                "POSTGRES_CONNECTION_STRING is not configured."
            )

    def test_connection(self) -> bool:
        """Test whether PostgreSQL is reachable."""
        try:
            with psycopg.connect(
                self.connection_string,
                connect_timeout=5,
            ) as connection:
                connection.execute("SELECT 1")
            return True
        except psycopg.Error:
            return False

    def get_schema(self) -> list[dict[str, str]]:
        """Return basic table and column metadata."""
        query = """
            SELECT
                table_schema,
                table_name,
                column_name,
                data_type
            FROM information_schema.columns
            WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
            ORDER BY table_schema, table_name, ordinal_position;
        """

        with psycopg.connect(self.connection_string) as connection:
            rows = connection.execute(query).fetchall()

        return [
            {
                "table_schema": row[0],
                "table_name": row[1],
                "column_name": row[2],
                "data_type": row[3],
            }
            for row in rows
        ]

    def execute(self, query: str) -> pd.DataFrame:
        """Execute a query and return the result as a DataFrame."""
        with psycopg.connect(self.connection_string) as connection:
            with connection.cursor() as cursor:
                cursor.execute(query)

                if cursor.description is None:
                    raise ValueError(
                        "The query did not return tabular results."
                    )

                columns = [
                    column.name for column in cursor.description
                ]
                rows = cursor.fetchall()

        return pd.DataFrame(rows, columns=columns)