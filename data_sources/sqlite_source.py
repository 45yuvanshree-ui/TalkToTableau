from pathlib import Path
from typing import Any

import pandas as pd
import sqlite3

from .base import DataSource


class SQLiteDataSource(DataSource):
    """Read-only SQLite data source for database-backed workflows."""

    def __init__(self, database_path: str | Path):
        self.database_path = Path(database_path)

    def test_connection(self) -> bool:
        """Return True when the SQLite database is accessible."""
        return (
            self.database_path.exists()
            and self.database_path.is_file()
        )

    def get_schema(self) -> dict[str, Any]:
        """Return SQLite table and column metadata."""
        if not self.test_connection():
            raise ValueError(
                f"SQLite database not found: {self.database_path}"
            )

        connection = sqlite3.connect(
            f"file:{self.database_path.resolve()}?mode=ro",
            uri=True,
        )

        try:
            tables = connection.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type = 'table'
                  AND name NOT LIKE 'sqlite_%'
                ORDER BY name
                """
            ).fetchall()

            schema: dict[str, Any] = {}

            for (table_name,) in tables:
                columns = connection.execute(
                    f'PRAGMA table_info("{table_name}")'
                ).fetchall()

                schema[table_name] = {
                    column[1]: column[2]
                    for column in columns
                }

            return schema

        finally:
            connection.close()

    def execute(self, query: str) -> pd.DataFrame:
        """Execute a read-only SQL query against the SQLite database."""
        if not self.test_connection():
            raise ValueError(
                f"SQLite database not found: {self.database_path}"
            )

        normalized_query = query.strip()

        if not normalized_query:
            raise ValueError("SQL query is empty.")

        if not normalized_query.lower().startswith("select"):
            raise ValueError(
                "SQLiteDataSource only permits SELECT queries."
            )

        connection = sqlite3.connect(
            f"file:{self.database_path.resolve()}?mode=ro",
            uri=True,
        )

        try:
            return pd.read_sql_query(
                normalized_query,
                connection,
            )
        finally:
            connection.close()