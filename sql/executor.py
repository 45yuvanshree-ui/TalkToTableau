from pathlib import Path

import pandas as pd

from data_sources import (
    DataSource,
    FileDataSource,
    PostgresDataSource,
    SQLiteDataSource,
)

from .models import SQLResult
from .validator import validate_sql


class SQLExecutor:
    """Execute validated read-only SQL against supported data sources."""

    def execute_source(
        self,
        source: DataSource,
        query: str,
    ) -> SQLResult:
        """
        Execute SQL through the common DataSource interface.
        """

        validation = validate_sql(
            query
        )

        if not validation.is_valid:
            raise ValueError(
                validation.reason
            )

        dataframe = source.execute(
            validation.normalized_query
        )

        return self._to_result(
            dataframe
        )

    def execute_file(
        self,
        file_path: str | Path,
        query: str,
    ) -> SQLResult:
        """Execute SQL against a CSV/Excel source."""

        source = FileDataSource(
            file_path
        )

        return self.execute_source(
            source=source,
            query=query,
        )

    def execute_postgres(
        self,
        connection_string: str,
        query: str,
    ) -> SQLResult:
        """Execute SQL against PostgreSQL."""

        source = PostgresDataSource(
            connection_string
        )

        return self.execute_source(
            source=source,
            query=query,
        )

    def execute_sqlite(
        self,
        database_path: str | Path,
        query: str,
    ) -> SQLResult:
        """Execute SQL against SQLite."""

        source = SQLiteDataSource(
            database_path
        )

        return self.execute_source(
            source=source,
            query=query,
        )

    @staticmethod
    def _to_result(
        dataframe: pd.DataFrame,
    ) -> SQLResult:
        return SQLResult(
            columns=[
                str(column)
                for column
                in dataframe.columns
            ],
            rows=(
                dataframe
                .astype(object)
                .where(
                    dataframe.notna(),
                    None,
                )
                .values
                .tolist()
            ),
            row_count=len(dataframe),
        )