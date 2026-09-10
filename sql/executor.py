from pathlib import Path

import duckdb
import pandas as pd

from data_sources.file_source import FileDataSource
from data_sources.postgres_source import PostgresDataSource
from data_sources.sqlite_source import SQLiteDataSource

from .models import SQLResult
from .validator import validate_sql


class SQLExecutor:
    """Execute validated read-only SQL against supported data sources."""

    def execute_file(
        self,
        file_path: str | Path,
        query: str,
    ) -> SQLResult:
        validation = validate_sql(query)

        if not validation.is_valid:
            raise ValueError(validation.reason)

        dataframe = FileDataSource(file_path).load()

        connection = duckdb.connect()

        try:
            connection.register(
                "data",
                dataframe,
            )

            result = connection.execute(
                validation.normalized_query
            ).fetchdf()

        finally:
            connection.close()

        return self._to_result(result)

    def execute_postgres(
        self,
        connection_string: str,
        query: str,
    ) -> SQLResult:
        validation = validate_sql(query)

        if not validation.is_valid:
            raise ValueError(validation.reason)

        dataframe = PostgresDataSource(
            connection_string
        ).execute(
            validation.normalized_query
        )

        return self._to_result(dataframe)

    def execute_sqlite(
        self,
        database_path: str | Path,
        query: str,
    ) -> SQLResult:
        """Execute validated read-only SQL against SQLite."""

        validation = validate_sql(query)

        if not validation.is_valid:
            raise ValueError(validation.reason)

        dataframe = SQLiteDataSource(
            database_path
        ).execute(
            validation.normalized_query
        )

        return self._to_result(dataframe)

    @staticmethod
    def _to_result(
        dataframe: pd.DataFrame,
    ) -> SQLResult:
        return SQLResult(
            columns=[
                str(column)
                for column in dataframe.columns
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