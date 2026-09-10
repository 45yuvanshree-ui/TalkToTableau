from pathlib import Path

from data_sources.file_source import FileDataSource
from data_sources.postgres_source import PostgresDataSource
from data_sources.sqlite_source import SQLiteDataSource

from .models import ColumnSchema, DataSchema, TableSchema


class SchemaRetriever:
    """Retrieve and normalize schemas from supported data sources."""

    def from_file(
        self,
        file_path: str | Path,
        table_name: str | None = None,
    ) -> DataSchema:
        """Build a normalized schema from a CSV or Excel file."""

        source = FileDataSource(file_path)
        raw_schema = source.get_schema()

        resolved_table_name = table_name or Path(file_path).stem

        table = TableSchema(
            name=resolved_table_name,
            source_type="file",
            columns=[
                ColumnSchema(
                    name=column_name,
                    data_type=data_type,
                )
                for column_name, data_type in raw_schema.items()
            ],
        )

        return DataSchema(
            source_type="file",
            tables=[table],
        )

    def from_postgres(
        self,
        connection_string: str,
    ) -> DataSchema:
        """Build a normalized schema from PostgreSQL."""

        source = PostgresDataSource(connection_string)

        raw_schema = source.get_schema()

        tables: dict[tuple[str, str], list[ColumnSchema]] = {}

        for column in raw_schema:
            table_key = (
                column["table_schema"],
                column["table_name"],
            )

            tables.setdefault(table_key, []).append(
                ColumnSchema(
                    name=column["column_name"],
                    data_type=column["data_type"],
                )
            )

        table_models = [
            TableSchema(
                name=f"{table_schema}.{table_name}",
                source_type="postgresql",
                columns=columns,
            )
            for (table_schema, table_name), columns in tables.items()
        ]

        return DataSchema(
            source_type="postgresql",
            tables=table_models,
        )

    def from_sqlite(
        self,
        database_path: str | Path,
    ) -> DataSchema:
        """Build a normalized schema from a SQLite database."""

        source = SQLiteDataSource(database_path)
        raw_schema = source.get_schema()

        tables = [
            TableSchema(
                name=table_name,
                source_type="sqlite",
                columns=[
                    ColumnSchema(
                        name=column_name,
                        data_type=data_type,
                    )
                    for column_name, data_type in columns.items()
                ],
            )
            for table_name, columns in raw_schema.items()
        ]

        return DataSchema(
            source_type="sqlite",
            tables=tables,
        )