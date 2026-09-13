from pathlib import Path

from data_sources import (
    DataSource,
    FileDataSource,
    PostgresDataSource,
    SQLiteDataSource,
)

from .models import (
    ColumnSchema,
    DataSchema,
    TableSchema,
)


class SchemaRetriever:
    """Retrieve and normalize schemas from supported data sources."""

    def from_source(
        self,
        source: DataSource,
        table_name: str | None = None,
    ) -> DataSchema:
        """
        Build a normalized schema from a DataSource.

        This is the preferred abstraction-level API.
        """

        if isinstance(
            source,
            FileDataSource,
        ):
            raw_schema = source.get_schema()

            resolved_table_name = (
                table_name
                or source.file_path.stem
            )

            table = TableSchema(
                name=resolved_table_name,
                source_type="file",
                columns=[
                    ColumnSchema(
                        name=column_name,
                        data_type=data_type,
                    )
                    for column_name, data_type
                    in raw_schema.items()
                ],
            )

            return DataSchema(
                source_type="file",
                tables=[table],
            )

        if isinstance(
            source,
            SQLiteDataSource,
        ):
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
                        for column_name, data_type
                        in columns.items()
                    ],
                )
                for table_name, columns
                in raw_schema.items()
            ]

            return DataSchema(
                source_type="sqlite",
                tables=tables,
            )

        if isinstance(
            source,
            PostgresDataSource,
        ):
            raw_schema = source.get_schema()

            tables: dict[
                tuple[str, str],
                list[ColumnSchema],
            ] = {}

            for column in raw_schema:
                table_key = (
                    column["table_schema"],
                    column["table_name"],
                )

                tables.setdefault(
                    table_key,
                    [],
                ).append(
                    ColumnSchema(
                        name=column["column_name"],
                        data_type=column["data_type"],
                    )
                )

            table_models = [
                TableSchema(
                    name=(
                        f"{table_schema}."
                        f"{table_name}"
                    ),
                    source_type="postgresql",
                    columns=columns,
                )
                for (
                    table_schema,
                    table_name,
                ), columns in tables.items()
            ]

            return DataSchema(
                source_type="postgresql",
                tables=table_models,
            )

        raise TypeError(
            "Unsupported DataSource implementation: "
            f"{type(source).__name__}"
        )

    def from_file(
        self,
        file_path: str | Path,
        table_name: str | None = None,
    ) -> DataSchema:
        """Build a normalized schema from a CSV or Excel file."""

        source = FileDataSource(
            file_path
        )

        return self.from_source(
            source=source,
            table_name=table_name,
        )

    def from_postgres(
        self,
        connection_string: str,
    ) -> DataSchema:
        """Build a normalized schema from PostgreSQL."""

        source = PostgresDataSource(
            connection_string
        )

        return self.from_source(
            source=source
        )

    def from_sqlite(
        self,
        database_path: str | Path,
    ) -> DataSchema:
        """Build a normalized schema from SQLite."""

        source = SQLiteDataSource(
            database_path
        )

        return self.from_source(
            source=source
        )