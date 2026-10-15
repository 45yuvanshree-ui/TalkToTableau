
from __future__ import annotations

from pathlib import Path
from urllib.parse import quote

from data_sources.base import DataSource
from data_sources.file_source import FileDataSource
from data_sources.postgres_source import PostgresDataSource
from data_sources.sqlite_source import SQLiteDataSource

from .models import (
    ColumnSchema,
    DataSchema,
    ForeignKeySchema,
    TableSchema,
)


class SchemaRetriever:
    """Retrieve and normalize schemas from supported data sources."""

    def from_source(
        self,
        source: DataSource,
        source_type: str | None = None,
    ) -> DataSchema:
        """
        Build a normalized DataSchema from a DataSource.

        Supports:
        - Original simple dictionary schemas
        - Rich dictionary schemas
        - PostgreSQL column-metadata list schemas
        """
        raw_schema = source.get_schema()

        resolved_source_type = (
            source_type or self._infer_source_type(source)
        )

        # Normalize PostgreSQL's column-based schema into a
        # table-based schema while preserving key metadata.
        if (
            isinstance(raw_schema, list)
            and resolved_source_type == "postgresql"
        ):
            normalized_schema: dict[str, dict] = {}

            def qualified_table_name(
                schema_name: str,
                name: str,
            ) -> str:
                # Preserve existing names for public tables.
                if schema_name == "public":
                    return name

                return f"{schema_name}.{name}"

            for item in raw_schema:
                if not isinstance(item, dict):
                    continue

                raw_table_name = item.get("table_name")
                raw_column_name = item.get("column_name")

                if not raw_table_name or not raw_column_name:
                    continue

                schema_name = str(
                    item.get("table_schema") or "public"
                )

                table_name = qualified_table_name(
                    schema_name,
                    str(raw_table_name),
                )

                column_name = str(raw_column_name)

                table_metadata = normalized_schema.setdefault(
                    table_name,
                    {
                        "columns": [],
                        "foreign_keys": [],
                    },
                )

                # Keep column metadata instead of dropping PK/FK
                # information returned by PostgresDataSource.
                table_metadata["columns"].append(
                    {
                        "name": column_name,
                        "data_type": str(
                            item.get("data_type") or "TEXT"
                        ),
                        "primary_key": bool(
                            item.get("primary_key", False)
                        ),
                        "nullable": item.get("nullable"),
                        "description": item.get("description"),
                    }
                )

                raw_foreign_keys = item.get("foreign_keys", [])

                if not isinstance(raw_foreign_keys, list):
                    continue

                for foreign_key in raw_foreign_keys:
                    if not isinstance(foreign_key, dict):
                        continue

                    referenced_table = foreign_key.get(
                        "referenced_table"
                    )

                    if not referenced_table:
                        continue

                    referenced_schema = str(
                        foreign_key.get(
                            "referenced_table_schema"
                        )
                        or "public"
                    )

                    table_metadata["foreign_keys"].append(
                        {
                            "column_name": column_name,
                            "referenced_table": qualified_table_name(
                                referenced_schema,
                                str(referenced_table),
                            ),
                            "referenced_column": foreign_key.get(
                                "referenced_column"
                            ),
                        }
                    )

            raw_schema = normalized_schema

        # Defensive validation.
        if not isinstance(raw_schema, dict):
            raise TypeError(
                "Unsupported schema format returned by data source: "
                f"{type(raw_schema).__name__}"
            )

        tables: list[TableSchema] = []

        for table_name, table_metadata in raw_schema.items():

            # Handle the original simple schema format:
            # {"sales": {"region": "TEXT", "revenue": "FLOAT"}}
            if self._is_simple_column_mapping(table_metadata):
                columns = [
                    ColumnSchema(
                        name=str(column_name),
                        data_type=str(data_type),
                    )
                    for column_name, data_type
                    in table_metadata.items()
                ]

                tables.append(
                    TableSchema(
                        name=str(table_name),
                        source_type=resolved_source_type,
                        columns=columns,
                    )
                )

                continue

            # Rich schema format.
            if not isinstance(table_metadata, dict):
                continue

            columns: list[ColumnSchema] = []

            raw_columns = table_metadata.get("columns", [])

            if isinstance(raw_columns, list):
                for column in raw_columns:

                    # Support schemas that represent columns as strings.
                    if not isinstance(column, dict):
                        if column is not None:
                            columns.append(
                                ColumnSchema(
                                    name=str(column),
                                    data_type="TEXT",
                                )
                            )
                        continue

                    column_name = column.get("name")

                    if not column_name:
                        continue

                    data_type = column.get("data_type", "TEXT")
                    nullable = column.get("nullable")
                    description = column.get("description")

                    columns.append(
                        ColumnSchema(
                            name=str(column_name),
                            data_type=str(data_type or "TEXT"),
                            primary_key=bool(
                                column.get("primary_key", False)
                            ),
                            nullable=(
                                bool(nullable)
                                if nullable is not None
                                else None
                            ),
                            description=(
                                str(description).strip()
                                if description
                                else None
                            ),
                        )
                    )

            # Normalize foreign-key relationships.
            foreign_keys: list[ForeignKeySchema] = []

            raw_foreign_keys = table_metadata.get(
                "foreign_keys",
                [],
            )

            if isinstance(raw_foreign_keys, list):
                for foreign_key in raw_foreign_keys:
                    if not isinstance(foreign_key, dict):
                        continue

                    column_name = foreign_key.get("column_name")
                    referenced_table = foreign_key.get(
                        "referenced_table"
                    )

                    if not column_name or not referenced_table:
                        continue

                    referenced_column = foreign_key.get(
                        "referenced_column"
                    )

                    foreign_keys.append(
                        ForeignKeySchema(
                            column_name=str(column_name),
                            referenced_table=str(referenced_table),
                            referenced_column=(
                                str(referenced_column)
                                if referenced_column
                                else None
                            ),
                        )
                    )

            table_description = table_metadata.get("description")

            tables.append(
                TableSchema(
                    name=str(table_name),
                    source_type=resolved_source_type,
                    columns=columns,
                    foreign_keys=foreign_keys,
                    description=(
                        str(table_description).strip()
                        if table_description
                        else None
                    ),
                )
            )

        return DataSchema(
            source_type=resolved_source_type,
            tables=tables,
        )

    def from_file(
        self,
        file_path: str | Path,
        table_name: str = "data",
    ) -> DataSchema:
        """Build a normalized schema from a CSV/XLSX file."""
        source = FileDataSource(file_path)

        raw_schema = source.get_schema()

        columns = [
            ColumnSchema(
                name=str(column_name),
                data_type=str(data_type),
            )
            for column_name, data_type in raw_schema.items()
        ]

        return DataSchema(
            source_type="file",
            tables=[
                TableSchema(
                    name=table_name,
                    source_type="file",
                    columns=columns,
                )
            ],
        )

    def from_sqlite(
        self,
        database_path: str | Path,
    ) -> DataSchema:
        """Build a normalized schema from SQLite."""
        source = SQLiteDataSource(database_path)

        return self.from_source(
            source,
            source_type="sqlite",
        )

    def from_postgres(
        self,
        connection_string: str | None = None,
        *,
        host: str | None = None,
        port: int | None = None,
        database: str | None = None,
        user: str | None = None,
        password: str | None = None,
    ) -> DataSchema:
        """Build a normalized schema from PostgreSQL."""

        if connection_string:
            source = PostgresDataSource(
                connection_string=connection_string
            )

        else:
            # Preserve the existing API for individual connection fields.
            if not host:
                raise ValueError("PostgreSQL host is required.")

            resolved_port = port or 5432

            if not database:
                raise ValueError("PostgreSQL database is required.")

            if not user:
                raise ValueError("PostgreSQL user is required.")

            if password is None:
                raise ValueError("PostgreSQL password is required.")

            connection_string = (
                "postgresql://"
                f"{quote(user, safe='')}:"
                f"{quote(password, safe='')}@"
                f"{quote(host, safe='')}:"
                f"{resolved_port}/"
                f"{quote(database, safe='')}"
            )

            source = PostgresDataSource(
                connection_string=connection_string
            )

        return self.from_source(
            source,
            source_type="postgresql",
        )

    @staticmethod
    def _infer_source_type(
        source: DataSource,
    ) -> str:
        """Infer the source type from the data source instance."""

        if isinstance(source, SQLiteDataSource):
            return "sqlite"

        if isinstance(source, PostgresDataSource):
            return "postgresql"

        if isinstance(source, FileDataSource):
            return "file"

        return source.__class__.__name__.lower()

    @staticmethod
    def _is_simple_column_mapping(
        table_metadata,
    ) -> bool:
        """
        Detect the original simple schema format.

        Example:
            {
                "region": "object",
                "revenue": "float64",
            }
        """
        if not isinstance(table_metadata, dict):
            return False

        return (
            "columns" not in table_metadata
            and "foreign_keys" not in table_metadata
            and "description" not in table_metadata
        )
