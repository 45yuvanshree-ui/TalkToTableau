from __future__ import annotations

import csv
import sqlite3
from pathlib import Path
from typing import Any

from .base import DataSource


class SQLiteDataSource(DataSource):
    """SQLite data source with schema metadata and optional BIRD descriptions."""

    def __init__(self, database_path: str | Path):
        self.database_path = Path(database_path)

        if not self.database_path.exists():
            raise FileNotFoundError(
                f"SQLite database not found: {self.database_path}"
            )

    def test_connection(self) -> bool:
        """Verify that the SQLite database can be opened read-only."""

        try:
            with sqlite3.connect(
                f"file:{self.database_path.resolve()}?mode=ro",
                uri=True,
            ) as connection:
                connection.execute("SELECT 1")

            return True

        except (sqlite3.Error, OSError):
            return False

    def get_schema(self) -> dict[str, Any]:
        """
        Retrieve SQLite schema metadata.

        Includes:
        - column names
        - data types
        - nullability
        - primary keys
        - foreign keys
        - optional BIRD column descriptions
        """

        schema: dict[str, Any] = {}

        with sqlite3.connect(
            f"file:{self.database_path.resolve()}?mode=ro",
            uri=True,
        ) as connection:

            cursor = connection.cursor()

            cursor.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type = 'table'
                  AND name NOT LIKE 'sqlite_%'
                ORDER BY name
                """
            )

            tables = [
                row[0]
                for row in cursor.fetchall()
            ]

            for table_name in tables:

                escaped_table_name = (
                    table_name.replace('"', '""')
                )

                # ----------------------------------------
                # Columns
                # ----------------------------------------

                cursor.execute(
                    f'PRAGMA table_info("{escaped_table_name}")'
                )

                table_info = cursor.fetchall()

                columns: list[dict[str, Any]] = []

                for column in table_info:
                    columns.append(
                        {
                            "name": column[1],
                            "data_type": column[2] or "TEXT",
                            "nullable": not bool(column[3]),
                            "primary_key": bool(column[5]),
                        }
                    )

                # ----------------------------------------
                # Foreign keys
                # ----------------------------------------

                cursor.execute(
                    f'PRAGMA foreign_key_list("{escaped_table_name}")'
                )

                foreign_key_rows = cursor.fetchall()

                foreign_keys: list[dict[str, Any]] = []

                for foreign_key in foreign_key_rows:
                    foreign_keys.append(
                        {
                            "column_name": foreign_key[3],
                            "referenced_table": foreign_key[2],
                            "referenced_column": foreign_key[4],
                        }
                    )

                schema[table_name] = {
                    "columns": columns,
                    "foreign_keys": foreign_keys,
                }

        # Add optional semantic metadata.
        self._attach_external_descriptions(schema)

        return schema

    def _attach_external_descriptions(
        self,
        schema: dict[str, Any],
    ) -> None:
        """
        Attach BIRD database_description metadata.

        BIRD uses files such as:

            database_description/frpm.csv
            database_description/schools.csv
            database_description/satscores.csv

        with columns:

            original_column_name
            column_name
            column_description
            data_format
            value_description
        """

        description_directory = (
            self.database_path.parent
            / "database_description"
        )

        if not description_directory.is_dir():
            return

        # Build a case-insensitive table lookup.
        schema_tables = {
            str(table_name).strip().lower(): table_name
            for table_name in schema
        }

        for description_file in description_directory.glob("*.csv"):

            table_key = (
                description_file.stem
                .strip()
                .lower()
            )

            actual_table_name = schema_tables.get(
                table_key
            )

            if actual_table_name is None:
                continue

            descriptions = self._read_description_file(
                description_file
            )

            if not descriptions:
                continue

            table_metadata = schema[actual_table_name]

            for column in table_metadata.get(
                "columns",
                [],
            ):

                column_name = str(
                    column.get("name", "")
                ).strip()

                if not column_name:
                    continue

                metadata = descriptions.get(
                    column_name.strip().lower()
                )

                if metadata is None:
                    continue

                description = metadata.get(
                    "description",
                    "",
                ).strip()

                value_description = metadata.get(
                    "value_description",
                    "",
                ).strip()

                data_format = metadata.get(
                    "data_format",
                    "",
                ).strip()

                pieces: list[str] = []

                if description:
                    pieces.append(
                        description
                    )

                if data_format:
                    pieces.append(
                        f"Format: {data_format}"
                    )

                if value_description:
                    pieces.append(
                        f"Known values: {value_description}"
                    )

                if pieces:
                    column["description"] = " ".join(
                        pieces
                    )

    @staticmethod
    def _read_description_file(
        description_file: Path,
    ) -> dict[str, dict[str, str]]:
        """
        Read one BIRD database_description CSV.

        Returns:

            {
                "cdscode": {
                    "description": "...",
                    "data_format": "...",
                    "value_description": "..."
                }
            }
        """

        result: dict[
            str,
            dict[str, str],
        ] = {}

        try:
            with description_file.open(
                "r",
                encoding="utf-8-sig",
                newline="",
            ) as handle:

                reader = csv.DictReader(
                    handle
                )

                if reader.fieldnames is None:
                    return result

                # Normalize header names.
                headers = {
                    str(header)
                    .strip()
                    .lower(): header
                    for header in reader.fieldnames
                    if header is not None
                }

                original_column_header = (
                    headers.get(
                        "original_column_name"
                    )
                )

                description_header = (
                    headers.get(
                        "column_description"
                    )
                )

                data_format_header = (
                    headers.get(
                        "data_format"
                    )
                )

                value_description_header = (
                    headers.get(
                        "value_description"
                    )
                )

                if original_column_header is None:
                    return result

                for row in reader:

                    raw_column_name = row.get(
                        original_column_header
                    )

                    if raw_column_name is None:
                        continue

                    column_name = str(
                        raw_column_name
                    ).strip()

                    if not column_name:
                        continue

                    description = ""

                    if description_header is not None:
                        description = str(
                            row.get(
                                description_header
                            )
                            or ""
                        ).strip()

                    data_format = ""

                    if data_format_header is not None:
                        data_format = str(
                            row.get(
                                data_format_header
                            )
                            or ""
                        ).strip()

                    value_description = ""

                    if value_description_header is not None:
                        value_description = str(
                            row.get(
                                value_description_header
                            )
                            or ""
                        ).strip()

                    result[
                        column_name.lower()
                    ] = {
                        "description": description,
                        "data_format": data_format,
                        "value_description": (
                            value_description
                        ),
                    }

        except (
            OSError,
            UnicodeError,
            csv.Error,
        ):
            return {}

        return result

    def execute(
        self,
        sql: str,
    ):
        """
        Execute a read-only SQL query.

        Returns a pandas DataFrame because the existing
        SQLExecutor expects DataSource.execute() to return
        dataframe-like tabular data.
        """

        normalized_sql = (
            sql.strip().lower()
        )

        if not (
            normalized_sql.startswith("select")
            or normalized_sql.startswith("with")
            or normalized_sql.startswith("pragma")
        ):
            raise ValueError(
                "SQLiteDataSource only allows "
                "read-only SQL queries."
            )

        import pandas as pd

        with sqlite3.connect(
            f"file:{self.database_path.resolve()}?mode=ro",
            uri=True,
        ) as connection:

            return pd.read_sql_query(
                sql,
                connection,
            )