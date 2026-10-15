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

    def get_schema(self) -> list[dict[str, object]]:
        """
        Return PostgreSQL column metadata, including primary and
        foreign keys.
        """
        columns_query = """
            SELECT
                table_schema,
                table_name,
                column_name,
                data_type,
                is_nullable
            FROM information_schema.columns
            WHERE table_schema NOT IN (
                'pg_catalog',
                'information_schema'
            )
            ORDER BY table_schema, table_name, ordinal_position;
        """

        primary_keys_query = """
            SELECT
                tc.table_schema,
                tc.table_name,
                kcu.column_name
            FROM information_schema.table_constraints AS tc
            JOIN information_schema.key_column_usage AS kcu
              ON kcu.constraint_catalog = tc.constraint_catalog
             AND kcu.constraint_schema = tc.constraint_schema
             AND kcu.constraint_name = tc.constraint_name
             AND kcu.table_schema = tc.table_schema
             AND kcu.table_name = tc.table_name
            WHERE tc.constraint_type = 'PRIMARY KEY'
              AND tc.table_schema NOT IN (
                  'pg_catalog',
                  'information_schema'
              );
        """



        foreign_keys_query = """
            SELECT
                kcu.table_schema,
                kcu.table_name,
                kcu.column_name,
                ccu.table_schema AS referenced_table_schema,
                ccu.table_name AS referenced_table,
                ccu.column_name AS referenced_column
            FROM information_schema.table_constraints AS tc
            JOIN information_schema.key_column_usage AS kcu
              ON kcu.constraint_catalog = tc.constraint_catalog
             AND kcu.constraint_schema = tc.constraint_schema
             AND kcu.constraint_name = tc.constraint_name
             AND kcu.table_schema = tc.table_schema
             AND kcu.table_name = tc.table_name
            JOIN information_schema.constraint_column_usage AS ccu
              ON ccu.constraint_catalog = tc.constraint_catalog
             AND ccu.constraint_schema = tc.constraint_schema
             AND ccu.constraint_name = tc.constraint_name
            WHERE tc.constraint_type = 'FOREIGN KEY'
              AND tc.table_schema NOT IN (
                  'pg_catalog',
                  'information_schema'
              )
            ORDER BY
                kcu.table_schema,
                kcu.table_name,
                kcu.ordinal_position;
        """



        with psycopg.connect(self.connection_string) as connection:
            column_rows = connection.execute(
                columns_query
            ).fetchall()

            primary_key_rows = connection.execute(
                primary_keys_query
            ).fetchall()

            foreign_key_rows = connection.execute(
                foreign_keys_query
            ).fetchall()

        primary_keys = {
            (str(row[0]), str(row[1]), str(row[2]))
            for row in primary_key_rows
        }

        foreign_keys_by_column: dict[
            tuple[str, str, str], list[dict[str, str]]
        ] = {}

        for row in foreign_key_rows:
            (
                table_schema,
                table_name,
                column_name,
                referenced_schema,
                referenced_table,
                referenced_column,
            ) = row

            key = (
                str(table_schema),
                str(table_name),
                str(column_name),
            )

            foreign_keys_by_column.setdefault(key, []).append(
                {
                    "referenced_table_schema": str(referenced_schema),
                    "referenced_table": str(referenced_table),
                    "referenced_column": str(referenced_column),
                }
            )

        schema: list[dict[str, object]] = []

        for row in column_rows:
            (
                table_schema,
                table_name,
                column_name,
                data_type,
                is_nullable,
            ) = row

            key = (
                str(table_schema),
                str(table_name),
                str(column_name),
            )

            schema.append(
                {
                    "table_schema": str(table_schema),
                    "table_name": str(table_name),
                    "column_name": str(column_name),
                    "data_type": str(data_type),
                    "nullable": is_nullable == "YES",
                    "primary_key": key in primary_keys,
                    "foreign_keys": foreign_keys_by_column.get(
                        key, []
                    ),
                }
            )

        return schema

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