from unittest.mock import MagicMock, patch

from data_sources.postgres_source import PostgresDataSource
from schema.retriever import SchemaRetriever


def test_postgres_source_returns_primary_and_foreign_key_metadata():
    connection = MagicMock()
    connection.execute.side_effect = [
        MagicMock(
            fetchall=MagicMock(
                return_value=[
                    ("public", "customers", "customer_id", "integer", "NO"),
                    ("public", "orders", "order_id", "integer", "NO"),
                    ("public", "orders", "customer_id", "integer", "NO"),
                ]
            )
        ),
        MagicMock(
            fetchall=MagicMock(
                return_value=[
                    ("public", "customers", "customer_id"),
                    ("public", "orders", "order_id"),
                ]
            )
        ),
        MagicMock(
            fetchall=MagicMock(
                return_value=[
                    (
                        "public",
                        "orders",
                        "customer_id",
                        "public",
                        "customers",
                        "customer_id",
                    )
                ]
            )
        ),
    ]

    with patch(
        "data_sources.postgres_source.psycopg.connect"
    ) as connect:
        connect.return_value.__enter__.return_value = connection

        source = PostgresDataSource(
            connection_string="postgresql://test:test@localhost/test"
        )
        schema = source.get_schema()

    customers_id = next(
        item
        for item in schema
        if item["table_name"] == "customers"
    )
    orders_customer_id = next(
        item
        for item in schema
        if item["table_name"] == "orders"
        and item["column_name"] == "customer_id"
    )

    assert customers_id["primary_key"] is True
    assert orders_customer_id["primary_key"] is False
    assert orders_customer_id["foreign_keys"] == [
        {
            "referenced_table_schema": "public",
            "referenced_table": "customers",
            "referenced_column": "customer_id",
        }
    ]


def test_schema_retriever_preserves_postgres_relationships():
    source = MagicMock()
    source.get_schema.return_value = [
        {
            "table_schema": "public",
            "table_name": "customers",
            "column_name": "customer_id",
            "data_type": "integer",
            "nullable": False,
            "primary_key": True,
            "foreign_keys": [],
        },
        {
            "table_schema": "public",
            "table_name": "orders",
            "column_name": "customer_id",
            "data_type": "integer",
            "nullable": False,
            "primary_key": False,
            "foreign_keys": [
                {
                    "referenced_table_schema": "public",
                    "referenced_table": "customers",
                    "referenced_column": "customer_id",
                }
            ],
        },
    ]

    schema = SchemaRetriever().from_source(
        source,
        source_type="postgresql",
    )

    customers = next(table for table in schema.tables if table.name == "customers")
    orders = next(table for table in schema.tables if table.name == "orders")

    assert next(
        column for column in customers.columns
        if column.name == "customer_id"
    ).primary_key is True

    assert len(orders.foreign_keys) == 1
    assert orders.foreign_keys[0].column_name == "customer_id"
    assert orders.foreign_keys[0].referenced_table == "customers"
    assert orders.foreign_keys[0].referenced_column == "customer_id"