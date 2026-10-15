
from data_sources.postgres_source import PostgresDataSource
from schema import SchemaRetriever


def test_postgres_schema_is_normalized(monkeypatch):
    class FakeConnection:
        def __enter__(self):
            return self

        def __exit__(
            self,
            exc_type,
            exc_value,
            traceback,
        ):
            return False

        def execute(self, query):
            class FakeResult:
                def __init__(self, rows):
                    self.rows = rows

                def fetchall(self):
                    return self.rows

            # Query 1: column metadata.
            if "information_schema.columns" in query:
                return FakeResult(
                    [
                        (
                            "public",
                            "sales",
                            "sale_id",
                            "integer",
                            "NO",
                        ),
                        (
                            "public",
                            "sales",
                            "sale_date",
                            "date",
                            "YES",
                        ),
                        (
                            "public",
                            "sales",
                            "product",
                            "character varying",
                            "YES",
                        ),
                        (
                            "public",
                            "sales",
                            "revenue",
                            "numeric",
                            "NO",
                        ),
                    ]
                )

            # Query 2: primary-key metadata.
            if "PRIMARY KEY" in query:
                return FakeResult(
                    [
                        ("public", "sales", "sale_id"),
                    ]
                )

            # Query 3: foreign-key metadata.
            if "FOREIGN KEY" in query:
                return FakeResult([])

            raise AssertionError(
                f"Unexpected PostgreSQL schema query: {query}"
            )

    monkeypatch.setattr(
        "data_sources.postgres_source.psycopg.connect",
        lambda *args, **kwargs: FakeConnection(),
    )

    source = PostgresDataSource(
        "postgresql://user:password@localhost:5432/testdb"
    )

    retriever = SchemaRetriever()

    schema = retriever.from_source(
        source,
        source_type="postgresql",
    )

    assert schema.source_type == "postgresql"
    assert len(schema.tables) == 1

    table = schema.tables[0]

    assert table.name == "sales"

    column_names = [
        column.name
        for column in table.columns
    ]

    assert column_names == [
        "sale_id",
        "sale_date",
        "product",
        "revenue",
    ]

    # Ensure PostgreSQL primary-key metadata survives normalization.
    sale_id_column = next(
        column
        for column in table.columns
        if column.name == "sale_id"
    )

    assert sale_id_column.primary_key is True

    # No foreign keys were returned by this fixture.
    assert table.foreign_keys == []
