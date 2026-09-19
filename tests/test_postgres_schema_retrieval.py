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
                def fetchall(self):
                    return [
                        (
                            "public",
                            "sales",
                            "sale_id",
                            "integer",
                        ),
                        (
                            "public",
                            "sales",
                            "sale_date",
                            "date",
                        ),
                        (
                            "public",
                            "sales",
                            "product",
                            "character varying",
                        ),
                        (
                            "public",
                            "sales",
                            "revenue",
                            "numeric",
                        ),
                    ]

            return FakeResult()

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