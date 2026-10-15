
from models import AnalyticalPlan
from schema import ColumnSchema, DataSchema, TableSchema
from schema.models import ForeignKeySchema
from sql import SQLGenerationOutput, generate_sql


class FakeModel:
    """Dummy model used by tests with a mocked agent factory."""

    pass


def test_sql_generation_output_model():
    """Verify the structured SQL output model."""

    result = SQLGenerationOutput(
        query=(
            "SELECT region, SUM(revenue) AS total_revenue "
            "FROM golden_sales "
            "GROUP BY region"
        )
    )

    assert result.query.startswith("SELECT")


def create_product_schema() -> DataSchema:
    """Create a minimal schema with a declared product foreign key."""

    order_items = TableSchema(
        name="olist_order_items_dataset",
        source_type="postgresql",
        columns=[
            ColumnSchema(name="order_id", data_type="text"),
            ColumnSchema(name="product_id", data_type="text"),
            ColumnSchema(name="price", data_type="numeric"),
        ],
        foreign_keys=[
            ForeignKeySchema(
                column_name="product_id",
                referenced_table="olist_products_dataset",
                referenced_column="product_id",
            )
        ],
    )

    products = TableSchema(
        name="olist_products_dataset",
        source_type="postgresql",
        columns=[
            ColumnSchema(name="product_id", data_type="text"),
            ColumnSchema(
                name="product_category_name",
                data_type="text",
            ),
        ],
    )

    return DataSchema(
        source_type="postgresql",
        tables=[order_items, products],
    )


def create_product_category_plan() -> AnalyticalPlan:
    """Create a plan that genuinely requires two tables."""

    return AnalyticalPlan(
        analytical_objective=(
            "Calculate total item price by product category"
        ),
        operation="aggregation",
        metric_definition="Total item price",
        metrics=["SUM(price)"],
        dimensions=["product_category_name"],
        filters=[],
        analytical_question=(
            "What is the total item price by product category?"
        ),
    )


def mock_sql_generator_dependencies(
    monkeypatch,
    query: str,
    captured: dict | None = None,
):
    """Mock agent creation and model execution for deterministic tests."""

    def fake_runner_run_sync(agent, prompt):
        if captured is not None:
            captured["agent"] = agent
            captured["prompt"] = prompt

        return type(
            "FakeRunResult",
            (),
            {
                "final_output": SQLGenerationOutput(query=query),
            },
        )()

    # Avoid SDK model validation because these tests do not call a real LLM.
    monkeypatch.setattr(
        "sql.generator.create_sql_generator",
        lambda model: object(),
    )

    monkeypatch.setattr(
        "sql.generator.Runner.run_sync",
        fake_runner_run_sync,
    )


def test_sql_generator_prompt_includes_declared_foreign_key(
    monkeypatch,
):
    """Verify that the prompt contains the declared relationship."""

    captured = {}

    expected_query = (
        "SELECT p.product_category_name, "
        "SUM(oi.price) AS total_item_price "
        "FROM olist_order_items_dataset AS oi "
        "JOIN olist_products_dataset AS p "
        "ON oi.product_id = p.product_id "
        "GROUP BY p.product_category_name"
    )

    mock_sql_generator_dependencies(
        monkeypatch=monkeypatch,
        query=expected_query,
        captured=captured,
    )

    query = generate_sql(
        model=FakeModel(),
        user_request=(
            "What is the total item price by product category?"
        ),
        plan=create_product_category_plan(),
        schema=create_product_schema(),
    )

    prompt = captured["prompt"]

    assert "The data source is PostgreSQL." in prompt
    assert "ACTUAL POSTGRESQL TABLES AND COLUMNS:" in prompt
    assert "Do NOT use a synthetic table named \"data\"." in prompt
    assert "The data source is file-based." not in prompt
    assert "The available DuckDB table is:" not in prompt


    assert "FOREIGN KEYS:" in prompt
    assert (
        "product_id -> olist_products_dataset.product_id"
        in prompt
    )
    assert "product_category_name" in prompt
    assert "SUM(oi.price)" in query
    assert "JOIN olist_products_dataset" in query
    assert "oi.product_id = p.product_id" in query


def test_sql_generator_returns_join_from_model_output(
    monkeypatch,
):
    """Verify that generated JOIN SQL passes through the public API."""

    expected_query = (
        "SELECT p.product_category_name, "
        "SUM(oi.price) AS total_item_price "
        "FROM olist_order_items_dataset AS oi "
        "JOIN olist_products_dataset AS p "
        "ON oi.product_id = p.product_id "
        "GROUP BY p.product_category_name"
    )

    mock_sql_generator_dependencies(
        monkeypatch=monkeypatch,
        query=expected_query,
    )

    query = generate_sql(
        model=FakeModel(),
        user_request=(
            "What is the total item price by product category?"
        ),
        plan=create_product_category_plan(),
        schema=create_product_schema(),
    )

    assert query == expected_query
