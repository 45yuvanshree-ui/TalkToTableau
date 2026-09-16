import time
from pathlib import Path

import pytest

from orchestration.workflow import TalkToTableauWorkflow


@pytest.mark.live_llm
def test_real_orchestration_with_configured_provider():
    """
    Verify the complete real orchestration using the provider
    configured through the environment.
    """

    data_file = Path("database/golden_sales.csv")

    workflow = TalkToTableauWorkflow(
        data_file=data_file
    )

    started = time.perf_counter()

    context = workflow.run(
        "What was the total revenue by region?"
    )

    total_time = time.perf_counter() - started

    print(
        f"\nTOTAL WORKFLOW TIME: "
        f"{total_time:.2f} seconds"
    )

    print("\n" + "=" * 60)
    print("GENERATED SQL")
    print("=" * 60)
    print(context.generated_sql)

    print("\n" + "=" * 60)
    print("SQL RESULT")
    print("=" * 60)
    print(
        context.sql_result.model_dump_json(
            indent=2
        )
    )

    print("\n" + "=" * 60)
    print("AGENT 2 — VERIFIED INSIGHTS")
    print("=" * 60)
    print(
        context.insights.model_dump_json(
            indent=2
        )
    )

    print("\n" + "=" * 60)
    print("VISUALIZATION")
    print("=" * 60)
    print(
        context.visualization.model_dump_json(
            indent=2
        )
    )

    print("\n" + "=" * 60)
    print("AGENT 3 — BUSINESS STORY")
    print("=" * 60)
    print(
        context.story.model_dump_json(
            indent=2
        )
    )

    print("\n" + "=" * 60)
    print("WORKFLOW COMPLETE")
    print("=" * 60)

    assert context.analytical_plan is not None
    assert context.generated_sql
    assert context.sql_result is not None
    assert context.sql_result.row_count > 0

    assert context.analysis_result is not None
    assert context.insights is not None
    assert context.visualization is not None
    assert context.story is not None