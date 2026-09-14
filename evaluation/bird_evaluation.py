from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from orchestration.workflow import TalkToTableauWorkflow
from sql.executor import SQLExecutor


DATA_DIR = Path("data/benchmarks/bird")

QUESTIONS_FILE = DATA_DIR / "mini_dev_sqlite_50.json"
DATABASES_DIR = DATA_DIR / "dev_databases"

OUTPUT_DIR = Path("evaluation/results")
OUTPUT_FILE = OUTPUT_DIR / "bird_mini_dev_50_results.json"

BENCHMARK_SIZE = 50


def utc_now() -> str:
    """Return the current UTC timestamp."""

    return datetime.now(
        timezone.utc
    ).isoformat()


def load_questions() -> list[dict[str, Any]]:
    """Load the selected BIRD Mini-Dev pilot questions."""

    if not QUESTIONS_FILE.exists():
        raise FileNotFoundError(
            f"BIRD question file not found: "
            f"{QUESTIONS_FILE}"
        )

    with QUESTIONS_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:
        questions = json.load(file)

    if not isinstance(questions, list):
        raise ValueError(
            "BIRD question file must contain a JSON list."
        )

    return questions


def find_database(database_id: str) -> Path:
    """Return the SQLite database path for a BIRD database ID."""

    database_path = (
        DATABASES_DIR
        / database_id
        / f"{database_id}.sqlite"
    )

    if not database_path.exists():
        raise FileNotFoundError(
            f"SQLite database not found: "
            f"{database_path}"
        )

    return database_path


def normalize_value(value: Any) -> Any:
    """
    Normalize values so generated and gold SQL results
    can be compared reliably.
    """

    if value is None:
        return None

    if isinstance(value, bool):
        return bool(value)

    if isinstance(value, (int, float)):
        return float(value)

    return str(value).strip()


def normalize_result(
    columns: list[str],
    rows: list[list[Any]],
) -> list[tuple[Any, ...]]:
    """
    Normalize SQL rows for deterministic comparison.

    Column names are deliberately excluded because
    equivalent SQL queries may use different aliases.
    """

    normalized_rows: list[tuple[Any, ...]] = []

    for row in rows:
        normalized_rows.append(
            tuple(
                normalize_value(value)
                for value in row
            )
        )

    return normalized_rows


def results_equal(
    generated_columns: list[str],
    generated_rows: list[list[Any]],
    gold_columns: list[str],
    gold_rows: list[list[Any]],
) -> bool:
    """
    Compare generated and gold SQL results.

    Row ordering is preserved because the benchmark's
    gold SQL may intentionally specify an ordering.
    """

    generated = normalize_result(
        generated_columns,
        generated_rows,
    )

    gold = normalize_result(
        gold_columns,
        gold_rows,
    )

    if len(generated) != len(gold):
        return False

    if not generated and not gold:
        return True

    if not generated or not gold:
        return False

    if len(generated[0]) != len(gold[0]):
        return False

    for generated_row, gold_row in zip(
        generated,
        gold,
    ):
        if len(generated_row) != len(gold_row):
            return False

        for generated_value, gold_value in zip(
            generated_row,
            gold_row,
        ):
            if (
                isinstance(generated_value, float)
                and isinstance(gold_value, float)
            ):
                difference = abs(
                    generated_value
                    - gold_value
                )

                tolerance = max(
                    1e-9,
                    abs(gold_value) * 1e-9,
                )

                if difference > tolerance:
                    return False

            elif generated_value != gold_value:
                return False

    return True


def execute_gold_sql(
    executor: SQLExecutor,
    database_path: Path,
    gold_sql: str,
):
    """Execute the benchmark's gold SQL."""

    return executor.execute_sqlite(
        database_path=database_path,
        query=gold_sql,
    )


def extract_workflow_data(
    result: Any,
) -> dict[str, Any]:
    """
    Extract useful evaluation information from a
    TalkToTableauWorkflow result.

    SessionContext currently exposes generated_sql,
    sql_result, analysis_result, insights,
    visualization, story, analytical_plan, etc.
    """

    data: dict[str, Any] = {}

    attributes = [
        "sql_query",
        "query",
        "generated_sql",
        "sql_result",
        "analysis_result",
        "insight_analysis",
        "insights",
        "visualization",
        "story",
        "trace_id",
        "session_id",
        "analytical_plan",
        "data_schema",
        "data_source",
    ]

    for attribute in attributes:
        if not hasattr(result, attribute):
            continue

        value = getattr(
            result,
            attribute,
        )

        if hasattr(value, "model_dump"):
            value = value.model_dump()

        data[attribute] = value

    return data


def extract_generated_sql(
    result_data: dict[str, Any],
) -> str | None:
    """Extract generated SQL from workflow output."""

    for key in [
        "sql_query",
        "generated_sql",
        "query",
    ]:
        value = result_data.get(key)

        if isinstance(value, str) and value.strip():
            return value.strip()

    sql_result = result_data.get(
        "sql_result"
    )

    if isinstance(sql_result, dict):
        value = sql_result.get("query")

        if isinstance(value, str) and value.strip():
            return value.strip()

    return None


def exception_chain(
    error: BaseException,
) -> list[BaseException]:
    """
    Return the exception and its explicit cause/context chain.

    Workflows commonly wrap lower-level provider exceptions
    inside higher-level RuntimeError instances. Preserving the
    chain allows the evaluator to identify provider quota
    exhaustion without changing the workflow architecture.
    """

    chain: list[BaseException] = []

    current: BaseException | None = error

    visited: set[int] = set()

    while current is not None:
        current_id = id(current)

        if current_id in visited:
            break

        visited.add(current_id)
        chain.append(current)

        cause = current.__cause__

        if cause is not None:
            current = cause
            continue

        context = current.__context__

        if context is not None:
            current = context
            continue

        break

    return chain


def is_provider_quota_exhaustion(
    error: BaseException,
) -> bool:
    """
    Detect explicit provider quota exhaustion anywhere in an
    exception chain.

    This is intentionally specific. Generic RuntimeError,
    provider errors, or HTTP 429 errors are not enough by
    themselves to classify a question as quota exhausted.
    """

    quota_patterns = (
        "free-models-per-day",
        "free models per day",
        "free-model daily limit",
        "free model daily limit",
        "daily free-model limit",
        "daily free model limit",
        "daily quota exhausted",
        "daily quota exceeded",
        "quota exhausted",
        "quota exceeded",
        "quota has been exhausted",
        "allowance has been exhausted",
        "openrouter_free_tier_daily",
    )

    for chained_error in exception_chain(error):
        message = str(
            chained_error
        ).strip().lower()

        if not message:
            continue

        if any(
            pattern in message
            for pattern in quota_patterns
        ):
            return True

        attempts = getattr(
            chained_error,
            "attempts",
            None,
        )

        if not isinstance(
            attempts,
            list,
        ):
            continue

        for attempt in attempts:
            error_message = getattr(
                attempt,
                "error_message",
                None,
            )

            if not isinstance(
                error_message,
                str,
            ):
                continue

            normalized_message = (
                error_message
                .strip()
                .lower()
            )

            if any(
                pattern in normalized_message
                for pattern in quota_patterns
            ):
                return True

    return False


def classify_status(
    record: dict[str, Any],
) -> str:
    """
    Classify the final evaluation state of one question.

    Provider quota exhaustion has explicit priority over the
    normal workflow/result classifications because the question
    could not be evaluated due to an external provider limit.
    """

    if record.get("provider_quota_exhausted", False):
        return "PROVIDER_QUOTA_EXHAUSTED"

    if record["result_correct"]:
        return "PASS"

    if record["sql_executed"]:
        return "WRONG"

    if record["sql_generated"]:
        return "SQL_EXECUTION_FAIL"

    if record["workflow_success"]:
        return "WORKFLOW_NO_SQL"

    return "PIPELINE_FAIL"


def create_question_record(
    question: dict[str, Any],
) -> dict[str, Any]:
    """Create the initial checkpoint record for one question."""

    question_id = question["question_id"]
    database_id = question["db_id"]
    difficulty = question["difficulty"]
    user_question = question["question"]
    gold_sql = question["SQL"].strip()

    return {
        "question_id": question_id,
        "database": database_id,
        "difficulty": difficulty,
        "question": user_question,
        "gold_sql": gold_sql,

        "workflow_success": False,
        "sql_generated": False,
        "sql_executed": False,
        "gold_sql_executed": False,
        "result_correct": False,

        "provider_quota_exhausted": False,

        "generated_sql": None,
        "generated_result": None,
        "gold_result": None,

        "analysis_result": None,
        "insights": None,
        "visualization": None,
        "story": None,

        "session_id": None,
        "trace_id": None,

        "error_type": None,
        "error": None,

        "elapsed_seconds": None,
        "status": "NOT_STARTED",

        "started_at": None,
        "completed_at": None,
    }


def save_checkpoint(
    document: dict[str, Any],
) -> None:
    """
    Atomically save the current evaluation document.

    The temporary file prevents a partial JSON document from
    replacing the previous valid checkpoint.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_file = OUTPUT_FILE.with_suffix(
        ".tmp"
    )

    with temporary_file.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            document,
            file,
            indent=2,
            ensure_ascii=False,
            default=str,
        )

    temporary_file.replace(
        OUTPUT_FILE
    )


def load_checkpoint() -> dict[str, Any]:
    """Load an existing checkpoint."""

    if not OUTPUT_FILE.exists():
        raise FileNotFoundError(
            f"Checkpoint file not found: "
            f"{OUTPUT_FILE}"
        )

    with OUTPUT_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:
        document = json.load(file)

    if not isinstance(document, dict):
        raise ValueError(
            "Evaluation checkpoint must contain a JSON object."
        )

    results = document.get(
        "results"
    )

    if not isinstance(results, list):
        raise ValueError(
            "Evaluation checkpoint has no valid results list."
        )

    return document


def build_result_index(
    results: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Index checkpoint records by question ID."""

    index: dict[str, dict[str, Any]] = {}

    for record in results:
        question_id = str(
            record.get("question_id")
        )

        index[question_id] = record

    return index


def is_completed_record(
    record: dict[str, Any],
) -> bool:
    """
    Return True when a question has already reached a final
    evaluable state and therefore can safely be skipped during
    resume.

    Provider quota exhaustion is included because retrying the
    same provider while its daily allowance is exhausted cannot
    produce a meaningful benchmark result.
    """

    return record.get("status") in {
        "PASS",
        "WRONG",
        "SQL_EXECUTION_FAIL",
        "PIPELINE_FAIL",
        "WORKFLOW_NO_SQL",
        "PROVIDER_QUOTA_EXHAUSTED",
    }


def calculate_summary(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Calculate evaluation metrics from question records."""

    total = len(results)

    workflow_success_count = sum(
        bool(result.get("workflow_success"))
        for result in results
    )

    sql_generated_count = sum(
        bool(result.get("sql_generated"))
        for result in results
    )

    sql_executed_count = sum(
        bool(result.get("sql_executed"))
        for result in results
    )

    gold_sql_executed_count = sum(
        bool(result.get("gold_sql_executed"))
        for result in results
    )

    correct_count = sum(
        bool(result.get("result_correct"))
        for result in results
    )

    provider_quota_exhausted_count = sum(
        bool(
            result.get(
                "provider_quota_exhausted"
            )
        )
        for result in results
    )

    failed_count = sum(
        not is_completed_record(result)
        for result in results
    )

    evaluable_count = sum(
        result.get("status")
        not in {
            "PROVIDER_QUOTA_EXHAUSTED",
            "NOT_STARTED",
        }
        for result in results
    )

    evaluable_correct_count = sum(
        bool(result.get("result_correct"))
        for result in results
        if result.get("status")
        != "PROVIDER_QUOTA_EXHAUSTED"
    )

    elapsed_values = [
        float(result["elapsed_seconds"])
        for result in results
        if isinstance(
            result.get("elapsed_seconds"),
            (int, float),
        )
    ]

    average_elapsed = (
        sum(elapsed_values)
        / len(elapsed_values)
        if elapsed_values
        else 0.0
    )

    status_counts: dict[str, int] = {}

    for result in results:
        status = str(
            result.get(
                "status",
                "UNKNOWN",
            )
        )

        status_counts[status] = (
            status_counts.get(status, 0)
            + 1
        )

    difficulty_summary: dict[str, dict[str, Any]] = {}

    for difficulty in [
        "simple",
        "moderate",
        "challenging",
    ]:
        difficulty_results = [
            result
            for result in results
            if str(
                result.get("difficulty", "")
            ).lower()
            == difficulty
        ]

        if not difficulty_results:
            continue

        difficulty_correct = sum(
            bool(
                result.get(
                    "result_correct"
                )
            )
            for result in difficulty_results
        )

        difficulty_quota_exhausted = sum(
            result.get("status")
            == "PROVIDER_QUOTA_EXHAUSTED"
            for result in difficulty_results
        )

        difficulty_evaluable = (
            len(difficulty_results)
            - difficulty_quota_exhausted
        )

        difficulty_summary[difficulty] = {
            "total": len(
                difficulty_results
            ),
            "correct": difficulty_correct,
            "provider_quota_exhausted": (
                difficulty_quota_exhausted
            ),
            "evaluable": difficulty_evaluable,
            "accuracy_percent": round(
                difficulty_correct
                / difficulty_evaluable
                * 100,
                2,
            )
            if difficulty_evaluable
            else 0.0,
        }

    return {
        "total_questions": total,
        "completed_questions": (
            total - failed_count
        ),
        "workflow_success": (
            workflow_success_count
        ),
        "sql_generated": (
            sql_generated_count
        ),
        "sql_executed": (
            sql_executed_count
        ),
        "gold_sql_executed": (
            gold_sql_executed_count
        ),
        "correct_results": correct_count,
        "result_accuracy_percent": (
            round(
                correct_count / total * 100,
                2,
            )
            if total
            else 0.0
        ),
        "evaluable_questions": (
            evaluable_count
        ),
        "evaluable_correct_results": (
            evaluable_correct_count
        ),
        "evaluable_accuracy_percent": (
            round(
                evaluable_correct_count
                / evaluable_count
                * 100,
                2,
            )
            if evaluable_count
            else 0.0
        ),
        "provider_quota_exhausted": (
            provider_quota_exhausted_count
        ),
        "average_question_time_seconds": round(
            average_elapsed,
            3,
        ),
        "status_counts": status_counts,
        "difficulty": difficulty_summary,
    }


def parse_arguments() -> argparse.Namespace:
    """Parse evaluator command-line options."""

    parser = argparse.ArgumentParser(
        description=(
            "Run the TalkToTableau BIRD Mini-Dev evaluation."
        )
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume from the existing checkpoint instead "
            "of starting a fresh evaluation."
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Evaluate only the first N questions. "
            "Useful for safe pilot testing."
        ),
    )

    return parser.parse_args()


def initialize_document(
    questions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Create a fresh evaluation document."""

    return {
        "benchmark": "BIRD Mini-Dev",
        "benchmark_size": BENCHMARK_SIZE,
        "pilot_size": len(questions),
        "evaluation_type": (
            "TalkToTableau end-to-end evaluation"
        ),
        "started_at": utc_now(),
        "last_checkpoint_at": utc_now(),
        "completed_at": None,
        "resumed": False,
        "results": [
            create_question_record(question)
            for question in questions
        ],
        "summary": calculate_summary([]),
    }


def prepare_document(
    questions: list[dict[str, Any]],
    resume: bool,
) -> dict[str, Any]:
    """
    Create or load the evaluation document.

    A fresh run is the default. This is important because
    an old checkpoint may belong to an earlier model/provider
    configuration.
    """

    if not resume:
        return initialize_document(
            questions
        )

    document = load_checkpoint()

    existing_results = document.get(
        "results",
        [],
    )

    existing_index = build_result_index(
        existing_results
    )

    ordered_results: list[dict[str, Any]] = []

    for question in questions:
        question_id = str(
            question["question_id"]
        )

        existing = existing_index.get(
            question_id
        )

        if existing is None:
            ordered_results.append(
                create_question_record(
                    question
                )
            )
        else:
            # Backward compatibility for checkpoints
            # created before provider quota tracking existed.
            existing.setdefault(
                "provider_quota_exhausted",
                False,
            )

            ordered_results.append(
                existing
            )

    document["results"] = ordered_results
    document["resumed"] = True
    document["last_checkpoint_at"] = utc_now()

    return document


def run_question(
    question: dict[str, Any],
    executor: SQLExecutor,
) -> dict[str, Any]:
    """
    Run one BIRD question through the complete
    TalkToTableau workflow.
    """

    database_id = question["db_id"]
    user_question = question["question"]
    gold_sql = question["SQL"].strip()

    database_path = find_database(
        database_id
    )

    record = create_question_record(
        question
    )

    record["started_at"] = utc_now()

    question_start = time.perf_counter()

    try:
        workflow = TalkToTableauWorkflow(
            data_file=database_path
        )

        workflow_result = workflow.run(
            user_question
        )

        record.update(
            extract_workflow_data(
                workflow_result
            )
        )

        # Preserve the completed application trace so the
        # evaluation record can be used for stage-level
        # latency and model-attempt analysis.
        if workflow.last_trace is not None:
            record["trace_id"] = (
                workflow.last_trace.trace_id
            )
            record["trace"] = (
                workflow.last_trace.model_dump(
                    mode="json"
                )
            )

        generated_sql = (
            extract_generated_sql(
                record
            )
        )

        record["generated_sql"] = (
            generated_sql
        )

        record["sql_generated"] = (
            generated_sql is not None
        )

        record["workflow_success"] = True

        if generated_sql is None:
            record["error_type"] = (
                "SQL_EXTRACTION"
            )

            record["error"] = (
                "Workflow completed but generated SQL "
                "could not be extracted."
            )

            return record

        try:
            generated_result = (
                executor.execute_sqlite(
                    database_path=database_path,
                    query=generated_sql,
                )
            )

            record["sql_executed"] = True

            record["generated_result"] = (
                generated_result.model_dump()
            )

        except Exception as exc:
            record["error_type"] = (
                "GENERATED_SQL_EXECUTION"
            )

            record["error"] = (
                f"{type(exc).__name__}: {exc}"
            )

            return record

        try:
            gold_result = execute_gold_sql(
                executor=executor,
                database_path=database_path,
                gold_sql=gold_sql,
            )

            record["gold_sql_executed"] = True

            record["gold_result"] = (
                gold_result.model_dump()
            )

        except Exception as exc:
            record["error_type"] = (
                "GOLD_SQL_EXECUTION"
            )

            record["error"] = (
                f"{type(exc).__name__}: {exc}"
            )

            return record

        record["result_correct"] = (
            results_equal(
                generated_columns=(
                    generated_result.columns
                ),
                generated_rows=(
                    generated_result.rows
                ),
                gold_columns=(
                    gold_result.columns
                ),
                gold_rows=(
                    gold_result.rows
                ),
            )
        )

        if not record["result_correct"]:
            record["error_type"] = (
                "RESULT_MISMATCH"
            )

            record["error"] = (
                "Generated SQL executed successfully, "
                "but its result did not match the gold result."
            )

    except Exception as exc:
        record["error_type"] = (
            "WORKFLOW"
        )

        record["error"] = (
            f"{type(exc).__name__}: {exc}"
        )

        if is_provider_quota_exhaustion(
            exc
        ):
            record["provider_quota_exhausted"] = True
            record["error_type"] = (
                "PROVIDER_QUOTA_EXHAUSTED"
            )

            record["error"] = (
                "LLM provider quota or daily free-model "
                "allowance was exhausted before the workflow "
                "could complete."
            )

    finally:
        record["elapsed_seconds"] = round(
            time.perf_counter()
            - question_start,
            3,
        )

        record["status"] = classify_status(
            record
        )

        record["completed_at"] = utc_now()

    return record


def print_question_status(
    index: int,
    total: int,
    record: dict[str, Any],
) -> None:
    """Print a compact question status."""

    status = record.get(
        "status",
        "UNKNOWN",
    )

    elapsed = record.get(
        "elapsed_seconds"
    )

    if isinstance(elapsed, (int, float)):
        elapsed_text = (
            f"{float(elapsed):.3f}s"
        )
    else:
        elapsed_text = "n/a"

    question_id = record.get(
        "question_id"
    )

    database = record.get(
        "database"
    )

    difficulty = record.get(
        "difficulty"
    )

    print(
        f"       {status:<25} "
        f"({elapsed_text}) "
        f"Q{question_id} | "
        f"{database} | "
        f"{difficulty}"
    )


def print_summary(
    document: dict[str, Any],
) -> None:
    """Print the current evaluation summary."""

    summary = document.get(
        "summary",
        {},
    )

    print()
    print("=" * 70)
    print("Evaluation Summary")
    print("=" * 70)

    print(
        f"Questions          : "
        f"{summary.get('total_questions', 0)}"
    )

    print(
        f"Completed          : "
        f"{summary.get('completed_questions', 0)}"
    )

    print(
        f"Workflow success   : "
        f"{summary.get('workflow_success', 0)}"
    )

    print(
        f"SQL generated      : "
        f"{summary.get('sql_generated', 0)}"
    )

    print(
        f"SQL executed       : "
        f"{summary.get('sql_executed', 0)}"
    )

    print(
        f"Gold SQL executed  : "
        f"{summary.get('gold_sql_executed', 0)}"
    )

    print(
        f"Correct results    : "
        f"{summary.get('correct_results', 0)}"
    )

    print(
        f"Result accuracy    : "
        f"{summary.get('result_accuracy_percent', 0.0):.2f}%"
    )

    print(
        f"Evaluable questions: "
        f"{summary.get('evaluable_questions', 0)}"
    )

    print(
        f"Evaluable accuracy : "
        f"{summary.get('evaluable_accuracy_percent', 0.0):.2f}%"
    )

    print(
        f"Provider quota     : "
        f"{summary.get('provider_quota_exhausted', 0)}"
    )

    print(
        f"Average question time : "
        f"{summary.get('average_question_time_seconds', 0.0):.3f}s"
    )

    status_counts = summary.get(
        "status_counts",
        {},
    )

    if status_counts:
        print()
        print("Status breakdown:")

        for status, count in sorted(
            status_counts.items()
        ):
            print(
                f"  {status:<28}: {count}"
            )

    difficulty_summary = summary.get(
        "difficulty",
        {},
    )

    if difficulty_summary:
        print()
        print("Difficulty accuracy:")

        for difficulty, values in (
            difficulty_summary.items()
        ):
            print(
                f"  {difficulty:<12}: "
                f"{values['correct']}/"
                f"{values['total']} "
                f"({values['accuracy_percent']:.2f}%) "
                f"evaluable={values['evaluable']} "
                f"quota={values['provider_quota_exhausted']}"
            )

    print()
    print(
        f"Checkpoint: "
        f"{OUTPUT_FILE.resolve()}"
    )


def main() -> None:
    """Run the BIRD Mini-Dev evaluation."""

    arguments = parse_arguments()

    questions = load_questions()

    if len(questions) != BENCHMARK_SIZE:
        raise ValueError(
            f"Expected exactly {BENCHMARK_SIZE} BIRD pilot "
            f"questions, but found {len(questions)}."
        )

    if (
        arguments.limit is not None
        and arguments.limit <= 0
    ):
        raise ValueError(
            "--limit must be greater than zero."
        )

    document = prepare_document(
        questions=questions,
        resume=arguments.resume,
    )

    results = document["results"]

    executor = SQLExecutor()

    total_start = time.perf_counter()

    questions_to_process = questions

    if arguments.limit is not None:
        questions_to_process = (
            questions[
                :arguments.limit
            ]
        )

    print("=" * 70)
    print(
        "TalkToTableau - "
        "BIRD Mini-Dev Evaluation"
    )
    print("=" * 70)

    print(
        f"Benchmark questions : {len(questions)}"
    )

    print(
        f"Questions this run  : "
        f"{len(questions_to_process)}"
    )

    print(
        f"Mode                : "
        f"{'RESUME' if arguments.resume else 'FRESH'}"
    )

    print(
        f"Checkpoint           : "
        f"{OUTPUT_FILE}"
    )

    print()

    result_index = build_result_index(
        results
    )

    for index, question in enumerate(
        questions_to_process,
        start=1,
    ):
        question_id = str(
            question["question_id"]
        )

        existing_record = result_index.get(
            question_id
        )

        if (
            arguments.resume
            and existing_record is not None
            and is_completed_record(
                existing_record
            )
        ):
            print(
                f"[{index:02d}/{len(questions_to_process)}] "
                f"Q{question_id} "
                f"already completed - SKIP"
            )
            continue

        print(
            f"[{index:02d}/{len(questions_to_process)}] "
            f"Q{question_id} | "
            f"{question['db_id']} | "
            f"{question['difficulty']}"
        )

        record = run_question(
            question=question,
            executor=executor,
        )

        result_index[question_id] = record

        results[:] = [
            result_index[
                str(question_item["question_id"])
            ]
            for question_item in questions
            if str(
                question_item["question_id"]
            )
            in result_index
        ]

        document["last_checkpoint_at"] = (
            utc_now()
        )

        document["summary"] = (
            calculate_summary(
                results
            )
        )

        save_checkpoint(
            document
        )

        print_question_status(
            index=index,
            total=len(
                questions_to_process
            ),
            record=record,
        )

        print(
            "       Checkpoint saved."
        )

    total_elapsed = (
        time.perf_counter()
        - total_start
    )

    document["total_elapsed_seconds"] = round(
        total_elapsed,
        3,
    )

    document["summary"] = (
        calculate_summary(
            results
        )
    )

    document["completed_at"] = utc_now()

    save_checkpoint(
        document
    )

    print_summary(
        document
    )


if __name__ == "__main__":
    main()