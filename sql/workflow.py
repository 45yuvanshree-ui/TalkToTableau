from pathlib import Path
from typing import Callable

from .executor import SQLExecutor
from .models import SQLResult


MAX_SQL_RETRIES = 2


class SQLWorkflow:
    """
    Controlled SQL generation/validation/execution workflow.

    SQL generation is injected as a callable so the LLM provider
    can be connected later without changing the execution layer.
    """

    def __init__(
        self,
        sql_generator: Callable[[str], str],
        max_retries: int = MAX_SQL_RETRIES,
    ):
        self.sql_generator = sql_generator
        self.max_retries = max_retries
        self.executor = SQLExecutor()

    def run_file(
        self,
        file_path: str | Path,
        user_request: str,
    ) -> SQLResult:
        """
        Generate and execute SQL against a file source.

        Retries are bounded.
        """

        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                query = self.sql_generator(user_request)

                return self.executor.execute_file(
                    file_path=file_path,
                    query=query,
                )

            except (ValueError, RuntimeError) as error:
                last_error = error

                if attempt >= self.max_retries:
                    break

        raise RuntimeError(
            f"SQL workflow failed after "
            f"{self.max_retries + 1} attempt(s)."
        ) from last_error