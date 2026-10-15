from pathlib import Path

import duckdb
import pandas as pd

from .base import DataSource


SUPPORTED_EXTENSIONS = {
    ".csv",
    ".xlsx",
    ".xlsm",
}


class FileDataSource(DataSource):
    """Load and query supported tabular files."""

    def __init__(
        self,
        file_path: str | Path,
        sheet_name: str | int = 0,
    ):
        self.file_path = Path(file_path)
        self.sheet_name = sheet_name
        self.dataframe: pd.DataFrame | None = None

    def test_connection(self) -> bool:
        """Check whether the file exists and is supported."""

        return (
            self.file_path.exists()
            and self.file_path.is_file()
            and self.file_path.suffix.lower()
            in SUPPORTED_EXTENSIONS
        )

    def load(self) -> pd.DataFrame:
        """Load the file into a Pandas DataFrame."""

        if not self.test_connection():
            raise ValueError(
                f"Unsupported or missing file: "
                f"{self.file_path}"
            )

        suffix = self.file_path.suffix.lower()

        if suffix == ".csv":
            try:
                # Most CSV files are UTF-8.
                dataframe = pd.read_csv(
                    self.file_path,
                    encoding="utf-8",
                )

            except UnicodeDecodeError:
                # Some CSV files exported from Excel/Windows use
                # Windows-1252 encoding instead of UTF-8.
                dataframe = pd.read_csv(
                    self.file_path,
                    encoding="cp1252",
                )

        else:
            dataframe = pd.read_excel(
                self.file_path,
                sheet_name=self.sheet_name,
            )

        if dataframe.empty:
            raise ValueError(
                "The input file contains no data rows."
            )

        self.dataframe = dataframe

        return dataframe

    def get_schema(self) -> dict[str, str]:
        """Return basic column/type metadata."""

        if self.dataframe is None:
            self.load()

        assert self.dataframe is not None

        return {
            column: str(dtype)
            for column, dtype
            in self.dataframe.dtypes.items()
        }

    def execute(
        self,
        query: str,
    ) -> pd.DataFrame:
        """
        Execute a read-only SQL query against the loaded file.

        Files are exposed to DuckDB as the logical table `data`.
        """

        if self.dataframe is None:
            self.load()

        assert self.dataframe is not None

        normalized_query = query.strip()

        if not normalized_query:
            raise ValueError(
                "SQL query is empty."
            )

        if not normalized_query.lower().startswith(
            "select"
        ):
            raise ValueError(
                "FileDataSource only permits "
                "SELECT queries."
            )

        connection = duckdb.connect()

        try:
            connection.register(
                "data",
                self.dataframe,
            )

            try:
                return connection.execute(
                    normalized_query
                ).fetchdf()

            except Exception as error:
                raise ValueError(
                    "File SQL execution failed: "
                    f"{error}"
                ) from error

        finally:
            connection.close()