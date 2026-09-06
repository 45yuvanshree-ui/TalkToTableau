from pathlib import Path

import pandas as pd


SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xlsm"}


class FileDataSource:
    """Load supported tabular files into a Pandas DataFrame."""

    def __init__(self, file_path: str | Path, sheet_name: str | int = 0):
        self.file_path = Path(file_path)
        self.sheet_name = sheet_name
        self.dataframe: pd.DataFrame | None = None

    def test_connection(self) -> bool:
        """Check whether the file exists and is a supported type."""
        return (
            self.file_path.exists()
            and self.file_path.is_file()
            and self.file_path.suffix.lower() in SUPPORTED_EXTENSIONS
        )

    def load(self) -> pd.DataFrame:
        """Load the file into a Pandas DataFrame."""
        if not self.test_connection():
            raise ValueError(
                f"Unsupported or missing file: {self.file_path}"
            )

        suffix = self.file_path.suffix.lower()

        if suffix == ".csv":
            dataframe = pd.read_csv(self.file_path)
        else:
            dataframe = pd.read_excel(
                self.file_path,
                sheet_name=self.sheet_name,
            )

        if dataframe.empty:
            raise ValueError("The input file contains no data rows.")

        self.dataframe = dataframe
        return dataframe

    def get_schema(self) -> dict[str, str]:
        """Return basic column/type metadata."""
        if self.dataframe is None:
            self.load()

        assert self.dataframe is not None

        return {
            column: str(dtype)
            for column, dtype in self.dataframe.dtypes.items()
        }