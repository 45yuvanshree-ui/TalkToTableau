from pathlib import Path

from .base import DataSource
from .file_source import FileDataSource
from .sqlite_source import SQLiteDataSource


SQLITE_EXTENSIONS = {
    ".sqlite",
    ".sqlite3",
    ".db",
}


def create_data_source(
    source: str | Path,
) -> DataSource:
    """
    Create the appropriate DataSource implementation.

    Supported automatic sources:
    - CSV
    - XLSX
    - XLSM
    - SQLite
    """

    source_path = Path(source)

    if (
        source_path.suffix.lower()
        in SQLITE_EXTENSIONS
    ):
        return SQLiteDataSource(
            source_path
        )

    return FileDataSource(
        source_path
    )