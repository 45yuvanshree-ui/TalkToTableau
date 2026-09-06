from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class DataSource(ABC):
    """Common interface for supported TalkToTableau data sources."""

    @abstractmethod
    def test_connection(self) -> bool:
        """Return True when the data source is accessible."""
        raise NotImplementedError

    @abstractmethod
    def get_schema(self) -> Any:
        """Return metadata describing the available data."""
        raise NotImplementedError

    @abstractmethod
    def execute(self, query: str) -> pd.DataFrame:
        """Execute a permitted query and return tabular results."""
        raise NotImplementedError