from pydantic import BaseModel, Field


class SQLRequest(BaseModel):
    """SQL request produced by the analytical workflow."""

    query: str
    max_rows: int = Field(default=1000, ge=1, le=10000)


class SQLValidationResult(BaseModel):
    """Result of SQL safety validation."""

    is_valid: bool
    normalized_query: str
    reason: str | None = None


class SQLResult(BaseModel):
    """Validated SQL execution result."""

    columns: list[str]
    rows: list[list[object]]
    row_count: int