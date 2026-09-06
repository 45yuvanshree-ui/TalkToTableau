from pydantic import BaseModel, Field


class ColumnSchema(BaseModel):
    """Normalized metadata for a single column."""

    name: str
    data_type: str


class TableSchema(BaseModel):
    """Normalized metadata for a logical table."""

    name: str
    source_type: str
    columns: list[ColumnSchema] = Field(default_factory=list)


class DataSchema(BaseModel):
    """Normalized schema exposed to the analytics workflow."""

    source_type: str
    tables: list[TableSchema] = Field(default_factory=list)