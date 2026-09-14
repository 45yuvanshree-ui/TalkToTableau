from __future__ import annotations

from pydantic import BaseModel, Field


class ColumnSchema(BaseModel):
    name: str
    data_type: str
    primary_key: bool = False
    nullable: bool | None = None
    description: str | None = None


class ForeignKeySchema(BaseModel):
    column_name: str
    referenced_table: str
    referenced_column: str | None = None


class TableSchema(BaseModel):
    name: str
    source_type: str
    columns: list[ColumnSchema] = Field(default_factory=list)
    foreign_keys: list[ForeignKeySchema] = Field(default_factory=list)
    description: str | None = None


class DataSchema(BaseModel):
    source_type: str
    tables: list[TableSchema] = Field(default_factory=list)