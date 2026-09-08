from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


class TraceEventType(str, Enum):
    """Types of events recorded during an analytical request."""

    REQUEST = "request"
    INPUT_GUARDRAIL = "input_guardrail"
    ROUTING = "routing"
    AGENT = "agent"
    SCHEMA = "schema"
    SQL = "sql"
    ANALYSIS = "analysis"
    EVIDENCE = "evidence"
    VISUALIZATION = "visualization"
    STORY = "story"
    OUTPUT = "output"
    ERROR = "error"


class TraceEvent(BaseModel):
    """A single event recorded in an application trace."""

    event_type: TraceEventType

    message: str

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    metadata: dict[str, object] = Field(
        default_factory=dict
    )


class TraceRecord(BaseModel):
    """Complete trace for one analytical request."""

    trace_id: str

    started_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    events: list[TraceEvent] = Field(
        default_factory=list
    )

    completed: bool = False