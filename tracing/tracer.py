from uuid import uuid4

from .models import (
    TraceEvent,
    TraceEventType,
    TraceRecord,
)


class ApplicationTracer:
    """
    Application-level tracer for one TalkToTableau request.

    The tracer keeps structured events in memory for the current
    application trace. Persistent logging/storage can be added later
    without changing the trace model.
    """

    def __init__(
        self,
        trace_id: str | None = None,
    ) -> None:
        self.record = TraceRecord(
            trace_id=trace_id or self._generate_trace_id()
        )

    @staticmethod
    def _generate_trace_id() -> str:
        """Generate a unique trace identifier."""

        return str(uuid4())

    @property
    def trace_id(self) -> str:
        """Return the current trace ID."""

        return self.record.trace_id

    def add_event(
        self,
        event_type: TraceEventType,
        message: str,
        metadata: dict[str, object] | None = None,
    ) -> TraceEvent:
        """Add a structured event to the current trace."""

        event = TraceEvent(
            event_type=event_type,
            message=message,
            metadata=metadata or {},
        )

        self.record.events.append(event)

        return event

    def complete(self) -> TraceRecord:
        """Mark the trace as completed and return the record."""

        self.record.completed = True

        return self.record

    def get_record(self) -> TraceRecord:
        """Return the current trace record."""

        return self.record