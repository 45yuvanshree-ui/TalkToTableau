from tracing import (
    ApplicationTracer,
    TraceEventType,
)


def test_tracer_generates_unique_trace_id():
    first = ApplicationTracer()
    second = ApplicationTracer()

    assert first.trace_id
    assert second.trace_id
    assert first.trace_id != second.trace_id


def test_tracer_can_add_event():
    tracer = ApplicationTracer(
        trace_id="test-trace-001"
    )

    event = tracer.add_event(
        event_type=TraceEventType.REQUEST,
        message="Request received.",
        metadata={
            "source": "test",
        },
    )

    assert event.event_type == TraceEventType.REQUEST
    assert event.message == "Request received."
    assert event.metadata["source"] == "test"

    record = tracer.get_record()

    assert record.trace_id == "test-trace-001"
    assert len(record.events) == 1
    assert record.completed is False


def test_tracer_can_record_multiple_workflow_events():
    tracer = ApplicationTracer(
        trace_id="test-trace-002"
    )

    tracer.add_event(
        event_type=TraceEventType.INPUT_GUARDRAIL,
        message="Input accepted.",
    )

    tracer.add_event(
        event_type=TraceEventType.ROUTING,
        message="Balanced model selected.",
        metadata={
            "tier": "balanced",
        },
    )

    tracer.add_event(
        event_type=TraceEventType.SQL,
        message="SQL executed successfully.",
        metadata={
            "row_count": 10,
        },
    )

    record = tracer.get_record()

    assert len(record.events) == 3
    assert record.events[0].event_type == (
        TraceEventType.INPUT_GUARDRAIL
    )
    assert record.events[1].event_type == (
        TraceEventType.ROUTING
    )
    assert record.events[2].event_type == (
        TraceEventType.SQL
    )


def test_tracer_can_complete():
    tracer = ApplicationTracer(
        trace_id="test-trace-003"
    )

    assert tracer.get_record().completed is False

    record = tracer.complete()

    assert record.completed is True
    assert record.trace_id == "test-trace-003"