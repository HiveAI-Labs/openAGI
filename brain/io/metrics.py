from __future__ import annotations

from brain.obs.metrics import REGISTRY

IO_LLM_REQUESTS = REGISTRY.counter(
    "io_llm_requests_total",
    "LLM requests issued by the IO query pipeline",
    labels=("outcome",),
)

IO_LLM_LATENCY = REGISTRY.histogram(
    "io_llm_latency_seconds",
    help="Latency for IO LLM requests in seconds",
    labels=(),
    buckets=(0.1, 0.25, 0.5, 1, 2, 3, 5, 8, 10),
)

IO_GATE_DECISIONS = REGISTRY.counter(
    "io_gate_accept_total",
    "Gate decisions for IO query responses",
    labels=("decision",),
)

IO_CITATION_MISSING = REGISTRY.counter(
    "io_citation_missing_total",
    "Count of responses missing required citations",
)

IO_PROBE_ISSUES = REGISTRY.counter(
    "io_probe_issues_total",
    "Probe issues surfaced during IO validation",
    labels=("kind", "severity"),
)

KB_RECORDS_TOTAL = REGISTRY.counter(
    "kb_records_total",
    "Accepted knowledge base records from IO queries",
)


def record_llm_request(outcome: str, latency: float | None) -> None:
    """Record a model request outcome and optional latency."""
    try:
        IO_LLM_REQUESTS.inc(outcome=outcome)
    except Exception:
        pass
    if latency is not None:
        try:
            IO_LLM_LATENCY.observe(float(latency))
        except Exception:
            pass


def record_gate(decision: str) -> None:
    """Record the final gate decision."""
    try:
        IO_GATE_DECISIONS.inc(decision=decision)
    except Exception:
        pass
    if decision == "accept":
        try:
            KB_RECORDS_TOTAL.inc()
        except Exception:
            pass


def record_missing_citation() -> None:
    try:
        IO_CITATION_MISSING.inc()
    except Exception:
        pass


def record_probe_issue(kind: str, severity: str) -> None:
    try:
        IO_PROBE_ISSUES.inc(kind=kind, severity=severity)
    except Exception:
        pass
