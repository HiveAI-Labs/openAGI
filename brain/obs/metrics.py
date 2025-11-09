from __future__ import annotations

from collections.abc import Iterable, Mapping
import os
import threading
from typing import Any, cast

__all__ = [
    "Counter",
    "Gauge",
    "Histogram",
    "MetricsRegistry",
    "REGISTRY",
    "metrics_text",
    "record_exec",
    "record_ok",
    "record_error",
    "latency_p95_ms",
    "error_rate",
    "metrics_snapshot",
    "learn_harvest_total",
    "learn_promote_total",
    "strategy_novelty_total",
    "brain_semantic_trace_total",
    "brain_semantic_failure_root_causes_total",
    "brain_semantic_capability_updates_total",
    "brain_wm_hypotheses_total",
    "brain_wm_llm_latency_ms",
    "brain_wm_validation_failures_total",
    "brain_wm_snapshots_total",
    "brain_wm_refinement_queue_size",
    "brain_wm_refinement_events_total",
    "brain_wm_retrain_queue_size",
    "brain_wm_retrain_events_total",
    "brain_wm_domain_accuracy",
    "brain_semantic_self_improvement_total",
    "brain_semantic_self_improvement_queue_size",
    "brain_plugin_host_outcomes_total",
    "brain_curriculum_suite_success_latest",
    "brain_curriculum_suite_success_average",
    "brain_curriculum_suite_progression_delta",
    "brain_curriculum_suite_momentum",
    "brain_curriculum_suite_failure_ratio",
    "brain_curriculum_suite_streak",
    "brain_strategy_curriculum_gate_weight",
    "brain_strategy_curriculum_gate_ok",
    "brain_strategy_curriculum_blockers_total",
    "brain_strategy_curriculum_progression",
    "brain_strategy_curriculum_momentum",
    "brain_curriculum_ack_totals",
    "brain_curriculum_ack_signature_checked",
    "brain_curriculum_ack_signature_valid_ratio",
    "brain_curriculum_ack_chaos_latency_ms",
    "brain_curriculum_ack_mttr_seconds",
    "brain_learning_budget_score",
    "brain_learning_budget_throttle_seconds",
    "brain_learning_budget_max_runs_per_hour",
    "brain_learning_budget_halt",
    "brain_learning_budget_latency_multiplier",
    "brain_learning_budget_cost_multiplier",
    "brain_learning_budget_failing_gates_total",
    "brain_learning_budget_updated_timestamp",
    "brain_wm_prompt_enrichment_total",
    "clear_curriculum_metrics",
    "record_learning_budget_state",
    "clear_learning_budget_metrics",
    "brain_depth_verification_total",
    "brain_chain_failure_total",
    "brain_soft_consensus_total",
    "brain_chain_score_ms",
    "brain_tool_reasoning_total",
    "brain_tool_reasoning_latency_ms",
    "brain_principle_refresh_due_total",
    "brain_causality_validation_total",
    "brain_consistency_goals_scheduled_total",
    "brain_consistency_ledger_updates_total",
    "brain_consistency_scheduler_events_total",
    "brain_consistency_scheduler_budget",
    "brain_consistency_contradiction_drift_pct",
    "brain_consistency_contradictions_prev",
    "brain_consistency_contradictions_last",
    "brain_consistency_drift_alerts_total",
    "brain_consistency_scheduler_latency_ms",
    "brain_goals_lifecycle_total",
    "brain_goals_completion_latency_ms",
    "brain_goals_open_total",
    "auto_expand_tasks_total",
    "auto_expand_added_total",
    "auto_expand_low_agreement_total",
    "auto_expand_filter_block_total",
    "auto_expand_latency_ms",
    "auto_expand_scheduled_runs_total",
    "auto_expand_scheduled_errors_total",
    "selector_consults_log_total",
    "selector_model_feedback_total",
    "teach_select_total",
    "teach_grade_total",
    "teach_lessons_generated_total",
    "teach_promotion_total",
    "teach_policy_training_runs_total",
    "teach_applied_total",
    "teach_canary_total",
    # Strategy selector / teach uplift observability
    "strategy_selector_weight_llm",
    "strategy_selector_weight_world_model",
    "strategy_selector_weight_policy",
    "strategy_selector_teach_last_apply_timestamp",
    # Navigation synthesis & selector adoption
    "navigation_planner_abstain_total",
    "selector_navigation_non_llm_total",
    "selector_ab_report_mode",
    # Fast-path vs selector decision metrics (navigation + rationale hardening)
    "nav_fastpath_total",
    "selector_decision_total",
    "selector_abstain_total",
    "selector_dedupe_dropped_total",
    # Planner invariant instrumentation
    "planner_confidence_clamped_total",
    "planner_early_abstain_total",
    "tool_attempts_histogram",
]


class Counter:
    def __init__(self, name: str, help: str = "", labels: tuple[str, ...] = ()) -> None:
        self.name = name
        self.help = help
        # store label names under a distinct attribute to avoid shadowing
        # the `labels()` method (callable) that returns a labelled cell.
        self._label_names = tuple(labels)
        self._lock = threading.Lock()
        self._values: dict[tuple[str, ...], float] = {}

    def labels(self, *label_values: str) -> _CounterCell:
        """Return a labelled counter cell for the provided label values.

        We keep the method name `labels()` for compatibility with callers
        (e.g. `metric.labels(a, b).inc()`), but store the label *names*
        in `_label_names` to avoid an instance attribute/method collision.
        """
        if not self._label_names:
            raise ValueError("metric has no labels")
        if len(label_values) != len(self._label_names):
            raise ValueError("label length mismatch")
        return _CounterCell(self, tuple(str(v) for v in label_values))

    def labels_values(self, *label_values: str) -> _CounterCell:
        """Compatibility alias used by existing tests."""
        if not self._label_names:
            raise ValueError("metric has no labels")
        if len(label_values) != len(self._label_names):
            raise ValueError("label length mismatch")
        return _CounterCell(self, tuple(str(v) for v in label_values))

    def inc(self, amount: float = 1.0, **label_kv: str) -> None:
        lv = self._normalize_kwargs(label_kv)
        with self._lock:
            self._values[lv] = self._values.get(lv, 0.0) + float(amount)

    def collect(self) -> dict[tuple[str, ...], float]:
        with self._lock:
            return dict(self._values)

    def _normalize_kwargs(self, label_kv: dict[str, str]) -> tuple[str, ...]:
        if self._label_names:
            if not label_kv:
                raise ValueError("label values required")
            return tuple(str(label_kv[name]) for name in self._label_names)
        # If the metric was created without labels, permissively ignore any provided
        # label key-values to avoid hard failures in callers that always pass kwargs.
        return ()

    @property
    def label_names(self) -> tuple[str, ...]:
        """Return the tuple of label names for this metric (read-only).

        Use `metric.labels(... )` to produce a labelled cell. External callers
        that previously inspected `metric.labels` for the name tuple should
        instead use `metric.label_names`.
        """
        return self._label_names


class _CounterCell:
    def __init__(self, parent: Counter, label_values: tuple[str, ...]) -> None:
        self._parent = parent
        self._lv = label_values

    def inc(self, amount: float = 1.0) -> None:
        with self._parent._lock:
            self._parent._values[self._lv] = self._parent._values.get(self._lv, 0.0) + float(amount)


class Gauge:
    def __init__(self, name: str, help: str = "", labels: tuple[str, ...] = ()) -> None:
        self.name = name
        self.help = help
        # Avoid colliding with the `labels()` method by keeping names here.
        self._label_names = tuple(labels)
        self._lock = threading.Lock()
        self._values: dict[tuple[str, ...], float] = {}

    def labels(self, *label_values: str) -> _GaugeCell:
        if not self._label_names:
            raise ValueError("metric has no labels")
        if len(label_values) != len(self._label_names):
            raise ValueError("label length mismatch")
        return _GaugeCell(self, tuple(str(v) for v in label_values))

    def labels_values(self, *label_values: str) -> _GaugeCell:
        """Compatibility alias used by existing tests."""
        if not self._label_names:
            raise ValueError("metric has no labels")
        if len(label_values) != len(self._label_names):
            raise ValueError("label length mismatch")
        return _GaugeCell(self, tuple(str(v) for v in label_values))

    def set(self, value: float, **label_kv: str) -> None:
        lv = self._normalize_kwargs(label_kv)
        _ensure_curriculum_namespace_for_gauge(self)
        with self._lock:
            self._values[lv] = float(value)

    def collect(self) -> dict[tuple[str, ...], float]:
        with self._lock:
            return dict(self._values)

    def _normalize_kwargs(self, label_kv: dict[str, str]) -> tuple[str, ...]:
        if self._label_names:
            if not label_kv:
                raise ValueError("label values required")
            return tuple(str(label_kv[name]) for name in self._label_names)
        # Metric created without labels: ignore any provided label kwargs for permissive behavior
        return ()

    @property
    def label_names(self) -> tuple[str, ...]:
        return self._label_names


class _GaugeCell:
    def __init__(self, parent: Gauge, label_values: tuple[str, ...]) -> None:
        self._parent = parent
        self._lv = label_values

    def set(self, value: float) -> None:
        _ensure_curriculum_namespace_for_gauge(self._parent)
        with self._parent._lock:
            self._parent._values[self._lv] = float(value)


_CURRICULUM_GAUGE_REGISTRY: set[Gauge] = set()
_CURRICULUM_METRIC_NAMESPACE: str | None = None


def _ensure_curriculum_namespace_for_gauge(gauge: Gauge) -> None:
    """Ensure curriculum metrics are scoped per artifacts namespace."""
    global _CURRICULUM_METRIC_NAMESPACE
    if gauge not in _CURRICULUM_GAUGE_REGISTRY:
        return

    current_ns = os.getenv("ARTIFACTS_DIR") or os.getenv("BRAIN_ARTIFACTS_DIR") or ""
    if _CURRICULUM_METRIC_NAMESPACE is None:
        _CURRICULUM_METRIC_NAMESPACE = current_ns
        return

    if current_ns != _CURRICULUM_METRIC_NAMESPACE:
        clear_curriculum_metrics()
        _CURRICULUM_METRIC_NAMESPACE = current_ns


class Histogram:
    def __init__(
        self,
        name: str,
        help: str = "",
        labels: tuple[str, ...] = (),
        buckets: Iterable[float] = (0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5),
    ) -> None:
        self.name = name
        self.help = help
        self._label_names = tuple(labels)
        self.buckets = tuple(sorted(float(b) for b in buckets))
        self._lock = threading.Lock()
        # Internal storage uses heterogeneous value types per key (sum: float, count: float, buckets: list[float])
        # Use `object` to avoid union typing errors in mypy while callers get a typed snapshot from `collect()`.
        self._values: dict[tuple[str, ...], dict[str, object]] = {}

    def labels(self, *label_values: str) -> _HistogramCell:
        if not self._label_names:
            raise ValueError("metric has no labels")
        if len(label_values) != len(self._label_names):
            raise ValueError("label length mismatch")
        return _HistogramCell(self, tuple(str(v) for v in label_values))

    def labels_values(self, *label_values: str) -> _HistogramCell:
        """Compatibility alias used by existing tests."""
        if not self._label_names:
            raise ValueError("metric has no labels")
        if len(label_values) != len(self._label_names):
            raise ValueError("label length mismatch")
        return _HistogramCell(self, tuple(str(v) for v in label_values))

    def observe(self, value: float, **label_kv: str) -> None:
        lv = self._normalize_kwargs(label_kv)
        v = float(value)
        with self._lock:
            state = self._values.setdefault(
                lv,
                {
                    "sum": 0.0,
                    "count": 0.0,
                    "buckets": [0.0 for _ in range(len(self.buckets) + 1)],
                },
            )
            # Use typed locals to avoid mypy complaints about heterogeneous internal storage
            sum_val = float(cast(Any, state.get("sum", 0.0)))
            count_val = float(cast(Any, state.get("count", 0.0)))
            buckets_list = list(cast(Any, state.get("buckets", [0.0 for _ in range(len(self.buckets) + 1)])))

            sum_val += v
            count_val += 1.0

            placed = False
            for idx, bound in enumerate(self.buckets):
                if v <= bound:
                    buckets_list[idx] += 1.0
                    placed = True
                    break
            if not placed:
                buckets_list[-1] += 1.0

            # persist back into the internal state
            state["sum"] = sum_val
            state["count"] = count_val
            state["buckets"] = buckets_list

    def collect(self) -> dict[tuple[str, ...], dict[str, float | list[float]]]:
        with self._lock:
            out: dict[tuple[str, ...], dict[str, float | list[float]]] = {}
            for lv, state in self._values.items():
                    # Create a typed snapshot for callers.
                    out[lv] = {
                        "sum": float(cast(Any, state.get("sum", 0.0))),
                        "count": float(cast(Any, state.get("count", 0.0))),
                        "buckets": list(cast(Any, state.get("buckets", []))),
                    }
            return out

    def _normalize_kwargs(self, label_kv: dict[str, str]) -> tuple[str, ...]:
        if self._label_names:
            if not label_kv:
                raise ValueError("label values required")
            return tuple(str(label_kv[name]) for name in self._label_names)
        # Metric created without labels: ignore any provided label kwargs for permissive behavior
        return ()

    @property
    def label_names(self) -> tuple[str, ...]:
        return self._label_names


class _HistogramCell:
    def __init__(self, parent: Histogram, label_values: tuple[str, ...]) -> None:
        self._parent = parent
        self._lv = label_values

    def observe(self, value: float) -> None:
        with self._parent._lock:
            state = self._parent._values.setdefault(
                self._lv,
                {
                    "sum": 0.0,
                    "count": 0.0,
                    "buckets": [0.0 for _ in range(len(self._parent.buckets) + 1)],
                },
            )
            # typed locals to avoid mypy errors when internal state is typed as object
            v = float(value)
            sum_val = float(cast(Any, state.get("sum", 0.0)))
            count_val = float(cast(Any, state.get("count", 0.0)))
            buckets_list = list(cast(Any, state.get("buckets", [0.0 for _ in range(len(self._parent.buckets) + 1)])))

            sum_val += v
            count_val += 1.0

            placed = False
            for idx, bound in enumerate(self._parent.buckets):
                if v <= bound:
                    buckets_list[idx] += 1.0
                    placed = True
                    break
            if not placed:
                buckets_list[-1] += 1.0

            state["sum"] = sum_val
            state["count"] = count_val
            state["buckets"] = buckets_list


class MetricsRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._metrics: dict[str, object] = {}

    def counter(self, name: str, help: str = "", labels: tuple[str, ...] = ()) -> Counter:
        with self._lock:
            metric = self._metrics.get(name)
            if metric is None:
                metric = Counter(name, help, labels)
                self._metrics[name] = metric
            return metric  # type: ignore[return-value]

    def gauge(self, name: str, help: str = "", labels: tuple[str, ...] = ()) -> Gauge:
        with self._lock:
            metric = self._metrics.get(name)
            if metric is None:
                metric = Gauge(name, help, labels)
                self._metrics[name] = metric
            return metric  # type: ignore[return-value]

    def histogram(
        self,
        name: str,
        help: str = "",
        labels: tuple[str, ...] = (),
        buckets: Iterable[float] = (0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5),
    ) -> Histogram:
        with self._lock:
            metric = self._metrics.get(name)
            if metric is None:
                metric = Histogram(name, help, labels, buckets)
                self._metrics[name] = metric
            return metric  # type: ignore[return-value]

    def all(self) -> dict[str, object]:
        with self._lock:
            return dict(self._metrics)


REGISTRY = MetricsRegistry()

brain_exec_total = REGISTRY.counter("brain_exec_total", "Total brain executions", ("route",))
brain_exec_errors_total = REGISTRY.counter("brain_exec_errors_total", "Total execution errors", ("route",))
brain_exec_latency_ms = REGISTRY.histogram(
    "brain_exec_latency_ms",
    "Execution latency (ms)",
    ("route",),
    buckets=(5, 10, 25, 50, 100, 200, 300, 500, 750, 1000, 2000, 5000),
)

# ---------------------------------------------------------------------------
# Planner invariants (P0) — counters and histograms
# ---------------------------------------------------------------------------
planner_confidence_clamped_total = REGISTRY.counter(
    "planner_confidence_clamped_total",
    "Count of times confidence values were clamped to [0,1]",
)
planner_early_abstain_total = REGISTRY.counter(
    "planner_early_abstain_total",
    "Count of early-stop abstentions during execution",
)
planner_steps_total = REGISTRY.counter(
    "planner_steps_total",
    "Total planner steps executed (post-materialization)",
)
tool_attempts_histogram = REGISTRY.histogram(
    "tool_attempts",
    "Distribution of attempts per tool execution",
    labels=("tool",),
    buckets=(1, 2, 3, 5, 8, 13),
)

brain_strategy_ensemble_consult_total = REGISTRY.counter(
    "brain_strategy_ensemble_consult_total",
    "Selector-initiated ensemble consult triggers",
    ("reason",),
)

brain_ensemble_broker_latency_ms = REGISTRY.histogram(
    "brain_ensemble_broker_latency_ms",
    "Brokered ensemble specialist latency (ms)",
    buckets=(50, 100, 250, 500, 1000, 2000, 4000, 8000, 12000),
)

# ---------------------------------------------------------------------------
# Autonomous expansion metrics
# ---------------------------------------------------------------------------
auto_expand_tasks_total = REGISTRY.counter(
    "auto_expand_tasks_total", "Tasks considered during autonomous expansion runs"
)
auto_expand_added_total = REGISTRY.counter(
    "auto_expand_added_total", "Notes added to memory during autonomous expansion"
)
auto_expand_low_agreement_total = REGISTRY.counter(
    "auto_expand_low_agreement_total", "Rejected notes due to low agreement"
)
auto_expand_filter_block_total = REGISTRY.counter(
    "auto_expand_filter_block_total", "Rejected notes by safety filter"
)
auto_expand_latency_ms = REGISTRY.histogram(
    "auto_expand_latency_ms", "Latency of autonomous expansion pass (ms)", buckets=(50,100,250,500,1000,2000,4000)
)
auto_expand_scheduled_runs_total = REGISTRY.counter(
    "auto_expand_scheduled_runs_total", "Number of autonomous expansion scheduler runs"
)
auto_expand_scheduled_errors_total = REGISTRY.counter(
    "auto_expand_scheduled_errors_total", "Scheduler run errors"
)

# ---------------------------------------------------------------------------
# Consensus / ensemble scoring metrics (Phase B rollout)
# ---------------------------------------------------------------------------
# Total consensus tool invocations, labelled by outcome ("consensus" when >=2
# verified answers survive cross-verification; otherwise "fallback").
consensus_requests_total = REGISTRY.counter(
    "consensus_requests_total",
    "Total LLM consensus broker invocations",
    ("outcome",),
)

# Histogram over average consensus ratio (mean verification_metadata.consensus
# across verified answers). Buckets chosen to give resolution in low-agreement
# regimes (<0.5) while retaining coverage up to 1.0.
consensus_agreement_ratio = REGISTRY.histogram(
    "consensus_agreement_ratio",
    "Distribution of average consensus agreement ratios",
    buckets=(0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0),
)

# Validator failures (syntax / hallucination / low factual score etc.) surfaced
# by CrossVerificationPipeline. Label is the failure reason category. This is
# incremented defensively by the consensus tool for transparency and future
# gating (proof bundle quality thresholds).
consensus_validator_fail_total = REGISTRY.counter(
    "consensus_validator_fail_total",
    "Total consensus answer validator failures",
    ("reason",),
)

# ---------------------------------------------------------------------------
# Selector consult logging & feedback (Phases 5-6)
# ---------------------------------------------------------------------------
# Logging status for consult persistence with optional HMAC signing.
selector_consults_log_total = REGISTRY.counter(
    "selector_consults_log_total",
    "Consult results persisted to artifacts with signing status",
    ("status",),
)

# Per-model feedback events recorded from consensus results (label=model, outcome)
selector_model_feedback_total = REGISTRY.counter(
    "selector_model_feedback_total",
    "Per-model feedback derived from consensus outcomes",
    ("model", "outcome"),
)

# ---------------------------------------------------------------------------
# Teaching loop (Phase A) metrics
# ---------------------------------------------------------------------------
# Lessons selected/generated: label difficulty (e.g. easy|medium|hard) and source
# (e.g. gap, expansion, curriculum). These feed adaptation and curriculum
# progression analytics.
teach_select_total = REGISTRY.counter(
    "teach_select_total",
    "Lessons generated/selected in teach cycle",
    ("difficulty", "source"),
)

# Grades produced for lesson responses. Label result (pass|fail|partial) and
# rubric (concise string identifying rubric variant). Downstream quality gates
# aggregate pass ratio and drift.
teach_grade_total = REGISTRY.counter(
    "teach_grade_total",
    "Grades emitted for lesson responses",
    ("result", "rubric"),
)

# ---------------------------------------------------------------------------
# Phase B – Teaching policy training & promotions
# ---------------------------------------------------------------------------
# Total lessons generated (post-kind resolution) labelled by kind + difficulty.
teach_lessons_generated_total = REGISTRY.counter(
    "teach_lessons_generated_total",
    "Lessons generated (Phase B extended) by kind and difficulty",
    ("kind", "difficulty"),
)

# Promotion events for lessons (manual curator or automated gating). Label result
# (promoted|rejected) to surface pass ratio drift and curation load.
teach_promotion_total = REGISTRY.counter(
    "teach_promotion_total",
    "Lesson promotion attempts",
    ("result",),
)

# Policy training runs (successful|failed) – increment once per attempted SFT
# dataset build + training invocation to track stability.
teach_policy_training_runs_total = REGISTRY.counter(
    "teach_policy_training_runs_total",
    "Policy training run attempts",
    ("status",),
)

# Selector application events for teach-derived adjustments
teach_applied_total = REGISTRY.counter(
    "teach_applied_total",
    "Selector routing adjustments applied from teaching uplift",
    ("strategy", "outcome"),
)

# Canary application outcomes for teach uplift
teach_canary_total = REGISTRY.counter(
    "teach_canary_total",
    "Teach uplift canary gate outcomes",
    ("result", "reason"),
)

rag_queries_total = REGISTRY.counter("rag_queries_total", "Total RAG queries")
rag_failed_total = REGISTRY.counter("rag_failed_total", "Total failed RAG queries")

learn_harvest_total = REGISTRY.counter(
    "learn_harvest_total",
    "Total /learn/harvest invocations",
    ("status",),
)
learn_promote_total = REGISTRY.counter(
    "learn_promote_total",
    "Total promotion attempts",
    ("status",),
)
strategy_novelty_total = REGISTRY.counter(
    "brain_strategy_novelty_total",
    "Strategy novelty events",
    ("novelty",),
)

brain_semantic_trace_total = REGISTRY.counter(
    "brain_semantic_trace_total",
    "Semantic trace emissions",
    ("strategy", "outcome"),
)
brain_semantic_failure_root_causes_total = REGISTRY.counter(
    "brain_semantic_failure_root_causes_total",
    "Semantic failure analyses by root cause",
    ("root_cause",),
)
brain_semantic_capability_updates_total = REGISTRY.counter(
    "brain_semantic_capability_updates_total",
    "Capability graph update outcomes",
    ("outcome",),
)

brain_wm_hypotheses_total = REGISTRY.counter(
    "brain_wm_hypotheses_total",
    "LLM-derived hypothesis lifecycle counts",
    ("status",),
)

brain_wm_llm_latency_ms = REGISTRY.histogram(
    "brain_wm_llm_latency_ms",
    "LLM teacher latency (ms)",
    buckets=(50, 100, 250, 500, 1000, 2000, 3500, 5000, 7000),
)

brain_wm_validation_failures_total = REGISTRY.counter(
    "brain_wm_validation_failures_total",
    "Validation failures during world model bootstrapping",
    ("reason",),
)

brain_wm_snapshots_total = REGISTRY.counter(
    "brain_wm_snapshots_total",
    "World model snapshots recorded",
    ("status",),
)

brain_wm_refinement_queue_size = REGISTRY.gauge(
    "brain_wm_refinement_queue_size",
    "Pending snapshots queued for refinement",
)

brain_wm_refinement_events_total = REGISTRY.counter(
    "brain_wm_refinement_events_total",
    "World model refinement events",
    ("event",),
)

brain_wm_retrain_queue_size = REGISTRY.gauge(
    "brain_wm_retrain_queue_size",
    "Pending ensemble retrain triggers",
)

brain_wm_retrain_events_total = REGISTRY.counter(
    "brain_wm_retrain_events_total",
    "World model retrain trigger events",
    ("event",),
)

brain_semantic_self_improvement_total = REGISTRY.counter(
    "brain_semantic_self_improvement_total",
    "Self-improvement actions queued from semantic self-model",
    ("action", "status"),
)

brain_semantic_self_improvement_queue_size = REGISTRY.gauge(
    "brain_semantic_self_improvement_queue_size",
    "Queue size for semantic self-improvement actions",
    ("queue",),
)

brain_plugin_host_outcomes_total = REGISTRY.counter(
    "brain_plugin_host_outcomes_total",
    "Plugin host load/execute outcomes",
    ("event", "outcome", "reason"),
)

brain_wm_curiosity_iterations_total = REGISTRY.counter(
    "brain_wm_curiosity_iterations_total",
    "Curiosity planning iterations",
    ("status",),
)

brain_wm_curiosity_tasks_total = REGISTRY.counter(
    "brain_wm_curiosity_tasks_total",
    "Curiosity tasks emitted to refinement queue",
    ("outcome",),
)

brain_wm_prompt_enrichment_total = REGISTRY.counter(
    "brain_wm_prompt_enrichment_total",
    "Selected prompt enrichment arm for WM bootstrap",
    ("arm",),
)

brain_consistency_goals_scheduled_total = REGISTRY.counter(
    "brain_consistency_goals_scheduled_total",
    "Consistency remediation goals scheduling outcomes",
    ("outcome",),
)

brain_consistency_ledger_updates_total = REGISTRY.counter(
    "brain_consistency_ledger_updates_total",
    "Consistency ledger update outcomes",
    ("outcome",),
)

brain_consistency_scheduler_events_total = REGISTRY.counter(
    "brain_consistency_scheduler_events_total",
    "Consistency scheduler events",
    ("outcome",),
)

brain_consistency_scheduler_budget = REGISTRY.gauge(
    "brain_consistency_scheduler_budget",
    "Remaining global scheduling budget after last run",
)

# Consistency drift gauges (populated by proof bundle when ledger stats are computed)
brain_consistency_contradiction_drift_pct = REGISTRY.gauge(
    "brain_consistency_contradiction_drift_pct",
    "Percent change in contradictions since previous ledger snapshot",
)
brain_consistency_contradictions_prev = REGISTRY.gauge(
    "brain_consistency_contradictions_prev",
    "Previous contradictions count from ledger stats",
)
brain_consistency_contradictions_last = REGISTRY.gauge(
    "brain_consistency_contradictions_last",
    "Latest contradictions count from ledger stats",
)

# Alerts related to consistency drift breaches
brain_consistency_drift_alerts_total = REGISTRY.counter(
    "brain_consistency_drift_alerts_total",
    "Total number of emitted alerts for contradiction drift breaches",
    ("reason",),
)

# Scheduler runtime
brain_consistency_scheduler_latency_ms = REGISTRY.histogram(
    "brain_consistency_scheduler_latency_ms",
    "Latency of schedule_consistency_goals execution (ms)",
    (),
    buckets=(1, 2, 5, 10, 20, 50, 100, 250, 500, 1000),
)

# Goal lifecycle observability
brain_goals_lifecycle_total = REGISTRY.counter(
    "brain_goals_lifecycle_total",
    "Goal lifecycle events",
    ("event", "status"),
)

brain_goals_completion_latency_ms = REGISTRY.histogram(
    "brain_goals_completion_latency_ms",
    "End-to-end goal lifecycle latency (ms)",
    (),
    buckets=(100, 250, 500, 1000, 2000, 5000, 10000, 20000, 40000),
)

brain_goals_open_total = REGISTRY.gauge(
    "brain_goals_open_total",
    "Open goals by current status",
    ("status",),
)

# Autonomous self-goals observability
brain_self_goals_generated_total = REGISTRY.counter(
    "brain_self_goals_generated_total",
    "Autonomous (self) goals generated",
    ("status",),
)
brain_self_goals_open_total = REGISTRY.gauge(
    "brain_self_goals_open_total",
    "Open autonomous goals by status",
    ("status",),
)

brain_self_goals_schedule_runs_total = REGISTRY.counter(
    "brain_self_goals_schedule_runs_total",
    "Background autonomous goal schedule runs",
    ("outcome",),
)

# Depth & chain quality metrics (added for AQG verification instrumentation)
brain_depth_verification_total = REGISTRY.counter(
    "brain_depth_verification_total",
    "Depth verification outcomes",
    ("depth","outcome"),
)
brain_chain_failure_total = REGISTRY.counter(
    "brain_chain_failure_total",
    "Reasoning chain failures by depth",
    ("depth",),
)
brain_soft_consensus_total = REGISTRY.counter(
    "brain_soft_consensus_total",
    "Soft keyword consensus outcomes",
    ("depth","outcome"),
)
brain_chain_score_ms = REGISTRY.histogram(
    "brain_chain_score_ms",
    "Reasoning chain score (unitless treated as ms for histogram bucketing)",
    ("depth",),
    buckets=(1,2,3,4,5,6,8,10,12,15),
)

# Tool reasoning metrics
brain_tool_reasoning_total = REGISTRY.counter(
    "brain_tool_reasoning_total",
    "Tool reasoning suggestion events",
    ("tool", "accepted"),
)
brain_tool_reasoning_latency_ms = REGISTRY.histogram(
    "brain_tool_reasoning_latency_ms",
    "Latency of tool reasoning selection (ms)",
    (),
    buckets=(0.1, 0.5, 1, 2, 5, 10),
)

brain_principle_refresh_due_total = REGISTRY.counter(
    "brain_principle_refresh_due_total",
    "Count of principles flagged for refresh by retention gate",
    ("domain",),
)

brain_causality_validation_total = REGISTRY.counter(
    "brain_causality_validation_total",
    "Physical causality principle validation outcomes",
    ("principle","outcome"),
)

brain_wm_domain_accuracy = REGISTRY.gauge(
    "brain_wm_domain_accuracy",
    "Rolling validation accuracy by domain",
    ("domain",),
)

brain_curriculum_suite_success_latest = REGISTRY.gauge(
    "brain_curriculum_suite_success_latest",
    "Latest success rate observed for a curriculum sandbox suite",
    ("suite", "strategy"),
)

brain_curriculum_suite_success_average = REGISTRY.gauge(
    "brain_curriculum_suite_success_average",
    "Average success rate across recent history for a curriculum sandbox suite",
    ("suite", "strategy"),
)

brain_curriculum_suite_progression_delta = REGISTRY.gauge(
    "brain_curriculum_suite_progression_delta",
    "Delta between latest and historical average success for a curriculum suite",
    ("suite", "strategy"),
)

brain_curriculum_suite_momentum = REGISTRY.gauge(
    "brain_curriculum_suite_momentum",
    "Momentum comparing recent and prior success windows for a curriculum suite",
    ("suite", "strategy"),
)

brain_curriculum_suite_failure_ratio = REGISTRY.gauge(
    "brain_curriculum_suite_failure_ratio",
    "Recent failure ratio for curriculum suite scenarios",
    ("suite", "strategy"),
)

brain_curriculum_suite_streak = REGISTRY.gauge(
    "brain_curriculum_suite_streak",
    "Current success/failure streak length for a curriculum suite",
    ("suite", "strategy", "type"),
)

brain_strategy_curriculum_gate_weight = REGISTRY.gauge(
    "brain_strategy_curriculum_gate_weight",
    "Curriculum-derived weight applied to strategy arbitration",
    ("strategy",),
)

brain_strategy_curriculum_gate_ok = REGISTRY.gauge(
    "brain_strategy_curriculum_gate_ok",
    "Indicator that curriculum gate allows the strategy (1 OK / 0 blocked)",
    ("strategy",),
)

brain_strategy_curriculum_blockers_total = REGISTRY.gauge(
    "brain_strategy_curriculum_blockers_total",
    "Count of active curriculum gate blockers per strategy",
    ("strategy",),
)

brain_strategy_curriculum_progression = REGISTRY.gauge(
    "brain_strategy_curriculum_progression",
    "Average curriculum progression delta aggregated per strategy",
    ("strategy",),
)

brain_strategy_curriculum_momentum = REGISTRY.gauge(
    "brain_strategy_curriculum_momentum",
    "Average curriculum momentum aggregated per strategy",
    ("strategy",),
)


_CURRICULUM_GAUGES = (
    brain_curriculum_suite_success_latest,
    brain_curriculum_suite_success_average,
    brain_curriculum_suite_progression_delta,
    brain_curriculum_suite_momentum,
    brain_curriculum_suite_failure_ratio,
    brain_curriculum_suite_streak,
    brain_strategy_curriculum_gate_weight,
    brain_strategy_curriculum_gate_ok,
    brain_strategy_curriculum_blockers_total,
    brain_strategy_curriculum_progression,
    brain_strategy_curriculum_momentum,
)

_CURRICULUM_GAUGE_REGISTRY.update(_CURRICULUM_GAUGES)


def clear_curriculum_metrics() -> None:
    """Remove previously recorded curriculum gauge samples."""
    for gauge in _CURRICULUM_GAUGES:
        try:
            gauge._values.clear()
        except AttributeError:
            # Fallback for alternative Gauge implementations that may not expose _values
            cleaned = getattr(gauge, "_values", None)
            if isinstance(cleaned, dict):
                cleaned.clear()

brain_curriculum_ack_totals = REGISTRY.gauge(
    "brain_curriculum_ack_totals",
    "Curriculum alert acknowledgement counts by state",
    ("state",),
)

brain_curriculum_ack_signature_checked = REGISTRY.gauge(
    "brain_curriculum_ack_signature_checked",
    "Flag indicating curriculum acknowledgement signatures were validated (1=yes,0=no)",
)

brain_curriculum_ack_signature_valid_ratio = REGISTRY.gauge(
    "brain_curriculum_ack_signature_valid_ratio",
    "Ratio of acknowledgements with valid signatures within the measurement window",
)

brain_curriculum_ack_chaos_latency_ms = REGISTRY.gauge(
    "brain_curriculum_ack_chaos_latency_ms",
    "Chaos drill acknowledgement signing duration (ms)",
)

brain_curriculum_ack_mttr_seconds = REGISTRY.gauge(
    "brain_curriculum_ack_mttr_seconds",
    "Curriculum acknowledgement mean time to resolve (seconds)",
    ("stat",),
)

brain_learning_budget_score = REGISTRY.gauge(
    "brain_learning_budget_score",
    "Adaptive learning budget score",
)

brain_learning_budget_throttle_seconds = REGISTRY.gauge(
    "brain_learning_budget_throttle_seconds",
    "Adaptive learning budget throttle duration (seconds)",
)

brain_learning_budget_max_runs_per_hour = REGISTRY.gauge(
    "brain_learning_budget_max_runs_per_hour",
    "Adaptive learning budget maximum runs per hour",
)

brain_learning_budget_halt = REGISTRY.gauge(
    "brain_learning_budget_halt",
    "Adaptive learning budget halt flag (1=halted)",
)

brain_learning_budget_latency_multiplier = REGISTRY.gauge(
    "brain_learning_budget_latency_multiplier",
    "Adaptive learning budget latency multiplier",
)

brain_learning_budget_cost_multiplier = REGISTRY.gauge(
    "brain_learning_budget_cost_multiplier",
    "Adaptive learning budget cost multiplier",
)

brain_learning_budget_failing_gates_total = REGISTRY.gauge(
    "brain_learning_budget_failing_gates_total",
    "Adaptive learning budget failing gate count",
)

brain_learning_budget_updated_timestamp = REGISTRY.gauge(
    "brain_learning_budget_updated_timestamp",
    "Adaptive learning budget last update timestamp (epoch seconds)",
)

# --- Strategy selector / teach uplift gauges ---
# Gauges are label-free; we export one value per process namespace.
strategy_selector_weight_llm = REGISTRY.gauge(
    "strategy_selector_weight_llm",
    "Current normalized weight for llm strategy (post-adaptation)",
)
strategy_selector_weight_world_model = REGISTRY.gauge(
    "strategy_selector_weight_world_model",
    "Current normalized weight for world_model strategy (post-adaptation)",
)
strategy_selector_weight_policy = REGISTRY.gauge(
    "strategy_selector_weight_policy",
    "Current normalized weight for policy strategy (post-adaptation)",
)
strategy_selector_teach_last_apply_timestamp = REGISTRY.gauge(
    "strategy_selector_teach_last_apply_timestamp",
    "Epoch timestamp of last successful teach uplift application",
)

# Navigation synthesis abstains (reason label for future taxonomy)
navigation_planner_abstain_total = REGISTRY.counter(
    "navigation_planner_abstain_total",
    "Navigation synthesis abstains",
    ("reason",),
)

# Non-LLM navigation executions (strategy label to differentiate policy vs world_model)
selector_navigation_non_llm_total = REGISTRY.counter(
    "selector_navigation_non_llm_total",
    "Navigation tasks executed with non-LLM strategy",
    ("strategy",),
)

# Strategy attempt/success counters and overall success rate gauge (Prometheus style)
selector_strategy_attempt_total = REGISTRY.counter(
    "selector_strategy_attempt_total",
    "Total strategy selection attempts (pre-outcome)",
    ("strategy", "task_type"),
)
selector_strategy_success_total = REGISTRY.counter(
    "selector_strategy_success_total",
    "Total successful strategy outcomes",
    ("strategy", "task_type"),
)
selector_overall_success_rate = REGISTRY.gauge(
    "selector_overall_success_rate",
    "Overall success rate across all strategies (successes/attempts)",
)

# AB report data source mode gauge (persisted | synthesized)
selector_ab_report_mode = REGISTRY.gauge(
    "selector_ab_report_mode",
    "AB report data source mode (persisted|synthesized)",
    ("mode",),
)

# ---------------------------------------------------------------------------
# Rationale / navigation fast-path metrics
# ---------------------------------------------------------------------------
# Fast-path navigation bypasses the planner; label reason for future taxonomy.
nav_fastpath_total = REGISTRY.counter(
    "nav_fastpath_total",
    "Navigation requests served via fast-path (planner bypass).",
    ("reason",),
)

# Selector decision events surfaced in rationale; label task_type+strategy.
selector_decision_total = REGISTRY.counter(
    "selector_decision_total",
    "Requests that emitted a selector_decision event.",
    ("task_type", "strategy"),
)

# Selector abstentions (not failures); label task_type only.
selector_abstain_total = REGISTRY.counter(
    "selector_abstain_total",
    "Selector abstentions recorded (not failures).",
    ("task_type",),
)

# Dedupe drop counter: normalized duplicate decision_id records ignored.
selector_dedupe_dropped_total = REGISTRY.counter(
    "selector_dedupe_dropped_total",
    "Records ignored due to duplicate decision_id.",
    ("reason",),
)

# --- AGI feature activation counters (rollout telemetry) ---
agi_tom_advisory_total = REGISTRY.counter(
    "agi_tom_advisory_total",
    "Theory-of-Mind advisory applications (navigation bias events)",
    ("action",),
)
agi_ontology_seed_total = REGISTRY.counter(
    "agi_ontology_seed_total",
    "Ontology seed attachments to task contexts",
    (),
)

agi_tom_snapshot_total = REGISTRY.counter(
    "agi_tom_snapshot_total",
    "Persisted Theory-of-Mind prediction snapshots",
    (),
)

# --- Belief store integration (P0) ---
brain_belief_ingest_total = REGISTRY.counter(
    "brain_belief_ingest_total",
    "Belief ingestion events from tool executions",
)

brain_belief_contradictions_total = REGISTRY.counter(
    "brain_belief_contradictions_total",
    "Contradictions detected when ingesting beliefs (same subject+relation, different object)",
)

def update_selector_overall_success_rate() -> None:
    """Recompute and set selector_overall_success_rate gauge.

    Iterates over attempt and success counters' internal value maps. Safe/no-op
    if internal structures not present. This avoids needing a full metrics
    snapshot parse on every outcome.
    """
    try:
        attempts = 0.0
        successes = 0.0
        # Access internal counter storage (bounded per-strategy labels)
        att_store = getattr(selector_strategy_attempt_total, "_values", {})
        suc_store = getattr(selector_strategy_success_total, "_values", {})
        if isinstance(att_store, dict):
            attempts = sum(float(v) for v in att_store.values())
        if isinstance(suc_store, dict):
            successes = sum(float(v) for v in suc_store.values())
        rate = (successes / attempts) if attempts > 0 else 0.0
        selector_overall_success_rate.set(float(rate))
    except Exception:
        # Defensive: never raise from metrics recompute
        pass

_LEARNING_BUDGET_GAUGES = (
    brain_learning_budget_score,
    brain_learning_budget_throttle_seconds,
    brain_learning_budget_max_runs_per_hour,
    brain_learning_budget_halt,
    brain_learning_budget_latency_multiplier,
    brain_learning_budget_cost_multiplier,
    brain_learning_budget_failing_gates_total,
    brain_learning_budget_updated_timestamp,
)

_LEARNING_BUDGET_NAMESPACE: str | None = None


def _ensure_learning_budget_namespace() -> None:
    """Ensure learning budget metrics reset when artifacts namespace changes."""
    global _LEARNING_BUDGET_NAMESPACE
    current_ns = os.getenv("ARTIFACTS_DIR") or os.getenv("BRAIN_ARTIFACTS_DIR") or ""
    if _LEARNING_BUDGET_NAMESPACE is None:
        _LEARNING_BUDGET_NAMESPACE = current_ns
        return
    if current_ns != _LEARNING_BUDGET_NAMESPACE:
        clear_learning_budget_metrics()
        _LEARNING_BUDGET_NAMESPACE = current_ns


def clear_learning_budget_metrics() -> None:
    """Remove previously recorded learning budget gauge samples."""
    for gauge in _LEARNING_BUDGET_GAUGES:
        cleaned = getattr(gauge, "_values", None)
        if isinstance(cleaned, dict):
            cleaned.clear()


def record_learning_budget_state(state: Mapping[str, Any] | None) -> None:
    """Record adaptive learning budget metrics."""
    _ensure_learning_budget_namespace()
    if not isinstance(state, Mapping):
        clear_learning_budget_metrics()
        return

    def _float(field: str, default: float = 0.0) -> float:
        try:
            value = state.get(field, default)
            return float(value)
        except Exception:
            return float(default)

    def _count(field: str) -> float:
        value = state.get(field, [])
        if isinstance(value, (list, tuple, set)):
            return float(len(value))
        if value is None:
            return 0.0
        try:
            return float(len(list(value)))
        except Exception:
            return 0.0

    brain_learning_budget_score.set(_float("score", 0.0))
    brain_learning_budget_throttle_seconds.set(_float("throttle_seconds", 0.0))
    brain_learning_budget_max_runs_per_hour.set(_float("max_runs_per_hour", 0.0))
    brain_learning_budget_halt.set(1.0 if state.get("halt_training") else 0.0)
    brain_learning_budget_latency_multiplier.set(_float("latency_multiplier", 0.0))
    brain_learning_budget_cost_multiplier.set(_float("cost_multiplier", 0.0))
    brain_learning_budget_failing_gates_total.set(_count("failing_gates"))
    brain_learning_budget_updated_timestamp.set(_float("updated_at", 0.0))

brain_wm_bootstrap_runs_total = REGISTRY.counter(
    "brain_wm_bootstrap_runs_total",
    "Navigation bootstrap orchestrations",
    ("outcome",),
)

brain_wm_bootstrap_latency_ms = REGISTRY.histogram(
    "brain_wm_bootstrap_latency_ms",
    "Navigation bootstrap orchestration latency (ms)",
    buckets=(50, 100, 250, 500, 1000, 2000, 3500, 5000, 7500, 10000),
)

_latency_samples: list[float] = []
_latency_lock = threading.Lock()
_error_total = 0
_ok_total = 0


def record_exec(route: str, ok: bool, latency_ms: float) -> None:
    try:
        brain_exec_total.labels(route).inc()
        if not ok:
            brain_exec_errors_total.labels(route).inc()
        brain_exec_latency_ms.labels(route).observe(float(latency_ms))
        with _latency_lock:
            _latency_samples.append(float(latency_ms))
            if len(_latency_samples) > 5000:
                del _latency_samples[: len(_latency_samples) - 5000]
    except Exception:
        pass


def latency_p95_ms() -> float:
    with _latency_lock:
        if not _latency_samples:
            return 0.0
        ordered = sorted(_latency_samples)
        idx = int(0.95 * (len(ordered) - 1))
        idx = max(0, min(idx, len(ordered) - 1))
        return float(ordered[idx])


def record_error() -> None:
    global _error_total
    try:
        _error_total += 1
    except Exception:
        pass


def record_ok() -> None:
    global _ok_total
    try:
        _ok_total += 1
    except Exception:
        pass


def error_rate() -> float:
    try:
        total = _ok_total + _error_total
        if total <= 0:
            return 0.0
        return float(_error_total) / float(total)
    except Exception:
        return 0.0


def metrics_text(reg: MetricsRegistry | None = None) -> str:
    reg = reg or REGISTRY
    lines: list[str] = []
    for name, metric in reg.all().items():
        if isinstance(metric, Counter):
            lines.append(f"# HELP {name} {metric.help}")
            lines.append(f"# TYPE {name} counter")
            collected = metric.collect()
            # Emit an aggregate (unlabeled) total first for stable scraping semantics in tests
            try:
                if metric.label_names and collected:
                    total_val = 0.0
                    for _labels, _value in collected.items():
                        try:
                            total_val += float(_value)
                        except Exception:
                            pass
                    # Cast to int when value is whole to keep output tidy
                    if abs(total_val - round(total_val)) < 1e-9:
                        lines.append(f"{name} {int(round(total_val))}")
                    else:
                        lines.append(f"{name} {total_val}")
            except Exception:
                pass
            for labels, value in collected.items():
                label_text = _format_labels(metric.label_names, labels)
                lines.append(f"{name}{label_text} {value}")
        elif isinstance(metric, Gauge):
            lines.append(f"# HELP {name} {metric.help}")
            lines.append(f"# TYPE {name} gauge")
            collected = metric.collect()
            try:
                if metric.label_names and collected:
                    total_val = 0.0
                    for _labels, _value in collected.items():
                        try:
                            total_val += float(_value)
                        except Exception:
                            pass
                    if abs(total_val - round(total_val)) < 1e-9:
                        lines.append(f"{name} {int(round(total_val))}")
                    else:
                        lines.append(f"{name} {total_val}")
            except Exception:
                pass
            for labels, value in collected.items():
                label_text = _format_labels(metric.label_names, labels)
                lines.append(f"{name}{label_text} {value}")
        elif isinstance(metric, Histogram):
            lines.append(f"# HELP {name} {metric.help}")
            lines.append(f"# TYPE {name} histogram")
            collected = metric.collect()
            for labels, state in collected.items():
                # state values come from `collect()` and may be typed as unions; cast to concrete types for processing
                buckets_vals = list(cast(Any, state.get("buckets", [])))
                cumulative = 0.0
                for idx, bound in enumerate(metric.buckets):
                    cumulative += float(buckets_vals[idx])
                    bucket_labels = _format_labels(metric.label_names + ("le",), labels + (str(bound),))
                    lines.append(f"{name}_bucket{bucket_labels} {cumulative}")
                cumulative += float(buckets_vals[-1])
                bucket_labels = _format_labels(metric.label_names + ("le",), labels + ("+Inf",))
                lines.append(f"{name}_bucket{bucket_labels} {cumulative}")
                base_labels = _format_labels(metric.label_names, labels)
                lines.append(f"{name}_sum{base_labels} {float(cast(Any, state.get('sum',0.0)))}")
                lines.append(f"{name}_count{base_labels} {float(cast(Any, state.get('count',0.0)))}")
    return "\n".join(lines) + ("\n" if lines else "")


def _format_labels(names: tuple[str, ...], values: tuple[str, ...]) -> str:
    if not names:
        return ""
    pairs = [f'{k}="{v}"' for k, v in zip(names, values, strict=False)]
    return "{" + ",".join(pairs) + "}"


def metrics_snapshot(reg: MetricsRegistry | None = None) -> dict[str, object]:
    """Return a structured snapshot of the metric registry."""
    reg = reg or REGISTRY
    counters: dict[str, list[dict[str, object]]] = {}
    gauges: dict[str, list[dict[str, object]]] = {}
    histograms: dict[str, list[dict[str, object]]] = {}
    for name, metric in reg.all().items():
        if isinstance(metric, Counter):
            samples: list[dict[str, object]] = []
            for labels, value in metric.collect().items():
                samples.append({
                    "labels": {label: val for label, val in zip(metric.label_names, labels, strict=False)},
                    "value": float(value),
                })
            counters[name] = samples
        elif isinstance(metric, Gauge):
            samples = []
            for labels, value in metric.collect().items():
                samples.append({
                    "labels": {label: val for label, val in zip(metric.label_names, labels, strict=False)},
                    "value": float(value),
                })
            gauges[name] = samples
        elif isinstance(metric, Histogram):
            samples = []
            collected = metric.collect()
            for labels, state in collected.items():
                samples.append({
                    "labels": {label: val for label, val in zip(metric.label_names, labels, strict=False)},
                    "sum": float(cast(Any, state.get("sum", 0.0))),
                    "count": float(cast(Any, state.get("count", 0.0))),
                    "bucket_bounds": list(metric.buckets),
                    "bucket_values": list(cast(Any, state.get("buckets", []))),
                })
            histograms[name] = samples
    return {
        "counters": counters,
        "gauges": gauges,
        "histograms": histograms,
    }
