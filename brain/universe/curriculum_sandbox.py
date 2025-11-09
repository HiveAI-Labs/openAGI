"""Deterministic curriculum sandbox runner with audit logging."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable, Mapping, Sequence, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

from brain.io.atomic import atomic_write_json, atomic_write_json_lines
from brain.universe.sim_manager import run_scripted_episode


def _read_json_lines_file(path: Path, *, limit: int | None = None) -> list[dict[str, Any]]:
    """Read JSON lines from *path* with optional tail limit."""
    if not path.exists() or not path.is_file():
        return []
    try:
        with path.open("r", encoding="utf-8") as handle:
            if limit is not None and limit > 0:
                buffer: deque[str] = deque(maxlen=limit)
                for line in handle:
                    stripped = line.strip()
                    if stripped:
                        buffer.append(stripped)
                lines = list(buffer)
            else:
                lines = [ln.strip() for ln in handle if ln.strip()]
    except OSError:
        return []

    records: list[dict[str, Any]] = []
    for entry in lines:
        try:
            payload = json.loads(entry)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            records.append(payload)
    return records

EpisodeRunner = Callable[..., Mapping[str, Any]]


def _stable_dump(payload: Mapping[str, Any]) -> str:
    """Return a deterministic JSON representation for hashing."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


logger = logging.getLogger(__name__)


@contextmanager
def _scoped_environ(overrides: Mapping[str, str | None]) -> Iterator[None]:
    """Temporarily set environment variables for the duration of a block."""
    saved: dict[str, str | None] = {}
    for key, value in overrides.items():
        saved[key] = os.environ.get(key)
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@dataclass(frozen=True)
class CurriculumScenario:
    """Scenario describing a deterministic simulator replay."""

    name: str
    mode: str = "curriculum"
    seed: int = 0
    description: str | None = None
    sim_env: str | None = None
    max_steps: int | None = None
    force_world_model: bool = False
    force_disable_world_model: bool = False

    def config(self, default_sim_env: str | None) -> dict[str, Any]:
        cfg = asdict(self)
        if cfg.get("sim_env") is None and default_sim_env is not None:
            cfg["sim_env"] = default_sim_env
        return cfg


class CurriculumSandbox:
    """Run curriculum scenarios with deterministic replay and audit logging."""

    def __init__(
        self,
        *,
        artifacts_dir: Path,
        suite: str,
        workspace: str = "curriculum_sandbox",
        default_sim_env: str | None = None,
        episode_runner: EpisodeRunner | None = None,
    ) -> None:
        self.artifacts_dir = Path(artifacts_dir)
        self.suite = suite
        self.workspace = workspace
        self.default_sim_env = default_sim_env
        self._run_episode: EpisodeRunner = episode_runner or run_scripted_episode
        self._base_dir = self.artifacts_dir / "sandbox" / suite
        self._base_dir.mkdir(parents=True, exist_ok=True)

    def run(self, scenarios: Sequence[CurriculumScenario]) -> dict[str, Any]:
        if not scenarios:
            raise ValueError("scenarios must not be empty")

        timestamp = datetime.now(UTC)
        ts_token = timestamp.strftime("%Y%m%dT%H%M%S%fZ")
        results: list[dict[str, Any]] = []
        audit_entries: list[dict[str, Any]] = []
        config = [scenario.config(self.default_sim_env) for scenario in scenarios]
        success_count = 0

        for scenario in scenarios:
            env_overrides: dict[str, str | None] = {}
            sim_env = scenario.sim_env or self.default_sim_env
            if sim_env is not None:
                env_overrides["BRAIN_SIM_ENV"] = sim_env
            if scenario.max_steps is not None:
                env_overrides["SIM_MAX_STEPS"] = str(int(scenario.max_steps))

            with _scoped_environ(env_overrides):
                result = self._run_episode(
                    workspace=self.workspace,
                    mode=scenario.mode,
                    seed=scenario.seed,
                    artifacts_dir=self.artifacts_dir,
                    force_disable_world_model=scenario.force_disable_world_model,
                    force_world_model=scenario.force_world_model,
                )

            sanitized = self._summarize_result(result)
            sanitized["name"] = scenario.name
            sanitized["mode"] = scenario.mode
            sanitized["seed"] = scenario.seed
            sanitized["sim_env"] = sim_env
            sanitized["force_world_model"] = scenario.force_world_model
            sanitized["force_disable_world_model"] = scenario.force_disable_world_model
            sanitized["max_steps"] = scenario.max_steps
            if scenario.description:
                sanitized["description"] = scenario.description

            if sanitized.get("success"):
                success_count += 1

            results.append(sanitized)
            audit_entries.append(
                {
                    "timestamp": timestamp.isoformat(),
                    "scenario": scenario.name,
                    "seed": scenario.seed,
                    "mode": scenario.mode,
                    "result": sanitized,
                },
            )

        success_rate = success_count / len(results)
        canonical_payload = {
            "config": config,
            "results": results,
        }
        determinism_token = hashlib.sha256(_stable_dump(canonical_payload).encode("utf-8")).hexdigest()

        summary = {
            "suite": self.suite,
            "workspace": self.workspace,
            "timestamp": timestamp.isoformat(),
            "config": config,
            "results": results,
            "success_rate": success_rate,
            "determinism_token": determinism_token,
        }

        summary_path = self._base_dir / f"run_{ts_token}.json"
        history_path = self._base_dir / "history.jsonl"
        audit_path = self._base_dir / "audit.jsonl"

        atomic_write_json(summary_path, summary)

        history_entries = self._load_json_lines(history_path)
        history_entries.append(
            {
                "timestamp": summary["timestamp"],
                "suite": self.suite,
                "workspace": self.workspace,
                "success_rate": success_rate,
                "run_path": str(summary_path.resolve()),
                "determinism_token": determinism_token,
            },
        )
        atomic_write_json_lines(history_path, history_entries)

        audit_history = self._load_json_lines(audit_path)
        audit_history.extend(audit_entries)
        atomic_write_json_lines(audit_path, audit_history)

        summary["summary_path"] = str(summary_path.resolve())
        summary["history_path"] = str(history_path.resolve())
        summary["audit_path"] = str(audit_path.resolve())
        try:
            updated_summary = load_curriculum_summary(self.artifacts_dir)
        except Exception:
            updated_summary = None
        if updated_summary is not None:
            notify_curriculum_update(updated_summary)
        return summary

    def _summarize_result(self, result: Mapping[str, Any]) -> dict[str, Any]:
        trace = result.get("policy_trace")
        if not isinstance(trace, list):
            trace = []
        world_model_steps = sum(1 for step in trace if isinstance(step, Mapping) and step.get("source") == "world_model")
        heuristic_steps = sum(1 for step in trace if isinstance(step, Mapping) and step.get("source") == "heuristic")
        planner_source = result.get("planner_source")
        summary: dict[str, Any] = {
            "success": bool(result.get("success")),
            "done": bool(result.get("done", False)),
            "reward": self._safe_float(result.get("reward")),
            "used_world_model": bool(result.get("used_world_model")),
            "planner_source": planner_source,
            "wm_planner_used": bool(result.get("wm_planner_used")),
            "policy_trace_steps": len(trace),
            "policy_trace_breakdown": {
                "world_model": world_model_steps,
                "heuristic": heuristic_steps,
            },
            "message": result.get("message"),
        }
        if "duration_ms" in result:
            summary["duration_ms"] = self._safe_float(result.get("duration_ms"))
        return summary

    def _load_json_lines(self, path: Path) -> list[dict[str, Any]]:
        return _read_json_lines_file(path)

    @staticmethod
    def _safe_float(value: Any) -> float | None:
        try:
            if value is None:
                return None
            return float(value)
        except (TypeError, ValueError):
            return None


@dataclass(frozen=True)
class CurriculumSuiteStats:
    """Aggregated curriculum sandbox statistics for a suite."""

    suite: str
    runs_total: int
    success_rate_latest: float | None
    success_rate_average: float | None
    latest_timestamp: str | None
    latest_workspace: str | None
    determinism_token: str | None
    recent_failures: int
    recent_failure_ratio: float | None
    recent_window: int
    audit_entries: int
    history_entries: int
    latest_success: bool | None
    success_progression_delta: float | None = None
    success_momentum: float | None = None
    success_streak: int = 0
    failure_streak: int = 0
    age_hours: float | None = None
    stale: bool = False


@dataclass(frozen=True)
class CurriculumSummary:
    """Overview of curriculum sandbox suites under an artifacts directory."""

    generated_at: str
    artifacts_dir: str
    suites: list[CurriculumSuiteStats]
    determinism_token: str | None = None
    stale_window_hours: float | None = None


def load_curriculum_summary(
    artifacts_dir: Path | str,
    *,
    max_history: int = 50,
    audit_window: int = 10,
) -> CurriculumSummary:
    """Load curriculum sandbox outcomes for gating and reporting.

    Args:
        artifacts_dir: Root artifacts directory containing ``sandbox/``.
        max_history: Number of history entries to include when averaging.
        audit_window: Tail window of audit entries considered "recent".

    Returns:
        CurriculumSummary with per-suite statistics. Missing directories yield
        an empty summary.

    """
    base_dir = Path(artifacts_dir)
    sandbox_root = base_dir / "sandbox"
    suites: list[CurriculumSuiteStats] = []
    generated_at_dt = datetime.now(UTC)
    generated_at = generated_at_dt.isoformat()
    canonical_entries: list[dict[str, Any]] = []

    stale_env = os.getenv("BRAIN_CURRICULUM_STALE_HOURS", "72")
    stale_window_hours: float | None
    try:
        stale_window_hours = float(stale_env)
        if stale_window_hours < 0:
            stale_window_hours = None
    except (TypeError, ValueError):
        stale_window_hours = None

    def _parse_timestamp(ts: str | None) -> datetime | None:
        if not ts:
            return None
        try:
            normalized = ts[:-1] + "+00:00" if ts.endswith("Z") else ts
            parsed = datetime.fromisoformat(normalized)
        except Exception:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)

    if sandbox_root.exists() and sandbox_root.is_dir():
        for suite_dir in sorted(p for p in sandbox_root.iterdir() if p.is_dir()):
            run_files = sorted(suite_dir.glob("run_*.json"))
            if not run_files and not (suite_dir / "history.jsonl").exists():
                continue

            latest_success_rate: float | None = None
            latest_timestamp: str | None = None
            latest_workspace: str | None = None
            determinism_token: str | None = None
            latest_success_flag: bool | None = None
            age_hours: float | None = None
            stale = False

            if run_files:
                latest_path = run_files[-1]
                try:
                    latest_payload = json.loads(latest_path.read_text(encoding="utf-8"))
                    latest_success_rate = CurriculumSandbox._safe_float(latest_payload.get("success_rate"))
                    latest_timestamp = latest_payload.get("timestamp")
                    latest_workspace = latest_payload.get("workspace")
                    determinism_token = latest_payload.get("determinism_token")
                    results = latest_payload.get("results") or []
                    if isinstance(results, list) and results:
                        flags = [bool((entry or {}).get("success")) for entry in results if isinstance(entry, dict)]
                        if flags and all(flags):
                            latest_success_flag = True
                        elif flags and not any(flags):
                            latest_success_flag = False
                except Exception:
                    latest_success_rate = None

            parsed_ts = _parse_timestamp(latest_timestamp)
            if parsed_ts is not None:
                age_hours = max(0.0, (generated_at_dt - parsed_ts).total_seconds() / 3600.0)

            if not run_files or parsed_ts is None or stale_window_hours is not None and age_hours is not None and age_hours > stale_window_hours:
                stale = True

            history_path = suite_dir / "history.jsonl"
            history_records_full = _read_json_lines_file(history_path)
            history_entries = len(history_records_full)
            if max_history > 0:
                history_records = history_records_full[-max_history:]
            else:
                history_records = history_records_full
            # success_values_raw may contain None entries; narrow to list[float]
            success_values_raw: list[float | None] = [
                CurriculumSandbox._safe_float(entry.get("success_rate"))
                for entry in history_records
                if isinstance(entry, dict)
            ]
            success_values: list[float] = [val for val in success_values_raw if val is not None]
            success_rate_average = (
                float(sum(success_values) / len(success_values)) if success_values else None
            )

            progression_delta: float | None = None
            if latest_success_rate is not None and success_rate_average is not None:
                progression_delta = float(latest_success_rate - success_rate_average)

            momentum: float | None = None
            if success_values:
                window = min(len(success_values) // 2, 5)
                if window >= 1 and len(success_values) >= window * 2:
                    recent_slice = success_values[-window:]
                    prior_slice = success_values[-2 * window : -window]
                    if prior_slice:
                        momentum = float(
                            (sum(recent_slice) / len(recent_slice))
                            - (sum(prior_slice) / len(prior_slice)),
                        )

            audit_path = suite_dir / "audit.jsonl"
            audit_records_full = _read_json_lines_file(audit_path)
            audit_entries = len(audit_records_full)
            if audit_window > 0:
                recent_audit = audit_records_full[-audit_window:]
            else:
                recent_audit = audit_records_full
            recent_failures = 0
            for entry in recent_audit:
                result = entry.get("result") if isinstance(entry, dict) else None
                if isinstance(result, dict) and not bool(result.get("success")):
                    recent_failures += 1
            recent_window = len(recent_audit)
            recent_failure_ratio = (
                float(recent_failures / recent_window) if recent_window else None
            )

            success_streak = 0
            failure_streak = 0
            for entry in reversed(audit_records_full):
                result = entry.get("result") if isinstance(entry, dict) else None
                if not isinstance(result, dict):
                    continue
                success_flag = bool(result.get("success"))
                if success_flag:
                    if failure_streak > 0:
                        break
                    success_streak += 1
                else:
                    if success_streak > 0:
                        break
                    failure_streak += 1

            suites.append(
                CurriculumSuiteStats(
                    suite=suite_dir.name,
                    runs_total=len(run_files),
                    success_rate_latest=latest_success_rate,
                    success_rate_average=success_rate_average,
                    latest_timestamp=latest_timestamp,
                    latest_workspace=latest_workspace,
                    determinism_token=determinism_token,
                    recent_failures=recent_failures,
                    recent_failure_ratio=recent_failure_ratio,
                    recent_window=recent_window,
                    audit_entries=audit_entries,
                    history_entries=history_entries,
                    latest_success=latest_success_flag,
                    success_progression_delta=progression_delta,
                    success_momentum=momentum,
                    success_streak=success_streak,
                    failure_streak=failure_streak,
                    age_hours=age_hours,
                    stale=stale,
                ),
            )

            canonical_entries.append(
                {
                    "suite": suite_dir.name,
                    "latest_timestamp": latest_timestamp,
                    "latest_success_rate": latest_success_rate,
                    "determinism_token": determinism_token,
                    "age_hours": age_hours,
                    "stale": stale,
                },
            )

    canonical_payload = json.dumps(
        canonical_entries,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    summary_token = hashlib.sha256(canonical_payload).hexdigest()

    return CurriculumSummary(
        generated_at=generated_at,
        artifacts_dir=str(base_dir.resolve()),
        suites=suites,
        determinism_token=summary_token,
        stale_window_hours=stale_window_hours,
    )


_CURRICULUM_OBSERVERS: list[Callable[[CurriculumSummary], None]] = []


def register_curriculum_observer(observer: Callable[[CurriculumSummary], None]) -> None:
    """Register a callback invoked when curriculum sandboxes emit updates."""
    if observer not in _CURRICULUM_OBSERVERS:
        _CURRICULUM_OBSERVERS.append(observer)
        logger.debug("registered curriculum observer %s (total=%d)", observer, len(_CURRICULUM_OBSERVERS))


def unregister_curriculum_observer(observer: Callable[[CurriculumSummary], None]) -> None:
    """Remove a previously registered curriculum sandbox observer."""
    try:
        _CURRICULUM_OBSERVERS.remove(observer)
    except ValueError:
        pass


def notify_curriculum_update(summary: CurriculumSummary) -> None:
    """Notify observers that curriculum sandbox results changed."""
    logger.debug(
        "notify_curriculum_update called: observers=%d determinism=%s",
        len(_CURRICULUM_OBSERVERS),
        getattr(summary, "determinism_token", None),
    )
    for observer in list(_CURRICULUM_OBSERVERS):
        try:
            logger.debug("calling curriculum observer %s", observer)
            observer(summary)
        except Exception:
            logger.exception("curriculum observer %s failed, unregistering", observer)
            try:
                _CURRICULUM_OBSERVERS.remove(observer)
            except ValueError:
                pass
            continue


__all__ = [
    "CurriculumScenario",
    "CurriculumSandbox",
    "CurriculumSuiteStats",
    "CurriculumSummary",
    "register_curriculum_observer",
    "unregister_curriculum_observer",
    "notify_curriculum_update",
    "load_curriculum_summary",
]
