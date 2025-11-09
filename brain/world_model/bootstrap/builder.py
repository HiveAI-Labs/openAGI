"""Sandboxed LLM teacher that proposes navigation hypotheses."""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
import json
import logging
import time
from typing import Any
from urllib.parse import urlparse

from pydantic import ValidationError

from brain.core.model_client import LLMError, ModelClient
from brain.knowledge.filters import LLMAnswerSafetyFilter
from brain.obs.metrics import brain_wm_llm_latency_ms

from .prompt_composer import append_curiosity_tasks, append_gap_summary
from .schemas import ActionHypothesis, DomainSnapshot, HypothesisBatch

try:
    from brain.arm_bandit import select as _arm_select  # reuse existing bandit
except Exception:  # pragma: no cover
    _arm_select = None  # type: ignore

_LOGGER = logging.getLogger(__name__)


class BootstrapError(RuntimeError):
    """Base class for bootstrap-related failures."""


class BootstrapTimeoutError(BootstrapError):
    """Raised when the LLM teacher exceeds the configured timeout."""


class BootstrapSafetyError(BootstrapError):
    """Raised when a response fails safety validation."""


class BootstrapParseError(BootstrapError):
    """Raised when the LLM response cannot be parsed into hypotheses."""


class LLMWorldModelBuilder:
    """Produce action hypotheses using a sandboxed LLM teacher."""

    _SYSTEM_PROMPT = (
        "You are an alignment-critical planning assistant. "
        "Output only JSON that matches the requested schema."
    )

    def __init__(
        self,
        model_client: ModelClient,
        *,
        timeout_s: float = 10.0,
        max_prompt_chars: int = 4096,
        max_response_chars: int = 8192,
        allowed_hosts: Sequence[str] | None = None,
        safety_filter: LLMAnswerSafetyFilter | None = None,
        extra_providers: Sequence[Any] | None = None,
    ) -> None:
        self._model = model_client
        self._timeout_s = float(timeout_s)
        self._max_prompt_chars = int(max_prompt_chars)
        self._max_response_chars = int(max_response_chars)
        self._allowed_hosts = tuple(allowed_hosts or ("localhost", "127.0.0.1"))
        self._filter = safety_filter or LLMAnswerSafetyFilter(procedural_only=True)
        self._validate_client_host()
        self._extra_providers = list(extra_providers or [])

    def propose_navigation_rules(self, snapshot: DomainSnapshot) -> HypothesisBatch:
        """Request new navigation rules for the provided domain snapshot."""
        prompt = self._build_prompt(snapshot)
        if len(prompt) > self._max_prompt_chars:
            raise BootstrapSafetyError("prompt exceeds configured size limit")

        response = self._call_llm(prompt)
        if len(response) > self._max_response_chars:
            raise BootstrapSafetyError("response exceeds configured size limit")

        hypotheses = self._parse_hypotheses(response)
        self._enforce_safety(prompt, hypotheses)

        extra = list(self._collect_extra_hypotheses(snapshot))
        if extra:
            dedup: dict[str, ActionHypothesis] = {item.fingerprint(): item for item in hypotheses}
            for candidate in extra:
                dedup.setdefault(candidate.fingerprint(), candidate)
            hypotheses = list(dedup.values())

        features = getattr(self, "_prompt_features", None)
        metadata = {
            "domain_fingerprint": snapshot.fingerprint(),
            "hypothesis_count": len(hypotheses),
        }
        if isinstance(features, dict):
            metadata["prompt_features"] = dict(features)
            if "template_id" in features:
                metadata["prompt_template_id"] = str(features["template_id"])  # duplicate for convenience
        if extra:
            ensemble_hashes = sorted({candidate.fingerprint() for candidate in extra})
            metadata["ensemble_hypotheses"] = len(ensemble_hashes)
            metadata["ensemble_hashes"] = ensemble_hashes
        return HypothesisBatch.build(
            prompt=prompt,
            response=response,
            hypotheses=hypotheses,
            metadata=metadata,
        )

    def _validate_client_host(self) -> None:
        candidate_attrs = ("base_url", "api_url")
        url_value = None
        for attr in candidate_attrs:
            url_value = getattr(self._model, attr, None)
            if url_value:
                break
        if not url_value:
            return  # Nothing to validate; treat client as in-process stub.
        try:
            parsed = urlparse(str(url_value))
        except Exception as exc:  # pragma: no cover - defensive branch
            raise BootstrapSafetyError(f"unable to parse model client URL: {exc}")
        host = (parsed.hostname or "").lower()
        if host not in self._allowed_hosts:
            raise BootstrapSafetyError(f"model client host '{host}' not in allowlist {self._allowed_hosts}")

    def _build_prompt(self, snapshot: DomainSnapshot) -> str:
        snapshot_json = json.dumps(snapshot.model_dump(), sort_keys=True, ensure_ascii=True, indent=2)
        schema_description = (
            "Return JSON with the shape {\n"
            '  "actions": [\n'
            "    {\n"
            '      "action": str,\n'
            '      "parameters": {str: str},\n'
            '      "preconditions": [str],\n'
            '      "effects": [str],\n'
            '      "confidence": float in [0,1],\n'
            '      "validation_steps": [str],\n'
            '      "test_cases": [\n'
            "        {\n"
            '          "description": str,\n'
            '          "initial_state": {str: Any},\n'
            '          "expected_outcome": {str: Any},\n'
            '          "max_steps": int,\n'
            '          "seed": int,\n'
            '          "timeout_s": float\n'
            "        }\n"
            "      ]\n"
            "    }\n"
            "  ]\n"
            "}\n"
            "Ensure validation_steps describe concrete numbered procedures."
        )
        instructions = (
            "You are teaching a navigation agent."
            " Provide action rules grounded in the provided observed transitions."
        )
        base = (
            f"{instructions}\n"
            f"Schema: {schema_description}\n"
            f"Domain snapshot:\n{snapshot_json}"
        )

        # Optional: enrich prompt with recent curiosity/gap summaries
        try:
            use_gaps = str(os.getenv("WM_PROMPT_USE_GAPS", "0")).strip().lower() in {"1", "true", "yes", "on"}
            gap_n_env = os.getenv("WM_PROMPT_GAP_N")
            gap_n = int(gap_n_env) if gap_n_env is not None else 3
        except Exception:
            use_gaps = False
            gap_n = 3
        # Optional bandit to choose which enrichments to apply (reuse brain.arm_bandit)
        try:
            use_bandit = str(os.getenv("WM_PROMPT_USE_BANDIT", "0")).strip().lower() in {"1", "true", "yes", "on"}
        except Exception:
            use_bandit = False
        template_id = "base"
        if use_bandit and _arm_select is not None:
            try:
                # Select among arms: base, gaps, tasks, gaps_tasks
                arm = _arm_select("wm_prompt_enrichment")
                template_id = str(arm or "base")
                if template_id == "gaps":
                    use_gaps, use_tasks = True, False
                elif template_id == "tasks":
                    use_gaps, use_tasks = False, True
                elif template_id == "gaps_tasks":
                    use_gaps, use_tasks = True, True
                else:
                    use_gaps, use_tasks = False, False
            except Exception:
                template_id = "base"
        # metric: record chosen arm
        try:
            from brain.obs.metrics import brain_wm_prompt_enrichment_total as _ENR
            _ENR.inc(1.0, arm=template_id)
        except Exception:
            pass

        if use_gaps:
            base = append_gap_summary(
                base,
                gap_limit=max(1, gap_n),
                max_chars=int(self._max_prompt_chars),
            )

        # Optional: include curiosity tasks recorded in the refinement queue
        try:
            use_tasks = str(os.getenv("WM_PROMPT_USE_TASKS", "0")).strip().lower() in {"1", "true", "yes", "on"}
            task_n_env = os.getenv("WM_PROMPT_TASK_N")
            task_n = int(task_n_env) if task_n_env is not None else 3
        except Exception:
            use_tasks = False
            task_n = 3
        # No second pass needed; mapping above already set both flags
        if use_tasks:
            base = append_curiosity_tasks(
                base,
                task_limit=max(1, task_n),
                max_chars=int(self._max_prompt_chars),
            )
        # Persist prompt features so propose_navigation_rules can include them in metadata
        try:
            self._prompt_features = {
                "use_gaps": bool(use_gaps),
                "use_tasks": bool(use_tasks),
                "gap_n": int(gap_n),
                "task_n": int(task_n),
                "template_id": template_id,
            }
        except Exception:
            self._prompt_features = {"use_gaps": bool(use_gaps), "use_tasks": bool(use_tasks), "template_id": template_id}
        return base

    def _call_llm(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": self._SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        def _invoke() -> str:
            text, _usage = self._model.generate(messages, temperature=0.2, max_tokens=1024)
            return text

        with ThreadPoolExecutor(max_workers=1) as executor:
            start = time.perf_counter()
            future = executor.submit(_invoke)
            try:
                response = future.result(timeout=self._timeout_s)
                latency_ms = (time.perf_counter() - start) * 1000.0
                brain_wm_llm_latency_ms.observe(latency_ms)
                return response
            except FuturesTimeout as exc:
                future.cancel()
                raise BootstrapTimeoutError("LLM teacher call timed out") from exc
            except LLMError:
                raise
            except Exception as exc:  # pragma: no cover - propagate unexpected errors
                raise BootstrapError(f"LLM teacher call failed: {exc}") from exc

    def _parse_hypotheses(self, response: str) -> Sequence[ActionHypothesis]:
        try:
            parsed = json.loads(response)
        except json.JSONDecodeError as exc:
            raise BootstrapParseError(f"response was not valid JSON: {exc}") from exc

        actions = parsed.get("actions") if isinstance(parsed, dict) else None
        if not isinstance(actions, list) or not actions:
            raise BootstrapParseError("response did not include an actions list")

        hypotheses: list[ActionHypothesis] = []
        for raw in actions:
            try:
                hypothesis = ActionHypothesis.model_validate(raw)
            except ValidationError as exc:
                raise BootstrapParseError(f"invalid hypothesis payload: {exc}") from exc
            hypotheses.append(hypothesis)
        return hypotheses

    def _enforce_safety(self, prompt: str, hypotheses: Sequence[ActionHypothesis]) -> None:
        for hypothesis in hypotheses:
            procedural_text = hypothesis.procedural_text()
            if not procedural_text:
                _LOGGER.warning(
                    "bootstrap_hypothesis_missing_steps",
                    extra={"action": hypothesis.action},
                )
                raise BootstrapSafetyError(f"hypothesis '{hypothesis.action}' lacked validation steps")
            if not self._filter.approve(
                task="bootstrap navigation world model",
                args={"prompt": prompt},
                answer=procedural_text,
            ):
                _LOGGER.warning(
                    "bootstrap_hypothesis_rejected",
                    extra={"action": hypothesis.action},
                )
                raise BootstrapSafetyError(
                    f"hypothesis '{hypothesis.action}' failed procedural safety filter",
                )

    def _collect_extra_hypotheses(self, snapshot: DomainSnapshot) -> Sequence[ActionHypothesis]:
        collected: list[ActionHypothesis] = []
        for provider in self._extra_providers:
            try:
                if hasattr(provider, "collect"):
                    extra = provider.collect(snapshot)
                else:  # pragma: no cover - callable providers
                    extra = provider(snapshot)  # type: ignore[operator]
                if extra:
                    collected.extend(list(extra))
            except Exception:
                continue
        return collected


__all__ = [
    "BootstrapError",
    "BootstrapParseError",
    "BootstrapSafetyError",
    "BootstrapTimeoutError",
    "LLMWorldModelBuilder",
]
