
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request


class LLMError(RuntimeError): ...
class ModelClient:
    def generate(self, messages: list[dict[str, str]], *, temperature: float = 0.2, max_tokens: int = 1024) -> tuple[str, dict]:
        raise NotImplementedError

class QwenHttpClient(ModelClient):
    """Minimal HTTP adapter for a local Qwen2.5 endpoint (vLLM/OpenAI-compatible).
    Configure with env:
      - QWEN_API_URL (e.g., http://localhost:8000/v1/chat/completions)
      - QWEN_API_KEY (optional)
      - QWEN_MODEL (e.g., qwen2.5-coder)
    """

    def __init__(self, api_url: str, api_key: str | None = None, model: str | None = None, timeout_s: float = 30.0, retries: int = 2, backoff_s: float = 0.5) -> None:
        self.api_url = api_url
        self.api_key = api_key
        self.model = model or "qwen2.5-coder"
        self.timeout_s = timeout_s
        self.retries = retries
        self.backoff_s = backoff_s

    def generate(self, messages: list[dict[str, str]], *, temperature: float = 0.2, max_tokens: int = 1024) -> tuple[str, dict]:
        payload = {"model": self.model, "messages": messages, "temperature": float(temperature), "max_tokens": int(max_tokens)}
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(self.api_url, data=data, headers=headers, method="POST")
        last_err = None
        for attempt in range(self.retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    body = resp.read().decode("utf-8")
                    js = json.loads(body)
                    # OpenAI-compatible
                    text = js["choices"][0]["message"]["content"]
                    usage = js.get("usage", {})
                    return text, usage
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
                last_err = e
                time.sleep(self.backoff_s * (2**attempt))
        raise LLMError(f"QwenHttpClient failed after retries: {last_err}")

class FakeModelClient(ModelClient):
    """Deterministic in-memory model used for tests.

    The client synthesizes a minimal plan that executes a single `llm` step which
    simply echoes a canned JSON plan. For general chat-style calls, it returns the
    most recent user message (effectively acting as an echo server). This keeps
    `/brain/run` and eval tooling deterministic without touching real backends.
    """

    def __init__(self, responses: list[str] | None = None) -> None:
        self._responses = list(responses or [])
        base_plan = {"steps": [{"action": "echo", "args": {"text": "ok"}}]}
        # When the planner asks for a steps array, respond with a single llm step
        # whose prompt is already the desired final JSON string. Execution then
        # echoes that string, satisfying downstream validations.
        self._echo_payload = json.dumps(base_plan, ensure_ascii=False)
        self._planner_payload = json.dumps(
            {
                "steps": [
                    {
                        "action": "llm",
                        "args": {
                            "prompt": self._echo_payload,
                            "temperature": 0.0,
                            "max_tokens": 256,
                        },
                    },
                ],
            },
            ensure_ascii=False,
        )

    def _looks_like_planner(self, messages: list[dict[str, str]]) -> bool:
        for msg in messages:
            content = str(msg.get("content", ""))
            if "Return a JSON object with a 'steps' array" in content:
                return True
            if "Extract or convert it into valid JSON" in content:
                return True
        return False

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> tuple[str, dict]:
        usage = {"prompt_tokens": 0, "completion_tokens": 0}

        if self._responses:
            return self._responses.pop(0), dict(usage)

        if self._looks_like_planner(messages):
            return self._planner_payload, dict(usage)

        # Default behaviour: echo the last user/system message to keep flows simple.
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("content"):
                return str(msg["content"]), dict(usage)

        return "ok", dict(usage)

__all__ = [
    "LLMError",
    "ModelClient",
    "QwenHttpClient",
    "FakeModelClient",
]
