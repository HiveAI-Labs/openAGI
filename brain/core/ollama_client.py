
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

from brain.core.model_client import LLMError, ModelClient


class OllamaChatClient(ModelClient):
    """Minimal client for Ollama's /api/chat endpoint (non-OpenAI schema).
    Expects a local Ollama daemon (default http://localhost:11434).
    """

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "qwen2.5-coder",
                 timeout_s: float = 30.0, retries: int = 2, backoff_s: float = 0.5):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self.retries = retries
        self.backoff_s = backoff_s

    def generate(self, messages: list[dict[str, str]], *, temperature: float = 0.2, max_tokens: int = 1024) -> tuple[str, dict]:
        # Pull configuration from instance if available; fall back to safe defaults.
        # This makes the method resilient when attached to a minimal test stub class.
        base_url = getattr(self, "base_url", "http://localhost:11434")
        try:
            base_url = base_url.rstrip("/")
        except Exception:
            base_url = "http://localhost:11434"
        model = getattr(self, "model", "qwen2.5-coder")
        timeout_s = float(getattr(self, "timeout_s", 30.0))
        retries = int(getattr(self, "retries", 2))
        backoff_s = float(getattr(self, "backoff_s", 0.5))

        # Try multiple Ollama-compatible endpoints to support different Ollama versions
        # Prefer OpenAI-compatible /v1/chat/completions when available, fall back to legacy /api/chat.
        endpoints = [
            (f"{base_url}/v1/chat/completions", "openai"),
            (f"{base_url}/api/chat", "legacy"),
        ]

        headers = {"Content-Type": "application/json"}
        last_err: Exception | None = None

        for attempt in range(retries + 1):
            for url, kind in endpoints:
                try:
                    if kind == "openai":
                        payload = {
                            "model": model,
                            "messages": messages,
                            "temperature": float(temperature),
                            "max_tokens": int(max_tokens),
                        }
                    else:
                        # legacy Ollama chat format
                        payload = {
                            "model": model,
                            "messages": messages,
                            "stream": False,
                            "options": {"temperature": float(temperature), "num_predict": int(max_tokens)},
                        }

                    data = json.dumps(payload).encode("utf-8")
                    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
                    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                        body = resp.read().decode("utf-8")
                        try:
                            js = json.loads(body)
                        except Exception:
                            js = {}

                        # Parse multiple possible response shapes
                        text = ""
                        if kind == "openai":
                            # OpenAI-style: choices[*].message.content or choices[*].text
                            choices = js.get("choices") or []
                            if choices:
                                first = choices[0]
                                text = (
                                    first.get("message", {}).get("content")
                                    or first.get("text")
                                    or ""
                                )
                            else:
                                # Some Ollama builds return {'response': '...'} or {'message': {'content': '...'}}
                                text = js.get("response") or js.get("message", {}).get("content", "")
                        else:
                            text = js.get("message", {}).get("content", "") or js.get("response", "")

                        usage = {"prompt_tokens": 0, "completion_tokens": 0}
                        if text:
                            return text, usage
                        # empty response — treat as error to try other endpoints / retries
                        last_err = RuntimeError(f"empty response from {url}")
                except urllib.error.HTTPError as e:
                    try:
                        body = e.read().decode("utf-8", errors="ignore")
                    except Exception:
                        body = None
                    last_err = e
                    logging = __import__("logging")
                    logging.error(f"Ollama HTTPError (url={url} status={getattr(e, 'code', None)}): {body}")
                except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
                    last_err = e

            # exponential backoff before next attempt
            time.sleep(backoff_s * (2 ** attempt))

        raise LLMError(f"OllamaChatClient failed after retries: {last_err}")
