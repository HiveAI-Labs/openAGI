from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import threading
import time

from brain.io.validators import LLMResponse, NormalizedRequest

LOGGER = logging.getLogger(__name__)


def _context_block(req: NormalizedRequest) -> str:
    lines = []
    for entry in req.context:
        snippet = entry.text.strip().replace("\n", " ")
        lines.append(f"ID={entry.cid} SHA256={entry.sha256} TEXT={snippet}")
    return "\n".join(lines)


def append_learning_example(
    artifacts_dir: str | Path,
    req: NormalizedRequest,
    response: LLMResponse,
    citations_hash: str,
    *,
    run_id: str,
) -> tuple[Path, bool, dict[str, object] | None]:
    """Append an SFT-style learning example derived from the IO response.

    Returns a tuple of ``(dataset_path, appended, entry)`` where ``entry`` is the
    serialized record when a new row was appended (``None`` when the example was
    already present).
    """
    base = Path(artifacts_dir)
    target = base / "learn" / "sft.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)

    prompt_sections = [
        "# IO Query",
        f"Question: {req.query}",
        "# Approved Context",
        _context_block(req) or "<none>",
        "# Instruction",
        "Provide the verified answer using cited sources.",
        "Answer:",
    ]
    prompt = "\n".join(prompt_sections)

    entry: dict[str, object] = {
        "prompt": prompt,
        "completion": response.answer,
        "metadata": {
            "query_hash": req.query_hash,
            "citations_hash": citations_hash,
            "run_id": run_id,
            "source": "io_query",
        },
    }

    duplicate = False
    if target.exists():
        try:
            with target.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    meta = obj.get("metadata") if isinstance(obj, dict) else {}
                    if not isinstance(meta, dict):
                        continue
                    if (
                        meta.get("query_hash") == req.query_hash
                        and meta.get("citations_hash") == citations_hash
                    ):
                        duplicate = True
                        break
        except Exception:  # pragma: no cover - best effort
            LOGGER.exception("Failed scanning sft.jsonl for duplicates")
    if duplicate:
        return target, False, None

    line = json.dumps(entry, ensure_ascii=False)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")

    return target, True, entry


def _train_adapter(cfg, artifacts_dir: str | Path) -> dict[str, object]:
    os.environ.setdefault("ARTIFACTS_DIR", str(artifacts_dir))
    try:
        from training.adapters import lora_runner
    except Exception as exc:  # pragma: no cover - training libs optional
        LOGGER.warning("io_autotrain unavailable: %s", exc)
        return {"ok": False, "error": "train_adapter_import_failed", "detail": str(exc)}

    name = f"io-auto-{int(time.time())}"
    base_model = cfg.io_autotrain_base_model or cfg.hf_base_model or "sshleifer/tiny-gpt2"
    try:
        result = lora_runner.train_adapter(
            name=name,
            base_model=base_model,
            epochs=max(1, int(cfg.io_autotrain_epochs)),
            lora_r=max(1, int(cfg.io_autotrain_lora_r)),
        )
    except Exception as exc:  # pragma: no cover - guarded execution
        LOGGER.exception("io_autotrain failed: %s", exc)
        return {"ok": False, "error": "train_adapter_exception", "detail": str(exc)}

    return result


_TRAIN_LOCK = threading.Lock()


def trigger_autotrain(
    cfg,
    artifacts_dir: str | Path,
    *,
    run_id: str,
    on_result=None,
) -> None:
    mode = (cfg.io_autotrain_mode or "off").lower()
    if mode not in {"always", "accept"}:
        return

    def _worker(on_result=None) -> None:
        with _TRAIN_LOCK:
            result = _train_adapter(cfg, artifacts_dir)
            LOGGER.info("io_autotrain result: %s", result)
            if on_result is not None:
                try:
                    on_result(result)
                except Exception:  # pragma: no cover
                    LOGGER.exception("io_autotrain on_result callback failed")

    if cfg.io_autotrain_async:
        threading.Thread(
            target=_worker,
            kwargs={"on_result": on_result},
            name=f"io-autotrain-{run_id[:8]}",
            daemon=True,
        ).start()
    else:
        _worker(on_result=on_result)


def handle_accept(
    cfg,
    artifacts_dir: str | Path,
    req: NormalizedRequest,
    response: LLMResponse,
    *,
    run_id: str,
    citations_hash: str,
    workspace: str | None = None,
) -> None:
    try:
        _dataset_path, appended, entry = append_learning_example(
            artifacts_dir, req, response, citations_hash, run_id=run_id,
        )
    except Exception:  # pragma: no cover - best-effort logging
        LOGGER.exception("Failed to append learning example for run %s", run_id)
        return
    if not appended:
        return

    if entry is not None:
        try:
            ws_name = str(workspace or getattr(cfg, "workspace", "default") or "default")
            ws_path = Path(artifacts_dir) / "ws" / ws_name / "learn" / "sft.jsonl"
            ws_path.parent.mkdir(parents=True, exist_ok=True)
            with ws_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:  # pragma: no cover - workspace mirroring is best-effort
            LOGGER.exception("Failed to mirror learning example into workspace %s", workspace)

    state_path = Path(artifacts_dir) / "learn" / "autotrain_state.json"
    state = {"pending": 0, "last_run_ts": None}
    if state_path.exists():
        try:
            state.update(json.loads(state_path.read_text(encoding="utf-8")))
        except Exception:  # pragma: no cover
            LOGGER.warning("Failed to read %s", state_path)

    state["pending"] = int(state.get("pending", 0)) + 1

    def _save_state(data: dict[str, object]) -> None:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    _save_state(state)

    threshold = max(1, int(getattr(cfg, "io_autotrain_threshold", 1)))
    if state["pending"] < threshold:
        return

    def _on_result(res: dict[str, object] | None) -> None:
        try:
            if isinstance(res, dict) and res.get("ok"):
                state.update({"pending": 0, "last_run_ts": time.time()})
            _save_state(state)
        except Exception:  # pragma: no cover
            LOGGER.exception("Failed to update autotrain state")

    trigger_autotrain(cfg, artifacts_dir, run_id=run_id, on_result=_on_result)


__all__ = [
    "append_learning_example",
    "handle_accept",
    "trigger_autotrain",
]
