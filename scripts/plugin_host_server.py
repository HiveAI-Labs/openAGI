#!/usr/bin/env python3
"""Persistent plugin host.

Run this script with a plugin path to start a long-lived host that accepts
JSON commands (one per line) on stdin and writes JSON responses on stdout (one
per line). Commands:
  {"cmd": "run", "func": "name", "args": [...], "kwargs": {...}}

Responses are JSON objects with {"ok": True, "result": ...} or {"ok": False, "error": ...}

This allows the caller to spawn a host once and execute multiple plugin calls
without repeated process startup overhead.
"""
from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import os
import shutil
import signal
import sys
import tempfile
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("plugin_path")
    parser.add_argument("--base", dest="base", default=None)
    return parser.parse_args()


def _policy_error(**payload) -> None:
    print(json.dumps(payload))
    sys.stdout.flush()
    sys.exit(9)


def load_module_from_path(p: Path):
    module_name = p.stem
    spec = importlib.util.spec_from_file_location(module_name, str(p))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def main():
    args = _parse_args()
    raw_path = Path(args.plugin_path)
    if not raw_path.exists():
        _policy_error(error="not_found", path=str(raw_path.resolve()))
    if raw_path.is_symlink():
        _policy_error(error="symlink_not_allowed", path=str(raw_path))
    p = raw_path.resolve()
    if not p.exists():
        _policy_error(error="not_found", path=str(p))

    base_path = Path(args.base).resolve() if args.base else None
    if base_path is not None:
        try:
            p.relative_to(base_path)
        except ValueError:
            _policy_error(error="outside_base", path=str(p), base=str(base_path))

    # Create sandbox environment
    base = Path(tempfile.mkdtemp(prefix="pluginhost-"))
    mod_dir = base / "mod"
    cwd_dir = base / "cwd"
    home_dir = base / "home"
    mod_dir.mkdir(parents=True, exist_ok=True)
    cwd_dir.mkdir(parents=True, exist_ok=True)
    home_dir.mkdir(parents=True, exist_ok=True)
    sandbox_plugin = mod_dir / p.name
    shutil.copy2(str(p), str(sandbox_plugin))
    os.chdir(str(cwd_dir))
    os.environ["HOME"] = str(home_dir)

    def _cleanup():
        try:
            shutil.rmtree(str(base), ignore_errors=True)
        except Exception:
            pass

    # Ensure cleanup on SIGTERM/SIGINT and normal exit
    def _handle_signal(signum, frame):
        _cleanup()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    try:
        mod = load_module_from_path(sandbox_plugin)
    except Exception as exc:
        _cleanup()
        _policy_error(error="import_failed", exc=str(exc))

    # Emit ready
    sys.stdout.write(json.dumps({"ok": True, "ready": True}) + "\n")
    sys.stdout.flush()

    for line in sys.stdin:
        if not line:
            break
        line = line.strip()
        if not line:
            continue
        try:
            cmd = json.loads(line)
        except Exception as e:
            sys.stdout.write(json.dumps({"ok": False, "error": f"json_parse:{e}"}) + "\n")
            sys.stdout.flush()
            continue
        if cmd.get("cmd") == "run":
            func_name = cmd.get("func")
            args = cmd.get("args", []) or []
            kwargs = cmd.get("kwargs", {}) or {}
            try:
                fn = getattr(mod, func_name)
            except Exception as e:
                sys.stdout.write(json.dumps({"ok": False, "error": f"no_such_function:{e}"}) + "\n")
                sys.stdout.flush()
                continue
            try:
                res = fn(*args, **kwargs)
                try:
                    json.dumps(res)
                    out = {"ok": True, "result": res}
                except Exception:
                    out = {"ok": True, "result": str(res)}
                sys.stdout.write(json.dumps(out) + "\n")
                sys.stdout.flush()
            except Exception as e:
                import traceback

                tb = traceback.format_exc()
                sys.stdout.write(json.dumps({"ok": False, "error": str(e), "traceback": tb}) + "\n")
                sys.stdout.flush()
        else:
            sys.stdout.write(json.dumps({"ok": False, "error": "unknown_cmd"}) + "\n")
            sys.stdout.flush()

    # Normal EOF — cleanup
    _cleanup()


if __name__ == "__main__":
    main()
