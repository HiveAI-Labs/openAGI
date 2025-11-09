#!/usr/bin/env python3
"""Plugin host process.

Given the path to a plugin .py file, this script loads the module in a separate
process and emits JSON metadata about available functions. This lets the main
process avoid importing plugin modules directly (avoids import-time side-effects
affecting the main process state like sys.modules or global variables).

Usage:
  python scripts/plugin_host.py /abs/path/to/plugin.py

Output (stdout):
  JSON object {"module": "<name>", "functions": [{"name": ..., "doc": ...}, ...]}
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any


FORBIDDEN_TOKENS = (
    # High-risk imports and patterns — block at discovery time
    "import subprocess",
    "from subprocess",
    "import os.system",
    "os.system(",
    "open(\"/etc/passwd",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("plugin_path")
    parser.add_argument("--base", dest="base", default=None)
    parser.add_argument("action", nargs="?")
    parser.add_argument("func_name", nargs="?")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _policy_error(**payload: Any) -> None:
    print(json.dumps(payload))
    sys.exit(9)


def _mk_sandbox(plugin_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    base = Path(tempfile.mkdtemp(prefix="pluginhost-"))
    mod_dir = base / "mod"
    cwd_dir = base / "cwd"
    home_dir = base / "home"
    mod_dir.mkdir(parents=True, exist_ok=True)
    cwd_dir.mkdir(parents=True, exist_ok=True)
    home_dir.mkdir(parents=True, exist_ok=True)
    # Copy plugin into sandbox so __file__ points to an isolated location
    sandbox_plugin = mod_dir / plugin_path.name
    shutil.copy2(str(plugin_path), str(sandbox_plugin))
    # Set process cwd and HOME
    os.chdir(str(cwd_dir))
    os.environ["HOME"] = str(home_dir)
    return base, sandbox_plugin, mod_dir, cwd_dir, home_dir


def _cleanup_sandbox(base: Path) -> None:
    try:
        shutil.rmtree(str(base), ignore_errors=True)
    except Exception:
        pass


def main() -> None:
    args = _parse_args()
    raw_path = Path(args.plugin_path)
    if not raw_path.exists():
        _policy_error(error="not_found", path=str(raw_path.resolve()))
    if raw_path.is_symlink():
        _policy_error(error="symlink_not_allowed", path=str(raw_path))

    plugin_path = raw_path.resolve()
    if not plugin_path.exists():
        _policy_error(error="not_found", path=str(plugin_path))

    base_path = Path(args.base).resolve() if args.base else None
    if base_path is not None:
        try:
            plugin_path.relative_to(base_path)
        except ValueError:
            _policy_error(
                error="outside_base",
                module=plugin_path.stem,
                path=str(plugin_path),
                base=str(base_path),
            )

    module_name = plugin_path.stem
    file_hash = _sha256(plugin_path)

    try:
        source_text = plugin_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        source_text = ""
    for token in FORBIDDEN_TOKENS:
        if token in source_text:
            _policy_error(
                error="policy_violation",
                blocked=True,
                token=token,
                module=module_name,
                sha256=file_hash,
            )

    sandbox_base, sandbox_plugin, mod_dir, cwd_dir, home_dir = _mk_sandbox(plugin_path)
    created_files_before = {entry.name for entry in mod_dir.iterdir() if entry.is_file()}
    created_dirs_before = {entry.name for entry in mod_dir.iterdir() if entry.is_dir()}

    try:
        spec = importlib.util.spec_from_file_location(module_name, str(sandbox_plugin))
        if spec is None or spec.loader is None:
            raise RuntimeError("spec_loader_missing")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[attr-defined]
    except Exception as exc:
        _cleanup_sandbox(sandbox_base)
        _policy_error(error="import_failed", module=module_name, exc=str(exc), sha256=file_hash)

    if args.action == "run":
        try:
            func_name = args.func_name
            if not func_name:
                _policy_error(error="missing_function_name")
            try:
                payload = json.load(sys.stdin)
            except Exception:
                payload = {"args": [], "kwargs": {}}
            fn_args = payload.get("args", []) if isinstance(payload, dict) else []
            fn_kwargs = payload.get("kwargs", {}) if isinstance(payload, dict) else {}
            try:
                func = getattr(module, func_name)
            except AttributeError as exc:
                _policy_error(ok=False, error=f"no_such_function: {exc}")
            try:
                result = func(*fn_args, **fn_kwargs)
            except Exception as exc:  # pragma: no cover - exercised in integration tests
                import traceback

                _policy_error(ok=False, error=str(exc), traceback=traceback.format_exc())
            try:
                json.dumps(result)
                output = {"ok": True, "result": result}
            except Exception:
                output = {"ok": True, "result": str(result)}
            print(json.dumps(output))
            return
        finally:
            _cleanup_sandbox(sandbox_base)

    created_files_after = {entry.name for entry in mod_dir.iterdir() if entry.is_file()}
    created_dirs_after = {entry.name for entry in mod_dir.iterdir() if entry.is_dir()}
    new_files_raw = [name for name in (created_files_after - created_files_before) if name != sandbox_plugin.name]
    new_dirs_raw = list(created_dirs_after - created_dirs_before)
    new_files = sorted(name for name in new_files_raw if not (name.endswith('.pyc') or name.endswith('.pyo')))
    new_dirs = sorted(name for name in new_dirs_raw if name != '__pycache__')

    functions = []
    for name, obj in inspect.getmembers(module, inspect.isfunction):
        if getattr(obj, "__module__", None) == module_name:
            functions.append({"name": name, "doc": inspect.getdoc(obj) or ""})

    payload = {
        "module": module_name,
        "functions": functions,
        "sha256": file_hash,
        "source_path": str(plugin_path),
    }
    if new_files or new_dirs:
        payload.update({
            "error": "import_side_effect",
            "created_files": new_files,
            "created_dirs": new_dirs,
        })
    print(json.dumps(payload))
    _cleanup_sandbox(sandbox_base)


if __name__ == "__main__":  # pragma: no cover
    main()
