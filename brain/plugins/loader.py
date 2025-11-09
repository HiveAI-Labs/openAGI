
from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
import importlib
import inspect
import json
import os
from pathlib import Path, Path as _Path
import re
import subprocess
import sys
from typing import Any

from brain.obs.metrics import brain_plugin_host_outcomes_total


class PluginLoader:
    def __init__(self, base_path: str | Path | None = None) -> None:
        self.base_path = Path(base_path).resolve() if base_path else None
        self._loaded: dict[str, object] = {}
        self._functions: dict[str, Callable[..., object]] = {}
        # hosted metadata for plugins loaded via a separate process
        self._hosted: dict[str, dict[str, Any]] = {}
        # persistent host server processes keyed by module name
        self._host_servers: dict[str, subprocess.Popen] = {}

    def _ensure_path(self) -> None:
        if self.base_path:
            p = str(self.base_path)
            if p not in sys.path:
                sys.path.insert(0, p)

    @staticmethod
    def _host_env(additional: dict[str, str] | None = None) -> dict[str, str]:
        env = {
            "PYTHONIOENCODING": "utf-8",
        }
        path = os.environ.get("PATH")
        if path is not None:
            env["PATH"] = path
        pythonpath = os.environ.get("PYTHONPATH")
        if pythonpath is not None:
            env["PYTHONPATH"] = pythonpath
        if additional:
            env.update(additional)
        return env

    def load_attr(self, dotted_path: str):
        mod_path, _, attr = dotted_path.rpartition(":")
        if not mod_path:
            raise ValueError("expected module:attr")
        original_sys_path = list(sys.path)
        try:
            self._ensure_path()
            mod = importlib.import_module(mod_path)
            return getattr(mod, attr)
        finally:
            # Restore sys.path to avoid leaking plugin base paths into caller
            sys.path[:] = original_sys_path

    def load_all(self, hosted: bool = True, host_script: str | None = None, *, timeout: float = 5.0) -> None:
        """Discover and register plugin functions.

        By default this method uses hosted discovery (safer): it spawns a separate
        process that imports the plugin and returns metadata, avoiding importing
        plugin modules into the main process. To preserve legacy behavior you can
        call `load_all(hosted=False)` which will import modules in-process.
        """
        if not self.base_path:
            return
        if hosted:
            # Use the hosted discovery flow
            self.load_all_hosted(host_script, timeout=timeout)
            # Convert hosted metadata into the internal functions mapping for
            # compatibility: we don't have callable objects, but we record
            # placeholders that callers can inspect.
            for mod_name, meta in self._hosted.items():
                # Skip registering functions for modules flagged with governance errors
                if meta.get("error"):
                    continue
                funcs = meta.get("functions") or []
                for f in funcs:
                    fq = f"{mod_name}.{f.get('name')}"
                    # Store a lightweight proxy that executes the function via the hosted RPC
                    def _make_proxy(_mod: str, _fn: str):
                        def _proxy(*a, **k):
                            return self.execute_hosted(_mod, _fn, args=list(a), kwargs=k)
                        return _proxy

                    self._functions[fq] = _make_proxy(mod_name, f.get("name"))
            return
        # Legacy in-process import behavior (opt-in)
        import os as _os
        allow_inprocess = str(_os.environ.get("BRAIN_ALLOW_INPROCESS_PLUGINS", "")).strip().lower() in {"1", "true", "yes", "on"}
        if not allow_inprocess:
            raise RuntimeError("in-process plugin import requires BRAIN_ALLOW_INPROCESS_PLUGINS=1 opt-in")
        self._ensure_path()
        for file in self.base_path.glob("*.py"):
            mod_name = file.stem
            mod = importlib.import_module(mod_name)
            self._loaded[mod_name] = mod
            for name, obj in inspect.getmembers(mod, inspect.isfunction):
                fq = f"{mod_name}.{name}"
                self._functions[fq] = obj

    def load_all_hosted(self, host_script: str | None = None, *, timeout: float = 5.0) -> None:
        """Discover plugins by running a plugin host script in a separate process.

        For each `*.py` in `base_path` this method runs the host script with the
        absolute path and expects a JSON object on stdout like:
          {"module": "name", "functions": [{"name": "fn", "doc": "..."}, ...]}

        This avoids importing plugin modules into the main process (no sys.modules
        entries will be created as a result of discovery).
        """
        if not self.base_path:
            return
        host = host_script or str(_Path(__file__).resolve().parents[2] / "scripts" / "plugin_host.py")
        for file in self.base_path.glob("*.py"):
            cmd = [sys.executable, host]
            if self.base_path:
                cmd.extend(["--base", str(self.base_path)])
            cmd.append(str(file))
            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    check=False,
                    close_fds=True,
                    env=self._host_env(),
                    cwd=str(self.base_path),
                )
            except Exception as e:
                self._hosted[file.stem] = {"error": str(e)}
                self._record_plugin_metric("load", "error", e)
                continue
            if proc.returncode != 0:
                # try parsing JSON error if present
                try:
                    j = json.loads(proc.stdout or proc.stderr or "{}")
                except Exception:
                    j = {"error": f"nonzero_exit({proc.returncode})", "stdout": proc.stdout, "stderr": proc.stderr}
                self._hosted[file.stem] = j
                self._record_plugin_metric("load", "error", j.get("error") or f"exit_{proc.returncode}")
                continue
            try:
                j = json.loads(proc.stdout)
            except Exception as e:
                self._hosted[file.stem] = {"error": f"json_parse:{e}", "stdout": proc.stdout}
                self._record_plugin_metric("load", "error", "json_parse")
                continue
            self._hosted[file.stem] = j
            if j.get("error"):
                self._record_plugin_metric("load", "error", j.get("error"))
                continue
            self._record_plugin_metric("load", "success", None)

    def execute_hosted(self, module_name: str, func_name: str, args: list | None = None, kwargs: dict | None = None, host_script: str | None = None, *, timeout: float = 10.0) -> Any:
        """Execute a function in a plugin via the hosted plugin host.

        module_name: stem of the plugin file (e.g., 'my_plugin' for my_plugin.py)
        func_name: name of the function to call
        args/kwargs: JSON-serializable arguments
        Returns the function result (or raises RuntimeError on failure).
        """
        if not self.base_path:
            raise RuntimeError("no base_path configured for PluginLoader")
        # If persistent hosting is enabled via environment, reuse or start a server
        use_persistent = (str(__import__("os").environ.get("BRAIN_PLUGIN_HOST_PERSISTENT", "0")).strip().lower() in {"1", "true", "yes", "on"})
        if use_persistent:
            proc = self._host_servers.get(module_name)
            if proc is None or proc.poll() is not None:
                proc = self.start_host_server(module_name, host_script=host_script)
                self._host_servers[module_name] = proc
            return self.execute_via_server(proc, func_name, args=args, kwargs=kwargs, timeout=timeout)

        # One-off execution path (default)
        plugin_path = self.base_path.joinpath(f"{module_name}.py")
        if not plugin_path.exists():
            self._record_plugin_metric("execute", "error", "not_found")
            raise FileNotFoundError(f"plugin not found: {plugin_path}")
        host = host_script or str(_Path(__file__).resolve().parents[2] / "scripts" / "plugin_host.py")
        cmd = [sys.executable, host]
        if self.base_path:
            cmd.extend(["--base", str(self.base_path)])
        cmd.append(str(plugin_path))
        cmd.extend(["run", func_name])
        payload = {"args": args or [], "kwargs": kwargs or {}}
        try:
            proc = subprocess.run(
                cmd,
                input=json.dumps(payload),
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
                close_fds=True,
                env=self._host_env(),
                cwd=str(self.base_path),
            )
        except Exception as e:
            self._record_plugin_metric("execute", "error", e)
            raise RuntimeError(f"host_exec_failed: {e}")
        out_text = proc.stdout or proc.stderr
        try:
            j = json.loads(out_text)
        except Exception as e:
            self._record_plugin_metric("execute", "error", "json_parse")
            raise RuntimeError(f"host_output_not_json: {e}; out={out_text!r}")
        if not j.get("ok"):
            self._record_plugin_metric("execute", "error", j.get("error"))
            raise RuntimeError(f"host_exec_error: {j.get('error')}")
        self._record_plugin_metric("execute", "success", None)
        return j.get("result")

    def start_host_server(self, module_name: str, host_script: str | None = None) -> subprocess.Popen:
        """Start a persistent host server for a plugin module and return the Popen handle.

        The caller is responsible for terminating the process. The server prints a
        ready JSON line on stdout which we consume before returning.
        """
        if not self.base_path:
            raise RuntimeError("no base_path configured for PluginLoader")
        plugin_path = self.base_path.joinpath(f"{module_name}.py")
        if not plugin_path.exists():
            raise FileNotFoundError(f"plugin not found: {plugin_path}")
        host = host_script or str(_Path(__file__).resolve().parents[2] / "scripts" / "plugin_host_server.py")
        cmd = [sys.executable, host]
        if self.base_path:
            cmd.extend(["--base", str(self.base_path)])
        cmd.append(str(plugin_path))
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            close_fds=True,
            env=self._host_env(),
            cwd=str(self.base_path),
        )
        # read ready line
        ready_line = proc.stdout.readline()
        try:
            j = json.loads(ready_line)
        except Exception:
            # Not ready or failed to start; collect stderr
            err = proc.stderr.read() if proc.stderr else ""
            proc.kill()
            raise RuntimeError(f"host_start_failed: ready_line={ready_line!r} stderr={err!r}")
        if not j.get("ok"):
            proc.kill()
            raise RuntimeError(f"host_start_failed: {j}")
        return proc

    def execute_via_server(self, proc: subprocess.Popen, func_name: str, args: list | None = None, kwargs: dict | None = None, *, timeout: float = 10.0) -> Any:
        """Send a run command to a started host server and read a single JSON response line.

        proc: Popen returned from start_host_server
        """
        if proc.stdin is None or proc.stdout is None:
            raise RuntimeError("invalid host process pipes")
        cmd = {"cmd": "run", "func": func_name, "args": args or [], "kwargs": kwargs or {}}
        proc.stdin.write(json.dumps(cmd) + "\n")
        proc.stdin.flush()
        # Blocking read for the response line
        line = proc.stdout.readline()
        if not line:
            self._record_plugin_metric("execute", "error", "no_response")
            raise RuntimeError("no_response_from_host")
        try:
            j = json.loads(line)
        except Exception as e:
            self._record_plugin_metric("execute", "error", "json_parse")
            raise RuntimeError(f"invalid_json_response: {e}; line={line!r}")
        if not j.get("ok"):
            self._record_plugin_metric("execute", "error", j.get("error"))
            raise RuntimeError(f"host_exec_error: {j.get('error')}")
        self._record_plugin_metric("execute", "success", None)
        return j.get("result")

    def functions_with_prefix(self, prefix: str) -> dict[str, Callable[..., object]]:
        return {k:v for k,v in self._functions.items() if k.split(".")[-1].startswith(prefix)}

    # --- Convenience helpers for batching calls in a persistent session ---
    def hosted_metadata(self) -> dict[str, dict[str, Any]]:
        """Return a deep copy of hosted plugin metadata."""
        return deepcopy(self._hosted)

    def registered_functions(self) -> list[str]:
        """Return the fully qualified names of registered plugin functions."""
        return sorted(self._functions.keys())

    class _HostedSession:
        def __init__(self, loader: PluginLoader, module_name: str) -> None:
            self.loader = loader
            self.module_name = module_name
            self.proc: subprocess.Popen | None = None

        def __enter__(self) -> PluginLoader._HostedSession:
            self.proc = self.loader.start_host_server(self.module_name)
            return self

        def run(self, func_name: str, /, *args, **kwargs) -> Any:
            return self.loader.execute_via_server(self.proc, func_name, args=list(args), kwargs=kwargs)

        def __exit__(self, exc_type, exc, tb) -> None:
            if self.proc is None:
                return
            try:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=2.0)
                except Exception:
                    self.proc.kill()
            finally:
                pass

    def hosted_session(self, module_name: str) -> PluginLoader._HostedSession:
        """Context manager for a persistent host to batch multiple calls.

        Usage:
            with loader.hosted_session("plugin") as s:
                s.run("fn", x=1)
        """
        return PluginLoader._HostedSession(self, module_name)

    def shutdown_persistent_hosts(self) -> None:
        """Terminate any persistent host servers started by execute_hosted()."""
        for name, proc in list(self._host_servers.items()):
            try:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=2.0)
                    except Exception:
                        proc.kill()
            finally:
                self._host_servers.pop(name, None)

    @staticmethod
    def _record_plugin_metric(event: str, outcome: str, reason: object) -> None:
        """Record plugin host telemetry without raising on metric failures."""
        try:
            slug = PluginLoader._normalize_reason(reason)
            brain_plugin_host_outcomes_total.inc(event=event, outcome=outcome, reason=slug)
        except Exception:
            # Metrics recording must never interfere with plugin governance path.
            pass

    @staticmethod
    def _normalize_reason(reason: object) -> str:
        """Convert arbitrary reason payloads into a bounded, lowercase slug."""
        if reason is None:
            return "none"
        if isinstance(reason, BaseException):
            base = reason.__class__.__name__ or "exception"
        else:
            base = str(reason or "unknown")
        base = base.strip().splitlines()[0]
        if ":" in base:
            base = base.split(":", 1)[0]
        base = base.lower()
        base = re.sub(r"[^a-z0-9]+", "_", base).strip("_")
        return base or "unknown"
