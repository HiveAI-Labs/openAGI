#!/usr/bin/env python3
"""Generate a governance inventory for dynamically loaded plugins.

The plugin loader imports third-party code inside a dedicated subprocess. This
utility enumerates the available plugins, captures the metadata emitted by the
host process (including SHA-256 hashes), and writes a JSON artifact that CI can
inspect. Failing plugins (policy violations, import side-effects, etc.) cause
this script to exit non-zero unless ``--allow-errors`` is provided.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from brain.plugins.loader import PluginLoader


def _default_artifact_path() -> Path:
    root = Path(os.environ.get("ARTIFACTS_DIR", "artifacts")).resolve()
    return root / "plugins" / "inventory.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def collect_inventory(*, base_path: Path, output_path: Path, allow_errors: bool = False) -> dict[str, Any]:
    loader = PluginLoader(base_path=base_path)
    loader.load_all(hosted=True)
    metadata = loader.hosted_metadata()
    registered = loader.registered_functions()

    summary = {
        "total_plugins": len(metadata),
        "registered_functions": len(registered),
        "errors": sorted(name for name, meta in metadata.items() if isinstance(meta, dict) and meta.get("error")),
    }

    payload = {
        "generated_at": _utc_now(),
        "base_path": str(base_path),
        "plugins": metadata,
        "functions": registered,
        "summary": summary,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if summary["errors"] and not allow_errors:
        raise RuntimeError(f"plugin_inventory_errors:{','.join(summary['errors'])}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect plugin sandbox inventory")
    parser.add_argument("--base", dest="base", default="brain/plugins", help="Directory containing plugin modules")
    parser.add_argument("--output", dest="output", default=None, help="Path to write the inventory JSON")
    parser.add_argument("--allow-errors", dest="allow_errors", action="store_true", help="Do not fail when plugins report errors")
    args = parser.parse_args()

    base_path = Path(args.base).resolve()
    if not base_path.exists():
        raise SystemExit(f"plugin directory not found: {base_path}")

    output_path = Path(args.output).resolve() if args.output else _default_artifact_path()

    try:
        payload = collect_inventory(base_path=base_path, output_path=output_path, allow_errors=args.allow_errors)
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc), "output": str(output_path)}))
        return 1

    print(json.dumps({
        "ok": True,
        "output": str(output_path),
        "total_plugins": payload["summary"]["total_plugins"],
        "registered_functions": payload["summary"]["registered_functions"],
        "errors": payload["summary"]["errors"],
    }))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
