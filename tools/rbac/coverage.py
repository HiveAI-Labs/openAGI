from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from brain.tools.builtin import register_builtin_tools
from brain.tools.registry import ToolRegistry, SENSITIVE_CAPABILITY_PREFIXES


@dataclass
class CoverageSummary:
    total_sensitive: int
    covered_sensitive: int
    violations: List[dict]
    checked_items: int

    @property
    def coverage(self) -> float:
        if self.total_sensitive == 0:
            return 1.0
        return self.covered_sensitive / self.total_sensitive

    @property
    def ok(self) -> bool:
        return self.coverage == 1.0 and not self.violations

    def as_dict(self) -> dict:
        return {
            "total_sensitive": self.total_sensitive,
            "covered_sensitive": self.covered_sensitive,
            "coverage": round(self.coverage, 4),
            "violations": self.violations,
            "checked_items": self.checked_items,
            "ok": self.ok,
        }


def _is_sensitive(capabilities: Sequence[str]) -> bool:
    return any(str(cap).startswith(SENSITIVE_CAPABILITY_PREFIXES) for cap in capabilities)


def _summarise_specs(specs: Iterable[ToolSpec]) -> CoverageSummary:
    total_sensitive = 0
    covered_sensitive = 0
    violations: List[dict] = []
    checked = 0

    for spec in specs:
        checked += 1
        caps = spec.capabilities or []
        if not caps:
            violations.append({"tool": spec.name, "error": "missing_capabilities"})
            continue
        if _is_sensitive(caps):
            total_sensitive += 1
            if spec.allowed_roles:
                covered_sensitive += 1
            else:
                violations.append({
                    "tool": spec.name,
                    "error": "missing_allowed_roles",
                    "capabilities": list(caps),
                })
    return CoverageSummary(total_sensitive, covered_sensitive, violations, checked)


def _collect_builtin_summary() -> CoverageSummary:
    registry = ToolRegistry()
    register_builtin_tools(registry)
    return _summarise_specs(registry.list_tools())


def _iter_promotion_specs(promotions_root: Path) -> Iterable[dict]:
    if not promotions_root.exists():
        return []
    for promotion_path in promotions_root.glob("**/promotion.json"):
        try:
            data = json.loads(promotion_path.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - defensive
            yield {
                "tool": promotion_path.stem,
                "error": f"json_error:{exc}",
                "path": str(promotion_path),
                "capabilities": [],
                "allowed_roles": [],
            }
            continue
        data.setdefault("path", str(promotion_path))
        yield data


def _summarise_promotions(promotions_root: Path) -> CoverageSummary:
    total_sensitive = 0
    covered_sensitive = 0
    violations: List[dict] = []
    checked = 0

    for record in _iter_promotion_specs(promotions_root):
        checked += 1
        caps = record.get("capabilities") or []
        roles = record.get("allowed_roles") or []
        if not isinstance(caps, list) or not all(isinstance(cap, str) for cap in caps):
            violations.append({
                "tool": record.get("tool_name") or record.get("tool"),
                "error": "invalid_capabilities",
                "path": record.get("path"),
            })
            continue
        if not isinstance(roles, list) or not all(isinstance(role, str) for role in roles):
            violations.append({
                "tool": record.get("tool_name") or record.get("tool"),
                "error": "invalid_allowed_roles",
                "path": record.get("path"),
            })
            continue
        if _is_sensitive(caps):
            total_sensitive += 1
            if roles:
                covered_sensitive += 1
            else:
                violations.append({
                    "tool": record.get("tool_name") or record.get("tool"),
                    "error": "missing_allowed_roles",
                    "capabilities": caps,
                    "path": record.get("path"),
                })
    return CoverageSummary(total_sensitive, covered_sensitive, violations, checked)


def generate_coverage(*, promotions_root: Path | None = None) -> dict:
    artifacts_dir = Path(os.getenv("BRAIN_ARTIFACTS_DIR") or "artifacts")
    root = promotions_root or artifacts_dir / "toolforge" / "promotions"

    builtin_summary = _collect_builtin_summary()
    promotions_summary = _summarise_promotions(root)
    ok = builtin_summary.ok and promotions_summary.ok
    return {
        "generated_at": int(time.time()),
        "sensitive_prefixes": list(SENSITIVE_CAPABILITY_PREFIXES),
        "builtin": builtin_summary.as_dict(),
        "promotions": promotions_summary.as_dict(),
        "ok": ok,
    }


def _default_output_path() -> Path:
    artifacts_dir = Path(os.getenv("BRAIN_ARTIFACTS_DIR") or "artifacts")
    out_dir = artifacts_dir / "tools" / "rbac"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / "coverage.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate RBAC coverage report for tools")
    parser.add_argument("--promotions-root", type=Path, default=None, help="override promotions directory")
    parser.add_argument("--output", type=Path, default=None, help="output path for coverage JSON")
    args = parser.parse_args()

    report = generate_coverage(promotions_root=args.promotions_root)
    output_path = args.output or _default_output_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
