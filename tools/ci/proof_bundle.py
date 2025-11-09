import json
import math
import hashlib
import subprocess
import os
import shlex
import sys
import time
from collections import deque
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import xml.etree.ElementTree as ET
import tarfile

try:
    from tools.rbac.coverage import generate_coverage
except Exception:  # pragma: no cover - coverage tooling optional during import
    generate_coverage = None

try:
    from tools.ci.plugin_inventory import collect_inventory as _plugin_collect_inventory
except Exception:  # pragma: no cover - inventory tooling optional during import
    _plugin_collect_inventory = None

from brain.obs.metrics import metrics_snapshot
from brain.obs.enhanced_observability import (
    curriculum_ack_summary,
    curriculum_dashboard_snapshot,
    selector_consult_snapshot,
    generate_alert_ack_signature,
    load_curriculum_alert_records,
    resolve_alert_ack_signing_key,
    verify_alert_ack_signature,
)
from brain.safety.fuzz_recorder import FuzzRecord, read_fuzz_records
from brain.io.atomic import atomic_write_json
from tools.ci.curriculum_mttr import build_mttr_gate_payload, resolve_mttr_config
from brain.consistency.goals import build_goals_from_contradictions
from experiments.skills.compose_from_capabilities import compose_and_score
from tools.skills.register_from_candidates import register_skills


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def _split_extra_args(payload: str) -> List[str]:
    try:
        return shlex.split(payload.strip())
    except Exception:
        return [part for part in payload.split() if part]


def _artifact_key(path: Path, artifacts_root: Path) -> str:
    try:
        rel = path.resolve().relative_to(artifacts_root.resolve())
        return str(Path("artifacts") / rel)
    except Exception:
        try:
            rel = path.relative_to(artifacts_root)
            return str(Path("artifacts") / rel)
        except Exception:
            return str(path)


def _artifact_dir_key(path: Path, artifacts_root: Path) -> str:
    try:
        rel = path.resolve().relative_to(artifacts_root.resolve())
        return str(Path("artifacts") / rel)
    except Exception:
        try:
            rel = path.relative_to(artifacts_root)
            return str(Path("artifacts") / rel)
        except Exception:
            return str(path)


def _file_metadata(artifacts_root: Path, path: Path) -> Dict[str, object]:
    stat = path.stat()
    key = _artifact_key(path, artifacts_root)
    return {
        "path": key,
        "size": stat.st_size,
        "mtime": stat.st_mtime,
        "sha256": sha256_file(path),
    }


def _add_file(
    proof: Dict[str, object],
    artifacts_root: Path,
    path: Path,
    *,
    verify_hash: bool = True,
    verify_size: bool = True,
) -> Dict[str, object]:
    meta = _file_metadata(artifacts_root, path)
    if not verify_hash:
        meta["verify_hash"] = False
    if not verify_size:
        meta["verify_size"] = False
    files = proof.setdefault("files", {})  # type: ignore[assignment]
    if isinstance(files, dict):
        rel_key = meta["path"]
        files[rel_key] = meta
        abs_key = str(path.resolve())
        if abs_key != rel_key:
            # Backward/forward compatibility:
            # - Keep canonical relative key under artifacts/
            # - Also expose absolute path key for tests that dereference via absolute paths
            files.setdefault(abs_key, meta)
            aliases = proof.setdefault("file_aliases", {})  # type: ignore[assignment]
            if isinstance(aliases, dict):
                aliases.setdefault(abs_key, rel_key)
    return meta


def _sanitize_repro_payload(payload: Any) -> Dict[str, Any]:
    """Return a compact, safe subset of a repro manifest payload.

    We only keep known, bounded-size keys and ignore unknown fields. Nested
    dict values are preserved as-is; callers should treat this as metadata,
    not a source of truth for outputs.
    """
    if not isinstance(payload, dict):
        return {}

    allow_keys = (
        "seed",
        "seeds",
        "command",
        "version",
        "num_tasks",
        "inputs",
        "outputs",
        "metrics",
        "expected_hash",
    )
    sanitized: Dict[str, Any] = {}
    for key in allow_keys:
        try:
            value = payload.get(key)
        except Exception:
            value = None
        if value is not None:
            sanitized[key] = value
    return sanitized


def _collect_repro_artifacts(proof: Dict[str, Any], artifacts_root: Path) -> None:
    repro_root = artifacts_root / "repro"
    entries: List[Dict[str, Any]] = []
    missing = False

    if repro_root.exists() and repro_root.is_dir():
        for repro_file in sorted(repro_root.glob("*.json")):
            entry: Dict[str, Any] = {}
            meta: Optional[Dict[str, Any]] = None
            try:
                meta = _add_file(proof, artifacts_root, repro_file)
            except Exception:
                meta = None

            entry["path"] = meta.get("path") if isinstance(meta, dict) else str(repro_file.resolve())
            if isinstance(meta, dict):
                for key in ("sha256", "size"):
                    if key in meta:
                        entry[key] = meta[key]

            try:
                raw_payload = json.loads(repro_file.read_text(encoding="utf-8"))
            except Exception as exc:
                entry["error"] = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
            else:
                sanitized = _sanitize_repro_payload(raw_payload)
                if sanitized:
                    entry["details"] = sanitized
                    for key in ("seed", "seeds", "command", "version", "num_tasks"):
                        if key in sanitized and sanitized[key] is not None:
                            entry[key] = sanitized[key]
                    try:
                        canonical_payload = json.dumps(
                            sanitized,
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                    except TypeError:
                        normalized = json.loads(
                            json.dumps(
                                sanitized,
                                default=str,
                                sort_keys=True,
                                separators=(",", ":"),
                            )
                        )
                        entry["details"] = normalized
                        canonical_payload = json.dumps(
                            normalized,
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                    entry["payload_sha256"] = hashlib.sha256(canonical_payload.encode("utf-8")).hexdigest()

                if raw_payload.get("inputs") is not None:
                    entry["inputs_present"] = True
                if raw_payload.get("outputs") is not None:
                    entry["outputs_present"] = True

            entries.append(entry)
    else:
        missing = True

    canonical_entries = [
        {
            "path": entry.get("path"),
            "sha256": entry.get("sha256"),
            "payload_sha256": entry.get("payload_sha256"),
            "seed": entry.get("seed"),
            "command": entry.get("command"),
            "error": entry.get("error"),
        }
        for entry in entries
    ]

    determinism_token = hashlib.sha256(
        json.dumps(canonical_entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    repro_summary: Dict[str, Any] = {
        "root": str(repro_root),
        "count": len(entries),
        "entries": entries,
        "determinism_token": determinism_token,
    }
    if missing:
        repro_summary["missing"] = True

    proof["repro"] = repro_summary


def _collect_plugin_inventory(proof: Dict[str, Any], artifacts_root: Path, gates: Dict[str, Any]) -> None:
    """Collect plugin sandbox inventory metadata and attach it to the proof bundle."""

    inventory_path = artifacts_root / "plugins" / "inventory.json"
    default_inventory_path = str(inventory_path.resolve())
    plugin_section: Dict[str, Any] = {
        "present": False,
        "inventory_path": default_inventory_path,
    }
    gate_payload: Dict[str, Any] = {
        "present": False,
        "ok": False,
        "inventory_path": default_inventory_path,
    }

    base_override = os.getenv("PROOF_PLUGIN_BASE") or os.getenv("BRAIN_PLUGIN_BASE")
    candidates: List[Path] = []
    if base_override:
        candidates.append(Path(base_override))
    try:
        candidates.append(Path(__file__).resolve().parents[2] / "brain" / "plugins")
    except Exception:
        pass
    try:
        candidates.append(Path.cwd() / "brain" / "plugins")
    except Exception:
        pass

    base_path: Optional[Path] = None
    for candidate in candidates:
        try:
            if candidate.exists():
                base_path = candidate.resolve()
                break
        except Exception:
            continue
    if base_path is None and candidates:
        try:
            base_path = candidates[0].resolve()
        except Exception:
            base_path = candidates[0]

    if base_path is not None:
        plugin_section["base_path"] = str(base_path)

    payload: Optional[Dict[str, Any]] = None
    collect_error: Optional[str] = None

    if _plugin_collect_inventory and base_path and base_path.exists():
        try:
            payload = _plugin_collect_inventory(
                base_path=base_path,
                output_path=inventory_path,
                allow_errors=True,
            )
        except Exception as exc:  # pragma: no cover - defensive guard
            collect_error = f"collect_failed:{type(exc).__name__}:{exc}"
    else:
        if not _plugin_collect_inventory:
            collect_error = "collector_unavailable"
        elif not base_path:
            collect_error = "base_path_unresolved"
        elif not base_path.exists():
            collect_error = f"base_path_missing:{base_path}"

    if payload is None and inventory_path.exists():
        try:
            payload = json.loads(inventory_path.read_text(encoding="utf-8"))
        except Exception as exc:
            suffix = f"inventory_read_failed:{type(exc).__name__}:{exc}"
            collect_error = f"{collect_error};{suffix}" if collect_error else suffix

    meta: Optional[Dict[str, Any]] = None
    if inventory_path.exists():
        try:
            meta = _add_file(proof, artifacts_root, inventory_path)
        except Exception as exc:
            suffix = f"attach_failed:{type(exc).__name__}:{exc}"
            collect_error = f"{collect_error};{suffix}" if collect_error else suffix

    if meta:
        plugin_section["inventory_path"] = meta.get("path")
        plugin_section["inventory_sha256"] = meta.get("sha256")
        gate_payload["inventory_path"] = meta.get("path")

    if isinstance(payload, dict):
        plugin_section["present"] = True
        summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
        plugin_section["summary"] = summary
        plugin_section["generated_at"] = payload.get("generated_at")
        plugin_section["errors"] = list(summary.get("errors") or [])
        determinism_token = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        plugin_section["determinism_token"] = determinism_token

        gate_payload.update(
            {
                "present": True,
                "ok": not plugin_section["errors"],
                "total_plugins": summary.get("total_plugins"),
                "registered_functions": summary.get("registered_functions"),
                "errors": plugin_section["errors"] or None,
                "determinism_token": determinism_token,
            }
        )
    else:
        gate_payload["present"] = False
        gate_payload["ok"] = False

    if collect_error:
        plugin_section["collect_error"] = collect_error
        gate_payload.setdefault("error", collect_error)

    proof["plugins"] = plugin_section
    gates["plugin_inventory"] = gate_payload


def _collect_teach_phase_b_artifacts(proof: Dict[str, Any], artifacts_root: Path) -> None:
    """Collect Phase B teaching artifacts (promotions, trainer, eval scorecards).

    Artifacts scanned:
    - artifacts/teach/promoted/*.jsonl (promoted lessons datasets)
    - artifacts/teach/trainer/trainer.json (latest trainer summary)
    - artifacts/teach/trainer/ledger.jsonl (append-only ledger)
    - artifacts/teach/eval/scorecard.json (shadow evaluation outcome)

    Each file is added with sha256 metadata; promoted datasets also include
    a compact summary (count, first_sha256, determinism_token over line digests).
    """
    teach_root = artifacts_root / "teach"
    if not teach_root.exists():
        proof.setdefault("teach_phase_b", {"present": False})
        return

    payload: Dict[str, Any] = {"present": True}

    # Promotions
    promoted_root = teach_root / "promoted"
    promoted_entries: List[Dict[str, Any]] = []
    if promoted_root.exists():
        for ds in sorted(promoted_root.glob("promoted_*.jsonl")):
            try:
                meta = _add_file(proof, artifacts_root, ds)
            except Exception:
                meta = {"path": str(ds), "error": "file_meta_error"}
            entry: Dict[str, Any] = {"path": meta.get("path")}
            if meta.get("sha256"):
                entry["sha256"] = meta.get("sha256")
            # compute determinism token over line payload sha256 values
            try:
                line_digests: List[str] = []
                with ds.open("r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except Exception:
                            continue
                        # prefer embedded sha256 if present else canonical
                        embedded = obj.get("sha256")
                        if embedded and isinstance(embedded, str):
                            line_digests.append(embedded)
                        else:
                            canonical = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
                            line_digests.append(hashlib.sha256(canonical).hexdigest())
                det_token = hashlib.sha256(
                    json.dumps(line_digests, sort_keys=True, separators=(",", ":")).encode("utf-8")
                ).hexdigest()
                entry["count"] = len(line_digests)
                if line_digests:
                    entry["first_sha256"] = line_digests[0]
                entry["determinism_token"] = det_token
            except Exception as exc:
                entry["error"] = {"type": type(exc).__name__, "message": str(exc)}
            promoted_entries.append(entry)
    payload["promoted"] = {"datasets": promoted_entries, "count": len(promoted_entries)}

    # Trainer summary
    trainer_dir = teach_root / "trainer"
    trainer_summary_path = trainer_dir / "trainer.json"
    if trainer_summary_path.exists():
        try:
            meta = _add_file(proof, artifacts_root, trainer_summary_path)
            payload["trainer_summary"] = {"path": meta.get("path"), "sha256": meta.get("sha256"), "size": meta.get("size")}
            try:
                raw_obj = json.loads(trainer_summary_path.read_text(encoding="utf-8"))
                # keep bounded subset for proof surface
                keep_keys = [
                    "run_id",
                    "dataset_path",
                    "examples",
                    "actions_added",
                    "states_added",
                    "training_steps_before",
                    "training_steps_after",
                    "status",
                    "error",
                    "duration_ms",
                ]
                payload["trainer_summary"].update({k: raw_obj.get(k) for k in keep_keys if k in raw_obj})
            except Exception:
                pass
        except Exception:
            payload["trainer_summary_error"] = True

    # Trainer ledger (optional)
    ledger_path = trainer_dir / "ledger.jsonl"
    if ledger_path.exists():
        try:
            meta = _add_file(proof, artifacts_root, ledger_path)
            payload["trainer_ledger"] = {"path": meta.get("path"), "sha256": meta.get("sha256"), "size": meta.get("size")}
            # compute ledger determinism token over embedded sha256 fields
            try:
                hashes: List[str] = []
                with ledger_path.open("r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                            emb = obj.get("sha256")
                            if emb:
                                hashes.append(str(emb))
                        except Exception:
                            continue
                payload["trainer_ledger"]["entries"] = len(hashes)
                payload["trainer_ledger"]["determinism_token"] = hashlib.sha256(
                    json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
                ).hexdigest()
            except Exception:
                pass
        except Exception:
            payload["trainer_ledger_error"] = True

    # Shadow eval scorecard (optional)
    eval_dir = teach_root / "eval"
    scorecard_path = eval_dir / "scorecard.json"
    if scorecard_path.exists():
        try:
            meta = _add_file(proof, artifacts_root, scorecard_path)
            payload["shadow_eval_scorecard"] = {"path": meta.get("path"), "sha256": meta.get("sha256")}
            try:
                raw_obj = json.loads(scorecard_path.read_text(encoding="utf-8"))
                keep = [
                    "run_id",
                    "policy_success_rate",
                    "baseline_success_rate",
                    "abs_uplift",
                    "rel_uplift",
                    "cases",
                    "allowed",
                    "gating_reason",
                ]
                payload["shadow_eval_scorecard"].update({k: raw_obj.get(k) for k in keep if k in raw_obj})
            except Exception:
                pass
        except Exception:
            payload["shadow_eval_scorecard_error"] = True

    proof["teach_phase_b"] = payload


def _collect_teach_phase_a_artifacts(proof: Dict[str, Any], artifacts_root: Path) -> None:
    """Collect Phase A teaching artifacts (lesson packs + grading records).

    Artifacts scanned:
      - artifacts/lessons/pack_*.json (aggregated packs)
      - artifacts/lessons/pack_*.json.sha256 (hash sidecars)
      - artifacts/lessons/grading_*.jsonl (mirrored grading records)

    Each pack entry includes:
      {path, sha256_sidecar, embedded_provenance_sha256?, count_tasks, determinism_token}
    Determinism token is sha256 over ordered lesson prompt digests.
    Grading summary includes pass/fail counts and a determinism token over
    embedded grade sha256 fields.
    """
    lessons_root = artifacts_root / "lessons"
    if not lessons_root.exists():
        proof.setdefault("teach_phase_a", {"present": False})
        return
    packs: list[dict[str, Any]] = []
    for pack_path in sorted(lessons_root.glob("pack_*.json")):
        entry: dict[str, Any] = {"path": _artifact_key(pack_path, artifacts_root)}
        try:
            meta = _add_file(proof, artifacts_root, pack_path)
            entry.update({"sha256": meta.get("sha256"), "size": meta.get("size")})
        except Exception:
            entry["error_meta"] = True
        # Sidecar digest content (if present)
        sidecar = pack_path.with_suffix(pack_path.suffix + ".sha256")
        if sidecar.exists():
            try:
                digest = sidecar.read_text(encoding="utf-8").strip().splitlines()[0]
                entry["sha256_sidecar"] = digest
            except Exception:
                entry["sha256_sidecar_error"] = True
        # Parse pack JSON to extract lesson prompts and provenance hash
        try:
            raw = json.loads(pack_path.read_text(encoding="utf-8"))
            lessons = raw.get("lessons") or []
            prompts: list[str] = []
            for ls in lessons:
                try:
                    p = str(ls.get("prompt", "")).strip()
                except Exception:
                    p = ""
                if p:
                    prompts.append(p)
            # Canonical prompt digest list
            line_digests: list[str] = []
            for p in prompts:
                canonical = json.dumps({"p": p}, sort_keys=True, separators=(",", ":")).encode("utf-8")
                line_digests.append(hashlib.sha256(canonical).hexdigest())
            token = hashlib.sha256(json.dumps(line_digests, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
            entry["count_tasks"] = len(prompts)
            entry["determinism_token"] = token
            if isinstance(raw.get("provenance_sha256"), str):
                entry["provenance_sha256"] = raw["provenance_sha256"]
        except Exception as exc:
            entry["parse_error"] = {"type": type(exc).__name__, "message": str(exc)}
        packs.append(entry)
    # Grading records mirrored into lessons dir
    grading_entries: list[dict[str, Any]] = []
    for grade_path in sorted(lessons_root.glob("grading_*.jsonl")):
        g_entry: dict[str, Any] = {"path": _artifact_key(grade_path, artifacts_root)}
        try:
            meta = _add_file(proof, artifacts_root, grade_path)
            g_entry.update({"sha256": meta.get("sha256"), "size": meta.get("size")})
        except Exception:
            g_entry["error_meta"] = True
        pass_count = 0
        fail_count = 0
        digests: list[str] = []
        try:
            with grade_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    res = obj.get("result")
                    if res == "pass":
                        pass_count += 1
                    elif res == "fail":
                        fail_count += 1
                    emb = obj.get("sha256")
                    if isinstance(emb, str):
                        digests.append(emb)
            g_entry["pass"] = pass_count
            g_entry["fail"] = fail_count
            g_entry["determinism_token"] = hashlib.sha256(json.dumps(digests, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        except Exception as exc:
            g_entry["parse_error"] = {"type": type(exc).__name__, "message": str(exc)}
        grading_entries.append(g_entry)
    proof["teach_phase_a"] = {
        "present": True,
        "packs": packs,
        "grading": grading_entries,
        "pack_count": len(packs),
        "grading_count": len(grading_entries),
    }


def _collect_packs_artifacts(proof: Dict[str, Any], artifacts_root: Path) -> None:
    """Locate packs under artifacts/packs, add them to the proof and verify manifest signatures when possible."""
    packs_root = artifacts_root / "packs"
    packs_summary: Dict[str, Any] = {}
    if not packs_root.exists() or not packs_root.is_dir():
        proof.setdefault("packs", {})["count"] = 0
        return

    key = resolve_alert_ack_signing_key()
    for p in sorted(packs_root.glob("*.tgz")) + sorted(packs_root.glob("*.tar.gz")):
        try:
            meta = _add_file(proof, artifacts_root, p)
        except Exception:
            continue
        entry: Dict[str, Any] = {"path": meta.get("path"), "sha256": meta.get("sha256"), "size": meta.get("size")}
        # Use the centralized verifier (tools.pack.verify_pack) when available.
        try:
            try:
                # local import to avoid startup time or optional import issues
                from tools.pack import verify_pack as pack_verify

                ok, details = pack_verify.verify_pack(str(p), hmac_key=key)
                # copy relevant fields into the entry for proof consumption
                entry["manifest_present"] = details.get("manifest_present")
                entry["manifest_signed"] = details.get("signature_present")
                # prefer explicit verifier results for signature validity
                if details.get("hmac_valid") is not None:
                    entry["manifest_signature_valid"] = details.get("hmac_valid")
                elif details.get("ed25519_valid") is not None:
                    entry["manifest_signature_valid"] = details.get("ed25519_valid")
                else:
                    entry["manifest_signature_valid"] = None
                # include full verifier details for auditability
                entry["verifier"] = details
            except ImportError:
                # fall back to legacy inline inspection when verifier not importable
                with tarfile.open(p, "r:*") as tar:
                    try:
                        mf = tar.extractfile("manifest.json")
                    except KeyError:
                        mf = None
                    if mf:
                        try:
                            manifest_text = mf.read().decode("utf-8")
                            manifest = json.loads(manifest_text)
                            entry["manifest_file_count"] = manifest.get("file_count")
                            entry["manifest_signed"] = bool(manifest.get("signature"))
                            if manifest.get("signature") and key:
                                # verify HMAC signature inline
                                sig = manifest.get("signature")
                                mc = dict(manifest)
                                mc.pop("signature", None)
                                mc.pop("signed_with_env", None)
                                canonical = json.dumps(mc, sort_keys=True, separators=(",", ":"))
                                import hmac as _hmac

                                expected = _hmac.new(key.encode("utf-8"), canonical.encode("utf-8"), hashlib.sha256).hexdigest()
                                entry["manifest_signature_valid"] = (expected == sig)
                            else:
                                entry["manifest_signature_valid"] = None
                        except Exception:
                            entry["manifest_parse_error"] = True
        except Exception:
            # avoid failing proof assembly; record the inspection error
            entry["inspect_error"] = True

        packs_summary[p.name] = entry

    proof.setdefault("packs", {})["files"] = packs_summary
    proof.setdefault("packs", {})["count"] = len(packs_summary)


def _collect_skill_artifacts(
    proof: Dict[str, Any],
    artifacts_root: Path,
    *,
    evaluation_summary: Optional[Path] = None,
    registry_dir: Optional[Path] = None,
) -> None:
    """Register skills from evaluation summaries and record composition metrics."""

    experiments_skills = artifacts_root / "experiments" / "skills"
    primary_summary = experiments_skills / "latest_summary.json"
    summary_path = Path(evaluation_summary) if evaluation_summary else primary_summary
    if not summary_path.exists():
        fallback_dirs = [experiments_skills, artifacts_root / "skills"]
        for directory in fallback_dirs:
            candidates = sorted(directory.glob("skills_summary_*.json"))
            if candidates:
                summary_path = candidates[-1]
                break
    if not summary_path.exists():
        return

    registry_dir = Path(registry_dir) if registry_dir else Path("skills/registry")
    registry_dir.mkdir(parents=True, exist_ok=True)

    reg_summary = register_skills(str(summary_path), str(registry_dir), "proof_bundle", None)

    metrics_root = artifacts_root / "skills" / "latest"
    metrics_path = metrics_root / "metrics.json"
    metrics = compose_and_score(registry_dir, summary_path, metrics_path)

    proof.setdefault("skills", {})["registration"] = reg_summary
    proof["skills"]["metrics"] = metrics
    proof["skills"]["summary_path"] = str(summary_path.resolve())
    proof["skills"]["registry"] = str(registry_dir.resolve())
    proof["skills"]["metrics_path"] = str(metrics_path.resolve())


def _resolve_git_commit(repo_root: Optional[Path] = None) -> Optional[str]:
    """Return the current git commit hash if available."""

    root = repo_root or Path.cwd()
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    commit = result.stdout.strip()
    return commit or None


def _collect_tests_summary(artifacts_dir: Path) -> Optional[Dict[str, object]]:
    """Collect aggregate test run statistics from known report locations."""

    candidates: List[Path] = []
    override = os.getenv("PROOF_TEST_RESULTS")
    if override:
        candidates.append(Path(override))
    candidates.extend(
        [
            artifacts_dir / "ci" / "pytest-tests-only.xml",
            artifacts_dir / "pytest-tests-only.xml",
            Path("pytest-tests-only.xml"),
        ]
    )

    for candidate in candidates:
        if not candidate:
            continue
        try:
            path = candidate if candidate.is_absolute() else candidate
        except Exception:
            continue
        if not path.exists():
            continue
        try:
            tree = ET.parse(path)
            root = tree.getroot()
        except Exception:
            continue
        total = failures = errors = skipped = 0
        timestamp: Optional[str] = None
        duration: Optional[float] = None
        for suite in root.iter("testsuite"):
            try:
                total += int(suite.attrib.get("tests", 0) or 0)
                failures += int(suite.attrib.get("failures", 0) or 0)
                errors += int(suite.attrib.get("errors", 0) or 0)
                skipped += int(suite.attrib.get("skipped", 0) or 0)
            except Exception:
                # Ignore malformed counts; continue with best-effort totals
                pass
            if timestamp is None and suite.attrib.get("timestamp"):
                timestamp = str(suite.attrib["timestamp"])
            if duration is None and suite.attrib.get("time"):
                try:
                    duration = float(suite.attrib["time"])
                except Exception:
                    duration = None
        summary: Dict[str, object] = {
            "path": str(path),
            "tests": int(total),
            "failures": int(failures),
            "errors": int(errors),
            "skipped": int(skipped),
        }
        if timestamp is not None:
            summary["timestamp"] = timestamp
        if duration is not None:
            summary["time_seconds"] = float(duration)
        return summary
    return None


def _run_phase1_tests(artifacts_dir: Path) -> Dict[str, object]:
    """Execute Phase 1 research tests and capture their logs."""

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/research/test_simulation_validation.py",
        "tests/research/test_causal_reasoner.py",
        "--disable-warnings",
        "--maxfail=1",
    ]
    ci_dir = artifacts_dir / "ci"
    ci_dir.mkdir(parents=True, exist_ok=True)
    log_path = ci_dir / "phase1_tests.log"
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except Exception as exc:  # pragma: no cover - defensive
        payload = {
            "ok": False,
            "error": f"phase1_tests_invocation_failed: {exc}",
            "command": cmd,
        }
        return payload

    combined_output = (result.stdout or "") + ("\n" + result.stderr if result.stderr else "")
    try:
        log_path.write_text(combined_output, encoding="utf-8")
    except Exception:
        pass

    payload: Dict[str, object] = {
        "ok": result.returncode == 0,
        "returncode": result.returncode,
        "command": cmd,
        "log_path": str(log_path),
    }
    return payload


def _ensure_phase1_validation(artifacts_dir: Path) -> Optional[Path]:
    """Ensure the Phase 1 validation proof exists and return its path."""

    try:
        from brain.research.simulation.validation import run_phase1_validation
    except Exception:
        return None

    proof_dir = artifacts_dir / "proof" / "phase1"
    proof_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(proof_dir.glob("phase_1.1_*.json"))
    if existing:
        return existing[-1]
    proof = run_phase1_validation(output_dir=proof_dir)
    try:
        generated = sorted(proof_dir.glob("phase_1.1_*.json"))
        if generated:
            return generated[-1]
    except Exception:
        pass
    # Final fallback: derive path from proof timestamp if accessible
    try:
        return proof_dir / f"phase_1.1_{int(proof.timestamp)}.json"
    except Exception:
        return None


def _run_world_model_retrain_validation(artifacts_dir: Path) -> Optional[Dict[str, object]]:
    """Run the world-model retrain validation automation harness."""

    if os.getenv("PROOF_SKIP_WM_RETRAIN_AUTOMATION", "0").lower() in {"1", "true", "yes", "on"}:
        return {
            "ok": True,
            "skipped": True,
            "reason": "skip_flag",
        }

    script_module = "scripts.run_world_model_retrain_validation"
    script_path = Path("scripts") / "run_world_model_retrain_validation.py"
    if not script_path.exists():
        return {
            "ok": False,
            "error": "missing_runner",
            "path": str(script_path),
        }

    workspace = os.getenv("PROOF_WM_RETRAIN_WORKSPACE") or os.getenv("BRAIN_WORKSPACE")
    if not workspace:
        candidate_workspaces = ["sim_planner_ab", "default", "world_model_capture"]
        workspace = _select_world_model_workspace(artifacts_dir, candidate_workspaces)
    reset_cache = os.getenv("PROOF_WM_RETRAIN_RESET_CACHE", "1").lower() not in {"0", "false", "no", "off"}

    cmd: List[str] = [
        sys.executable,
        "-m",
        script_module,
        f"--artifacts-dir={artifacts_dir}",
        f"--workspace={workspace}",
    ]
    if reset_cache:
        cmd.append("--reset-cache")

    extra_args = os.getenv("PROOF_WM_RETRAIN_EXTRA_ARGS")
    if extra_args:
        cmd.extend(_split_extra_args(extra_args))

    ci_dir = artifacts_dir / "ci"
    ci_dir.mkdir(parents=True, exist_ok=True)
    log_path = ci_dir / "world_model_retrain_validation.log"

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except Exception as exc:  # pragma: no cover - defensive
        return {
            "ok": False,
            "error": f"retrain_validation_invocation_failed: {exc}",
            "command": cmd,
        }

    combined_output = (result.stdout or "") + ("\n" + result.stderr if result.stderr else "")
    try:
        log_path.write_text(combined_output, encoding="utf-8")
    except Exception:
        pass

    payload: Dict[str, object] = {
        "ok": result.returncode == 0,
        "returncode": result.returncode,
        "workspace": workspace,
        "command": cmd,
        "log_path": str(log_path),
    }
    if extra_args:
        payload["extra_args"] = extra_args
    if reset_cache:
        payload["reset_cache"] = True
    return payload


def _select_world_model_workspace(artifacts_dir: Path, candidates: Sequence[str]) -> str:
    """Pick a workspace with non-empty world-model state captures."""

    def _workspace_has_states(workspace: str) -> bool:
        wm_path = artifacts_dir / "ws" / workspace / "sim" / "world_model.json"
        if not wm_path.exists():
            return False
        try:
            payload = json.loads(wm_path.read_text(encoding="utf-8"))
        except Exception:
            return False
        states = payload.get("states")
        if isinstance(states, list) and states:
            return True
        # Fallback to v2 if v1 empty
        wm_v2_path = wm_path.with_name("world_model_v2.json")
        if wm_v2_path.exists():
            try:
                payload_v2 = json.loads(wm_v2_path.read_text(encoding="utf-8"))
            except Exception:
                payload_v2 = None
            policy = payload_v2.get("policy") if isinstance(payload_v2, dict) else None
            transitions = payload_v2.get("transitions") if isinstance(payload_v2, dict) else None
            if isinstance(policy, dict) and policy and isinstance(transitions, dict) and transitions:
                return True
        return False

    for workspace in candidates:
        if _workspace_has_states(workspace):
            return workspace
    return candidates[0] if candidates else "default"


def _maybe_run_emergence_tests(artifacts_dir: Path) -> Optional[Dict[str, object]]:
    """Run emergent routing regression when the feature flag is enabled."""

    if os.getenv("BRAIN_EMERGENT_ROUTING", "0") != "1":
        return None

    cmd = [
        "pytest",
        "-q",
        "test_emergence.py",
        "--disable-warnings",
        "--maxfail=1",
    ]
    ci_dir = artifacts_dir / "ci"
    ci_dir.mkdir(parents=True, exist_ok=True)
    log_path = ci_dir / "emergence_tests.log"
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except Exception as exc:  # pragma: no cover - defensive
        return {
            "ok": False,
            "error": f"emergence_tests_invocation_failed: {exc}",
            "command": cmd,
        }

    combined_output = (result.stdout or "") + ("\n" + result.stderr if result.stderr else "")
    try:
        log_path.write_text(combined_output, encoding="utf-8")
    except Exception:
        pass

    return {
        "ok": result.returncode == 0,
        "returncode": result.returncode,
        "command": cmd,
        "log_path": str(log_path),
    }


def _semantic_metrics_snapshot() -> Dict[str, Dict[str, List[Dict[str, object]]]]:
    """Return a deterministic snapshot of semantic self-model metrics."""

    snapshot = metrics_snapshot()

    def _sorted_entries(entries: List[Dict[str, object]]) -> List[Dict[str, object]]:
        sorted_entries: List[Dict[str, object]] = []
        for entry in entries:
            labels = entry.get("labels")
            if isinstance(labels, dict):
                ordered = {k: labels[k] for k in sorted(labels)}
                entry = dict(entry)
                entry["labels"] = ordered
            sorted_entries.append(entry)
        sorted_entries.sort(
            key=lambda item: tuple(
                (k, str(v))
                for k, v in (item.get("labels") or {}).items()  # type: ignore[union-attr]
            )
        )
        return sorted_entries

    counters = {
        name: _sorted_entries(list(samples))
        for name, samples in snapshot.get("counters", {}).items()
        if name.startswith("brain_semantic_")
    }
    gauges = {
        name: _sorted_entries(list(samples))
        for name, samples in snapshot.get("gauges", {}).items()
        if name.startswith("brain_semantic_")
    }
    histograms = {
        name: _sorted_entries(list(samples))
        for name, samples in snapshot.get("histograms", {}).items()
        if name.startswith("brain_semantic_")
    }
    return {
        "counters": counters,
        "gauges": gauges,
        "histograms": histograms,
    }


def compute_planner_invariant_gates(env: Mapping[str, str] | None = None,
                                     snapshot: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    """Compute planner invariant gate payload from a metrics snapshot and env.

    This helper mirrors the inline gating logic used when attaching
    proof['planner_invariants']['gates'] and is exposed for direct unit tests
    and external CI tooling without parsing the entire proof bundle.

    Inputs:
      env: optional mapping overlay for os.environ values.
      snapshot: optional pre-fetched metrics snapshot.

    Returns:
      Dict containing keys: enabled, hard_fail, status, rates{}, thresholds{}, raw_counts{}, violations?, determinism_token.
    """
    env_map = dict(os.environ)
    if env:
        env_map.update({str(k): str(v) for k, v in env.items()})
    if snapshot is None:
        snapshot = metrics_snapshot()
    counters = snapshot.get('counters', {}) if isinstance(snapshot, dict) else {}

    def _counter_value(name: str) -> float:
        try:
            samples = counters.get(name)
            if isinstance(samples, list) and samples:
                val = samples[0].get('value')
                return float(val) if val is not None else 0.0
        except Exception:
            return 0.0
        return 0.0

    steps_total = _counter_value('planner_steps_total')
    clamp_total = _counter_value('planner_confidence_clamped_total')
    early_abstain_total = _counter_value('planner_early_abstain_total')

    enabled = env_map.get('BRAIN_INVARIANTS_GATES_ENABLE', '0') == '1'
    clamp_rate_max = float(env_map.get('BRAIN_INVARIANTS_CLAMP_RATE_MAX', '0.05'))
    early_abstain_rate_max = float(env_map.get('BRAIN_INVARIANTS_EARLY_ABSTAIN_RATE_MAX', '0.02'))
    hard_fail = env_map.get('BRAIN_INVARIANTS_GATES_HARD_FAIL', '0') == '1'

    clamp_rate = (clamp_total / steps_total) if steps_total > 0 else 0.0
    early_abstain_rate = (early_abstain_total / steps_total) if steps_total > 0 else 0.0
    status = 'pass'
    violations: list[str] = []
    if enabled:
        if clamp_rate > clamp_rate_max:
            status = 'fail'
            violations.append('clamp_rate_exceeded')
        if early_abstain_rate > early_abstain_rate_max:
            status = 'fail'
            violations.append('early_abstain_rate_exceeded')

    gate_payload: Dict[str, object] = {
        'enabled': enabled,
        'hard_fail': hard_fail,
        'status': status,
        'rates': {
            'clamp_rate': clamp_rate,
            'early_abstain_rate': early_abstain_rate,
        },
        'thresholds': {
            'clamp_rate_max': clamp_rate_max,
            'early_abstain_rate_max': early_abstain_rate_max,
        },
        'raw_counts': {
            'steps_total': steps_total,
            'clamp_total': clamp_total,
            'early_abstain_total': early_abstain_total,
        },
        'violations': violations or None,
    }
    try:
        token_src = json.dumps({
            'status': status,
            'steps_total': steps_total,
            'clamp_total': clamp_total,
            'early_abstain_total': early_abstain_total,
            'clamp_rate': clamp_rate,
            'early_abstain_rate': early_abstain_rate,
            'thresholds': gate_payload['thresholds'],
            'violations': violations,
        }, sort_keys=True, separators=(",", ":")).encode('utf-8')
        gate_payload['determinism_token'] = hashlib.sha256(token_src).hexdigest()
    except Exception:
        pass
    return gate_payload


def _plugin_host_metrics_summary() -> Optional[Tuple[Dict[str, object], Dict[str, object]]]:
    """Return structured plugin-host metric summary and a gate payload."""

    snapshot = metrics_snapshot()
    samples = snapshot.get("counters", {}).get("brain_plugin_host_outcomes_total", [])

    events: Dict[str, Dict[str, Dict[str, float]]] = {}
    total = 0.0
    total_errors = 0.0
    policy_violations = 0.0

    for sample in samples:
        labels = sample.get("labels") or {}
        event = str(labels.get("event") or "unknown")
        outcome = str(labels.get("outcome") or "unknown")
        reason = str(labels.get("reason") or "unknown")
        value = float(sample.get("value") or 0.0)
        total += value
        if outcome == "error":
            total_errors += value
        if event == "load" and outcome == "error" and reason == "policy_violation":
            policy_violations = value
        events.setdefault(event, {}).setdefault(outcome, {})[reason] = value

    events_ordered: Dict[str, Dict[str, Dict[str, float]]] = {}
    for event in sorted(events):
        outcome_map: Dict[str, Dict[str, float]] = {}
        for outcome in sorted(events[event]):
            reason_map = events[event][outcome]
            ordered_reason_map = {reason: reason_map[reason] for reason in sorted(reason_map)}
            outcome_map[outcome] = ordered_reason_map
        events_ordered[event] = outcome_map

    totals = {
        "total": total,
        "errors": total_errors,
        "policy_violation": policy_violations,
    }

    canonical = {
        "events": events_ordered,
        "totals": totals,
    }
    token = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    summary: Dict[str, object] = {
        "events": events_ordered,
        "totals": totals,
        "determinism_token": token,
    }

    threshold_raw = os.getenv("PROOF_PLUGIN_POLICY_MAX", "0")
    try:
        threshold = float(threshold_raw)
    except Exception:
        threshold = 0.0

    gate: Dict[str, object] = {
        "ok": policy_violations <= threshold,
        "policy_violations": policy_violations,
        "total_errors": total_errors,
        "total_events": total,
        "threshold": threshold,
    }
    if total > 0:
        gate["error_rate"] = total_errors / max(total, 1e-9)

    return summary, gate


def _required_fuzz_suites() -> List[str]:
    env_value = os.getenv("PROOF_FUZZ_REQUIRED")
    if env_value:
        try:
            parsed = json.loads(env_value)
            if isinstance(parsed, list) and all(isinstance(item, str) for item in parsed):
                return [item.strip() for item in parsed if item and item.strip()]
        except Exception:
            pass
        return [item.strip() for item in env_value.split(",") if item and item.strip()]
    return ["planner_chaos", "tool_rbac"]


def _collect_fuzz_summaries(
    artifacts_dir: Path,
) -> Optional[Tuple[Dict[str, object], Dict[str, object], Dict[str, Path]]]:
    """Aggregate fuzz and chaos suite outputs for proof attestation."""

    required = _required_fuzz_suites()
    fuzz_dir = artifacts_dir / "proof" / "fuzz"
    suites: Dict[str, Tuple[List[FuzzRecord], Path]] = {}
    file_map: Dict[str, Path] = {}

    if fuzz_dir.exists():
        for path in sorted(fuzz_dir.glob("*.jsonl")):
            suite = path.stem
            try:
                records = read_fuzz_records(path)
            except Exception:
                continue
            if not records:
                continue
            suites[suite] = (records, path)
            file_map[suite] = path

    summary: Dict[str, object] = {"suites": {}}
    gate: Dict[str, object] = {
        "ok": True,
        "required": required,
        "missing": [],
        "suites": {},
        "present": sorted(suites.keys()),
    }

    if not suites:
        if required:
            gate["ok"] = False
            gate["missing"] = sorted(required)
        token = hashlib.sha256(
            json.dumps(
                {
                    "suites": summary["suites"],
                    "required": required,
                    "missing": gate.get("missing"),
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        summary["determinism_token"] = token
        gate["determinism_token"] = token
        return summary, gate, {}

    for suite in sorted(suites.keys()):
        records, path = suites[suite]
        timestamps: List[str] = []
        hashes: List[str] = []
        numeric: Dict[str, Dict[str, float]] = {}

        for record in records:
            timestamps.append(record.timestamp)
            if record.sha256:
                hashes.append(record.sha256)
            for key, value in record.payload.items():
                if isinstance(value, bool):
                    continue
                if isinstance(value, (int, float)):
                    stats = numeric.setdefault(
                        key,
                        {"count": 0.0, "sum": 0.0, "min": math.inf, "max": -math.inf},
                    )
                    stats["count"] += 1.0
                    val = float(value)
                    stats["sum"] += val
                    if val < stats["min"]:
                        stats["min"] = val
                    if val > stats["max"]:
                        stats["max"] = val

        formatted_numeric: Dict[str, Dict[str, Optional[float]]] = {}
        for key, stats in numeric.items():
            count = stats.get("count", 0.0)
            if count <= 0:
                continue
            avg = stats["sum"] / max(count, 1.0)
            min_val = stats["min"] if stats["min"] != math.inf else None
            max_val = stats["max"] if stats["max"] != -math.inf else None
            formatted_numeric[key] = {
                "avg": avg,
                "min": min_val,
                "max": max_val,
            }

        suite_summary: Dict[str, object] = {
            "records": len(records),
            "latest_timestamp": max(timestamps) if timestamps else None,
        }
        if formatted_numeric:
            suite_summary["numeric"] = formatted_numeric
        if hashes:
            suite_summary["sha256"] = sorted(set(hashes))
        summary["suites"][suite] = suite_summary

        suite_gate: Dict[str, object] = {
            "ok": True,
            "records": len(records),
        }

        if suite in required and len(records) == 0:
            suite_gate["ok"] = False

        if suite == "tool_rbac":
            denied_total = sum(int(record.payload.get("denied") or 0) for record in records)
            attempts_total = sum(int(record.payload.get("attempts") or 0) for record in records)
            suite_gate["denied"] = denied_total
            suite_gate["attempts"] = attempts_total
            if denied_total <= 0:
                suite_gate["ok"] = False
                suite_gate.setdefault("reason", "no_denials_recorded")

        if suite == "planner_chaos":
            failures_total = sum(int(record.payload.get("world_model_failures") or 0) for record in records)
            suite_gate["world_model_failures"] = failures_total
            if failures_total <= 0:
                suite_gate["ok"] = False
                suite_gate.setdefault("reason", "no_failures_produced")

        if not suite_gate["ok"]:
            gate["ok"] = False

        gate["suites"][suite] = suite_gate

    missing = [suite for suite in required if suite not in suites]
    if missing:
        gate["ok"] = False
        gate["missing"] = sorted(missing)

    token = hashlib.sha256(
        json.dumps(
            {
                "suites": summary["suites"],
                "required": required,
                "missing": gate.get("missing"),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    summary["determinism_token"] = token
    gate["determinism_token"] = token

    return summary, gate, file_map


def _build_semantic_self_model_payload(artifacts_dir: Path) -> Dict[str, object]:
    """Construct the semantic self-model bundle payload."""

    semantic_dir = artifacts_dir / "semantic_self_model"
    graph_path = semantic_dir / "capability_graph.jsonl"
    failure_path = semantic_dir / "failure_analyses.jsonl"

    graph_hash = sha256_file(graph_path) if graph_path.exists() else None
    failure_hash = sha256_file(failure_path) if failure_path.exists() else None
    metrics = _semantic_metrics_snapshot()
    commit = _resolve_git_commit()
    tests_summary = _collect_tests_summary(artifacts_dir)

    payload: Dict[str, object] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "capability_graph_hash": graph_hash,
        "failure_summary_hash": failure_hash,
        "metrics_snapshot": metrics,
        "git_commit": commit,
        "tests_ran": tests_summary,
    }
    if graph_path.exists():
        payload["capability_graph_path"] = str(graph_path)
    if failure_path.exists():
        payload["failure_summary_path"] = str(failure_path)

    canonical = json.dumps(
        {
            "capability_graph_hash": graph_hash or "",
            "failure_summary_hash": failure_hash or "",
            "metrics_snapshot": metrics,
            "git_commit": commit or "",
            "tests_ran": tests_summary or {},
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    payload["determinism_token"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload


def _resolve_artifact_path(artifacts_root: Path, record_path: str) -> Optional[Path]:
    try:
        candidate = Path(record_path)
    except Exception:
        return None
    if candidate.is_absolute():
        return candidate
    parts = candidate.parts
    if parts and parts[0] == "artifacts":
        remainder = Path(*parts[1:]) if len(parts) > 1 else Path()
        return (artifacts_root / remainder)
    return artifacts_root / candidate


def _collect_bootstrap_artifacts(proof: Dict[str, object], artifacts: Path, out_dir: Path) -> None:
    base_dir = artifacts / "world_model" / "bootstrap"
    if not base_dir.exists() or not base_dir.is_dir():
        return

    index_path = base_dir / "index.jsonl"
    index_stats: Dict[str, object] = {}
    all_records: List[Dict[str, object]] = []
    if index_path.exists():
        try:
            meta = _add_file(proof, artifacts, index_path)
            index_stats = {
                "path": meta.get("path"),
                "sha256": meta.get("sha256"),
                "size": meta.get("size"),
            }
        except Exception:
            index_stats = {"path": str(index_path)}
        try:
            raw_lines = [ln.strip() for ln in index_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
            for line in raw_lines:
                try:
                    record = json.loads(line)
                    if isinstance(record, dict):
                        all_records.append(record)
                except Exception:
                    continue
            index_stats.setdefault("entries", len(raw_lines))
            if all_records:
                totals = sum(int(rec.get("count", 0) or 0) for rec in all_records)
                index_stats.setdefault("batches", len(all_records))
                index_stats.setdefault("hypotheses_total", totals)
                latest = all_records[-1]
                index_stats.setdefault("latest_timestamp", latest.get("timestamp"))
                index_stats.setdefault("latest_prompt_hash", latest.get("prompt_hash"))
        except Exception:
            pass

    latest_records = all_records[-5:] if all_records else []
    latest_batches: List[Dict[str, object]] = []

    for record in latest_records:
        path_field = record.get("path")
        if not isinstance(path_field, str):
            continue
        batch_dir = _resolve_artifact_path(artifacts, path_field)
        if batch_dir is None or not batch_dir.exists():
            continue
        files_payload: Dict[str, object] = {}
        for filename in ("prompt.txt", "response.json", "hypotheses.jsonl", "metadata.json", "snapshot.json"):
            file_path = batch_dir / filename
            if not file_path.exists():
                continue
            try:
                meta = _add_file(proof, artifacts, file_path)
                files_payload[filename] = {
                    "path": meta.get("path"),
                    "sha256": meta.get("sha256"),
                    "size": meta.get("size"),
                }
            except Exception:
                files_payload[filename] = {"path": str(file_path)}

        validation_info: Dict[str, object] = {}
        validation_dir = batch_dir / "validation"
        if validation_dir.exists() and validation_dir.is_dir():
            summary_paths: List[str] = []
            for summary_file in sorted(validation_dir.glob("*_summary.json")):
                try:
                    meta = _add_file(proof, artifacts, summary_file)
                    summary_paths.append(str(meta.get("path")))
                except Exception:
                    continue

            log_count = 0
            sample_logs: List[str] = []
            for log_file in sorted(validation_dir.glob("**/*.json")):
                if log_file.name.endswith("_summary.json"):
                    continue
                log_count += 1
                try:
                    meta = _add_file(proof, artifacts, log_file)
                    if len(sample_logs) < 5:
                        sample_logs.append(str(meta.get("path")))
                except Exception:
                    continue
            if summary_paths or log_count:
                validation_info["summaries"] = summary_paths
                validation_info["log_count"] = log_count
                if sample_logs:
                    validation_info["sample_logs"] = sample_logs

        record_payload = {
            "timestamp": record.get("timestamp"),
            "prompt_hash": record.get("prompt_hash"),
            "response_hash": record.get("response_hash"),
            "count": record.get("count"),
            "domain_fingerprint": record.get("domain_fingerprint"),
            "path": record.get("path"),
            "ensemble_hypotheses": record.get("ensemble_hypotheses"),
        }

        entry: Dict[str, object] = {
            "record": record_payload,
            "directory": _artifact_dir_key(batch_dir, artifacts),
            "files": files_payload,
        }
        if validation_info:
            entry["validation"] = validation_info
        latest_batches.append(entry)

    if not index_stats and not latest_batches:
        return

    canonical = {
        "index_sha256": index_stats.get("sha256"),
        "index_entries": index_stats.get("entries"),
        "latest": [
            {
                "timestamp": item["record"].get("timestamp"),
                "prompt_hash": item["record"].get("prompt_hash"),
                "response_hash": item["record"].get("response_hash"),
                "count": item["record"].get("count"),
                "ensemble_hypotheses": item["record"].get("ensemble_hypotheses"),
                "directory": item.get("directory"),
                "validation_log_count": (item.get("validation") or {}).get("log_count"),
            }
            for item in latest_batches
        ],
    }
    determinism_token = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    queue_summary: Dict[str, object] = {}
    queue_path = base_dir / "refinement_queue.jsonl"
    if queue_path.exists():
        try:
            meta = _add_file(proof, artifacts, queue_path)
            lines = [ln.strip() for ln in queue_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
            queue_summary = {
                "path": meta.get("path"),
                "entries": len(lines),
            }
            if lines:
                try:
                    latest_entry = json.loads(lines[-1])
                    queue_summary["latest"] = latest_entry
                    if isinstance(latest_entry, dict) and latest_entry.get("ensemble_hypotheses"):
                        queue_summary["latest_ensemble_hypotheses"] = latest_entry.get("ensemble_hypotheses")
                except Exception:
                    queue_summary["latest"] = lines[-1]
        except Exception:
            queue_summary = {"path": str(queue_path)}

    summary_payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "registry_base_dir": _artifact_dir_key(base_dir, artifacts),
        "index": index_stats,
        "latest_batches": latest_batches,
        "determinism_token": determinism_token,
    }
    if queue_summary:
        summary_payload["refinement_queue"] = queue_summary

    retrain_summary: Dict[str, object] = {}
    retrain_path = base_dir / "retrain_queue.jsonl"
    if retrain_path.exists():
        try:
            meta = _add_file(proof, artifacts, retrain_path)
            lines = [ln.strip() for ln in retrain_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
            retrain_summary = {
                "path": meta.get("path"),
                "entries": len(lines),
            }
            if lines:
                try:
                    retrain_summary["latest"] = json.loads(lines[-1])
                except Exception:
                    retrain_summary["latest"] = lines[-1]
        except Exception:
            retrain_summary = {"path": str(retrain_path)}
    if retrain_summary:
        summary_payload["retrain_queue"] = retrain_summary

    summary_name = f"world_model_bootstrap_{out_dir.name}.json"
    summary_path = artifacts / "proof" / summary_name
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_text = json.dumps(summary_payload, indent=2, sort_keys=True)
    summary_path.write_text(summary_text, encoding="utf-8")
    summary_meta = _add_file(proof, artifacts, summary_path)

    signature_path = summary_path.with_suffix(".sig")
    signature_value = _maybe_write_signature(summary_text, signature_path=signature_path)
    if signature_value:
        try:
            _add_file(proof, artifacts, signature_path)
        except Exception:
            pass

    proof_summary: Dict[str, object] = {
        "artifact": str(summary_path.resolve()),
        "artifact_meta": summary_meta,
        "registry_base_dir": summary_payload["registry_base_dir"],
        "index": index_stats,
        "latest_batches": latest_batches,
        "determinism_token": determinism_token,
    }
    if queue_summary:
        proof_summary["refinement_queue"] = queue_summary
    if retrain_summary:
        proof_summary["retrain_queue"] = retrain_summary
    if signature_value:
        proof_summary["signature"] = signature_value
        proof_summary["signature_path"] = str(signature_path.resolve())

    proof["world_model_bootstrap"] = proof_summary


def _collect_retrain_validation_artifacts(proof: Dict[str, object], artifacts: Path) -> None:
    """Attach world-model retrain validation proofs to the bundle."""

    proof_dir = artifacts / "proof" / "retrain_validation"
    if not proof_dir.exists() or not proof_dir.is_dir():
        return

    records: List[Dict[str, object]] = []
    for path in sorted(proof_dir.glob("retrain_validation_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                continue
            meta = _add_file(proof, artifacts, path)
            record = {
                "timestamp": payload.get("timestamp"),
                "workspace": payload.get("workspace"),
                "ok": bool((payload.get("validation") or {}).get("ok")),
                "path": meta.get("path"),
                "sha256": meta.get("sha256"),
                "size": meta.get("size"),
                "stats_digest": payload.get("stats_digest"),
                "validation_digest": payload.get("validation_digest"),
            }
            records.append(record)
        except Exception:
            continue

    if not records:
        return

    latest_summary_info: Optional[Dict[str, Any]] = None
    latest_summary_path = proof_dir / "latest.json"
    if latest_summary_path.exists() and latest_summary_path.is_file():
        try:
            latest_payload = json.loads(latest_summary_path.read_text(encoding="utf-8"))
            if isinstance(latest_payload, dict):
                summary_meta = _add_file(proof, artifacts, latest_summary_path)
                latest_summary_info = {
                    "path": summary_meta.get("path"),
                    "sha256": summary_meta.get("sha256"),
                    "size": summary_meta.get("size"),
                    "workspace": latest_payload.get("workspace"),
                    "timestamp": latest_payload.get("timestamp"),
                    "ok": bool(latest_payload.get("ok")),
                    "reason_codes": list(latest_payload.get("reason_codes") or []),
                    "proof_path": latest_payload.get("proof_path"),
                }
        except Exception:
            latest_summary_info = None

    canonical = json.dumps(
        [
            {
                "timestamp": rec.get("timestamp"),
                "workspace": rec.get("workspace"),
                "ok": rec.get("ok"),
                "stats_digest": rec.get("stats_digest"),
                "validation_digest": rec.get("validation_digest"),
            }
            for rec in records
        ],
        sort_keys=True,
        separators=(",", ":"),
    )
    summary: Dict[str, object] = {
        "entries": records,
        "latest": records[-1],
        "counts": {
            "total": len(records),
            "ok": sum(1 for rec in records if rec.get("ok")),
            "failures": sum(1 for rec in records if not rec.get("ok")),
        },
        "determinism_token": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }
    if latest_summary_info:
        summary["latest_summary"] = latest_summary_info
    proof["world_model_retrain_validation"] = summary


def _collect_world_model_validation_artifacts(proof: Dict[str, object], artifacts: Path) -> None:
    """Attach world-model validation proofs to the bundle."""

    proof_dir = artifacts / "proof" / "world_model_validation"
    if not proof_dir.exists() or not proof_dir.is_dir():
        return

    records: List[Dict[str, object]] = []
    for path in sorted(proof_dir.glob("world_model_validation_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                continue
            meta = _add_file(proof, artifacts, path)
            records.append(
                {
                    "timestamp": payload.get("timestamp"),
                    "workspace": payload.get("workspace"),
                    "ok": bool((payload.get("validation") or {}).get("ok")),
                    "path": meta.get("path"),
                    "sha256": meta.get("sha256"),
                    "size": meta.get("size"),
                    "stats_digest": payload.get("stats_digest"),
                    "validation_digest": payload.get("validation_digest"),
                    "summary_digest": payload.get("summary_digest"),
                }
            )
        except Exception:
            continue

    if not records:
        return

    canonical = json.dumps(
        [
            {
                "timestamp": rec.get("timestamp"),
                "workspace": rec.get("workspace"),
                "ok": rec.get("ok"),
                "summary_digest": rec.get("summary_digest"),
            }
            for rec in records
        ],
        sort_keys=True,
        separators=(",", ":"),
    )
    summary: Dict[str, object] = {
        "entries": records,
        "latest": records[-1],
        "counts": {
            "total": len(records),
            "ok": sum(1 for rec in records if rec.get("ok")),
            "failures": sum(1 for rec in records if not rec.get("ok")),
        },
        "determinism_token": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }
    proof["world_model_validation"] = summary


def _collect_resource_world_model_validation_artifacts(proof: Dict[str, object], artifacts: Path) -> None:
    """Attach resource-domain world-model validation proofs to the bundle."""

    proof_dir = artifacts / "proof" / "resource_world_model_validation"
    if not proof_dir.exists() or not proof_dir.is_dir():
        return

    records: List[Dict[str, object]] = []
    for path in sorted(proof_dir.glob("resource_world_model_validation_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                continue
            meta = _add_file(proof, artifacts, path)
            records.append(
                {
                    "timestamp": payload.get("timestamp"),
                    "workspace": payload.get("workspace"),
                    "ok": bool((payload.get("validation") or {}).get("ok")),
                    "path": meta.get("path"),
                    "sha256": meta.get("sha256"),
                    "size": meta.get("size"),
                    "stats_digest": payload.get("stats_digest"),
                    "validation_digest": payload.get("validation_digest"),
                    "summary_digest": payload.get("summary_digest"),
                }
            )
        except Exception:
            continue

    if not records:
        return

    canonical = json.dumps(
        [
            {
                "timestamp": rec.get("timestamp"),
                "workspace": rec.get("workspace"),
                "ok": rec.get("ok"),
                "summary_digest": rec.get("summary_digest"),
            }
            for rec in records
        ],
        sort_keys=True,
        separators=(",", ":"),
    )
    summary: Dict[str, object] = {
        "entries": records,
        "latest": records[-1],
        "counts": {
            "total": len(records),
            "ok": sum(1 for rec in records if rec.get("ok")),
            "failures": sum(1 for rec in records if not rec.get("ok")),
        },
        "determinism_token": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }
    proof["resource_world_model_validation"] = summary


def _collect_coding_gate_artifacts(proof: Dict[str, object], artifacts: Path) -> None:
    coder_dir = artifacts / "coder"
    scorecard_path = coder_dir / "scorecard.json"
    metrics_path = coder_dir / "metrics.json"
    if not scorecard_path.exists() and not metrics_path.exists():
        return

    summary: Dict[str, Any] = {}
    if scorecard_path.exists():
        try:
            payload = json.loads(scorecard_path.read_text(encoding="utf-8"))
            summary["scorecard"] = payload
            meta = _add_file(proof, artifacts, scorecard_path)
            summary["scorecard_path"] = meta.get("path")
        except Exception:
            summary.setdefault("errors", []).append("scorecard_parse_failed")

    if metrics_path.exists():
        try:
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            summary["metrics"] = payload
            meta = _add_file(proof, artifacts, metrics_path)
            summary["metrics_path"] = meta.get("path")
        except Exception:
            summary.setdefault("errors", []).append("metrics_parse_failed")

    if summary:
        proof["coding_gate"] = summary


def _collect_policy_validation_artifacts(proof: Dict[str, object], artifacts: Path) -> None:
    """Attach policy validation proofs to the bundle."""

    proof_dir = artifacts / "proof" / "policy_validation"
    if not proof_dir.exists() or not proof_dir.is_dir():
        return

    records: List[Dict[str, object]] = []
    for path in sorted(proof_dir.glob("policy_validation_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                continue
            meta = _add_file(proof, artifacts, path)
            records.append(
                {
                    "timestamp": payload.get("timestamp"),
                    "workspace": payload.get("workspace"),
                    "ok": bool((payload.get("validation") or {}).get("ok")),
                    "path": meta.get("path"),
                    "sha256": meta.get("sha256"),
                    "size": meta.get("size"),
                    "stats_digest": payload.get("stats_digest"),
                    "validation_digest": payload.get("validation_digest"),
                }
            )
        except Exception:
            continue

    if not records:
        return

    canonical = json.dumps(
        [
            {
                "timestamp": rec.get("timestamp"),
                "workspace": rec.get("workspace"),
                "ok": rec.get("ok"),
                "stats_digest": rec.get("stats_digest"),
                "validation_digest": rec.get("validation_digest"),
            }
            for rec in records
        ],
        sort_keys=True,
        separators=(",", ":"),
    )
    summary: Dict[str, object] = {
        "entries": records,
        "latest": records[-1],
        "counts": {
            "total": len(records),
            "ok": sum(1 for rec in records if rec.get("ok")),
            "failures": sum(1 for rec in records if not rec.get("ok")),
        },
        "determinism_token": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }
    proof["policy_retrain_validation"] = summary


def _load_retrain_validation_history(artifacts: Path, *, limit: int = 10) -> List[Dict[str, Any]]:
    """Load recent retrain validation proofs (most recent last)."""

    dirs: List[Path] = []
    primary = artifacts / "proof" / "retrain_validation"
    if primary.exists() and primary.is_dir():
        # When a primary artifacts dir is explicitly provided (tests set ARTIFACTS_DIR),
        # avoid merging in default ./artifacts to prevent cross-suite contamination.
        dirs.append(primary)
    else:
        default_artifacts = Path("artifacts") / "proof" / "retrain_validation"
        if default_artifacts.exists() and default_artifacts.is_dir():
            dirs.append(default_artifacts)

    seen: Dict[str, Path] = {}
    files: List[Path] = []
    for directory in dirs:
        for candidate in sorted(directory.glob("retrain_validation_*.json")):
            resolved: Optional[str]
            try:
                resolved = str(candidate.resolve())
            except Exception:
                resolved = str(candidate)
            if resolved in seen:
                continue
            seen[resolved] = candidate
            files.append(candidate)

    if not files:
        return []

    files.sort()
    if limit > 0:
        files = files[-limit:]

    history: List[Dict[str, Any]] = []
    for path in files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                continue
            record = {
                "path": str(path),
                "timestamp": payload.get("timestamp"),
                "workspace": payload.get("workspace"),
                "stats_summary": payload.get("stats_summary"),
                "validation": payload.get("validation"),
            }
            history.append(record)
        except Exception:
            continue
    return history


def _load_world_model_validation_history(artifacts: Path, *, limit: int = 10) -> List[Dict[str, Any]]:
    """Load recent world-model validation proofs (most recent last)."""

    dirs: List[Path] = []
    primary = artifacts / "proof" / "world_model_validation"
    if primary.exists() and primary.is_dir():
        dirs.append(primary)
    else:
        default_artifacts = Path("artifacts") / "proof" / "world_model_validation"
        if default_artifacts.exists() and default_artifacts.is_dir():
            dirs.append(default_artifacts)

    seen: Dict[str, Path] = {}
    files: List[Path] = []
    for directory in dirs:
        for candidate in sorted(directory.glob("world_model_validation_*.json")):
            try:
                resolved = str(candidate.resolve())
            except Exception:
                resolved = str(candidate)
            if resolved in seen:
                continue
            seen[resolved] = candidate
            files.append(candidate)

    if not files:
        return []

    files.sort()
    if limit > 0:
        files = files[-limit:]

    history: List[Dict[str, Any]] = []
    for path in files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                continue
            history.append(
                {
                    "path": str(path),
                    "timestamp": payload.get("timestamp"),
                    "workspace": payload.get("workspace"),
                    "stats_summary": payload.get("stats_summary"),
                    "validation": payload.get("validation"),
                    "summary": payload.get("summary"),
                }
            )
        except Exception:
            continue
    return history


def _load_resource_world_model_validation_history(artifacts: Path, *, limit: int = 10) -> List[Dict[str, Any]]:
    primary = artifacts / "proof" / "resource_world_model_validation"
    dirs: List[Path] = []
    if primary.exists() and primary.is_dir():
        dirs.append(primary)
    else:
        default_artifacts = Path("artifacts") / "proof" / "resource_world_model_validation"
        if default_artifacts.exists() and default_artifacts.is_dir():
            dirs.append(default_artifacts)

    seen: Dict[str, Path] = {}
    files: List[Path] = []
    for directory in dirs:
        for candidate in sorted(directory.glob("resource_world_model_validation_*.json")):
            try:
                resolved = str(candidate.resolve())
            except Exception:
                resolved = str(candidate)
            if resolved in seen:
                continue
            seen[resolved] = candidate
            files.append(candidate)

    if not files:
        return []

    files.sort()
    if limit > 0:
        files = files[-limit:]

    history: List[Dict[str, Any]] = []
    for path in files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                continue
            history.append(
                {
                    "path": str(path),
                    "timestamp": payload.get("timestamp"),
                    "workspace": payload.get("workspace"),
                    "stats_summary": payload.get("stats_summary"),
                    "validation": payload.get("validation"),
                    "summary": payload.get("summary"),
                }
            )
        except Exception:
            continue
    return history


def _extract_retrain_metrics(record: Mapping[str, Any]) -> Dict[str, Any]:
    """Project the retrain proof into comparable numeric metrics."""

    stats = record.get("stats_summary") if isinstance(record.get("stats_summary"), Mapping) else {}
    validation = record.get("validation") if isinstance(record.get("validation"), Mapping) else {}

    top1 = _safe_float((stats or {}).get("top1_accuracy"))
    avg_reward = _safe_float((stats or {}).get("avg_reward"))

    min_success: Optional[float] = None
    avg_success: Optional[float] = None
    success_rates: List[float] = []
    per_mode = stats.get("per_mode") if isinstance(stats, Mapping) else None
    if isinstance(per_mode, Mapping):
        for value in per_mode.values():
            if not isinstance(value, Mapping):
                continue
            rate = _safe_float(value.get("success_rate"))
            if rate is not None:
                success_rates.append(rate)
    if success_rates:
        min_success = min(success_rates)
        avg_success = sum(success_rates) / len(success_rates)

    failures = validation.get("failures") if isinstance(validation.get("failures"), list) else []

    return {
        "path": record.get("path"),
        "timestamp": record.get("timestamp"),
        "workspace": record.get("workspace"),
        "top1_accuracy": top1,
        "avg_reward": avg_reward,
        "min_success_rate": min_success,
        "avg_success_rate": avg_success,
        "validation_ok": bool(validation.get("ok")),
        "validation_failures": failures[:5] if failures else [],
        "failure_count": len(failures),
    }


def _load_policy_validation_history(artifacts: Path, *, limit: int = 10) -> List[Dict[str, Any]]:
    dirs: List[Path] = []
    primary = artifacts / "proof" / "policy_validation"
    if primary.exists() and primary.is_dir():
        dirs.append(primary)
    default_artifacts = Path("artifacts") / "proof" / "policy_validation"
    try:
        if default_artifacts.exists() and default_artifacts.is_dir():
            if primary.resolve() != default_artifacts.resolve():
                dirs.append(default_artifacts)
    except Exception:
        if default_artifacts.exists() and default_artifacts.is_dir():
            dirs.append(default_artifacts)

    seen: Dict[str, Path] = {}
    files: List[Path] = []
    for directory in dirs:
        for candidate in sorted(directory.glob("policy_validation_*.json")):
            try:
                resolved = str(candidate.resolve())
            except Exception:
                resolved = str(candidate)
            if resolved in seen:
                continue
            seen[resolved] = candidate
            files.append(candidate)

    if not files:
        return []

    files.sort()
    if limit > 0:
        files = files[-limit:]

    history: List[Dict[str, Any]] = []
    for path in files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                continue
            history.append(
                {
                    "path": str(path),
                    "timestamp": payload.get("timestamp"),
                    "workspace": payload.get("workspace"),
                    "stats_summary": payload.get("stats_summary"),
                    "validation": payload.get("validation"),
                    "control_metrics": payload.get("control_metrics"),
                }
            )
        except Exception:
            continue
    return history


def _extract_policy_metrics(record: Mapping[str, Any]) -> Dict[str, Any]:
    stats = record.get("stats_summary") if isinstance(record.get("stats_summary"), Mapping) else {}
    validation = record.get("validation") if isinstance(record.get("validation"), Mapping) else {}

    top1 = _safe_float((stats or {}).get("top1_accuracy"))
    top5 = _safe_float((stats or {}).get("top5_accuracy"))
    avg_latency = _safe_float((stats or {}).get("avg_inference_ms"))
    latency_p95 = _safe_float((stats or {}).get("latency_p95_ms"))
    avg_conf = _safe_float((stats or {}).get("avg_confidence"))
    samples = _safe_float((stats or {}).get("samples"))

    fairness = stats.get("fairness") if isinstance(stats, Mapping) else {}
    fairness_delta = _safe_float((fairness or {}).get("max_delta"))

    failures = validation.get("failures") if isinstance(validation.get("failures"), list) else []

    buckets: Dict[str, Dict[str, Any]] = {}
    bucket_payload = stats.get("bucket_metrics") if isinstance(stats.get("bucket_metrics"), Mapping) else {}
    if isinstance(bucket_payload, Mapping):
        for name, payload in bucket_payload.items():
            if not isinstance(payload, Mapping):
                continue
            buckets[str(name)] = {
                "support": _safe_float(payload.get("support")),
                "top1_accuracy": _safe_float(payload.get("accuracy")),
                "error_rate": _safe_float(payload.get("error_rate")),
            }

    controls: Dict[str, Dict[str, Any]] = {}
    control_payload = record.get("control_metrics") if isinstance(record.get("control_metrics"), Mapping) else {}
    if isinstance(control_payload, Mapping):
        for name, payload in control_payload.items():
            if not isinstance(payload, Mapping):
                continue
            controls[str(name)] = {
                "top1_accuracy": _safe_float(payload.get("top1_accuracy")),
                "avg_inference_ms": _safe_float(payload.get("avg_inference_ms")),
                "samples": _safe_float(payload.get("samples")),
                "fairness_delta": _safe_float(payload.get("fairness_max_delta")),
            }

    return {
        "timestamp": record.get("timestamp"),
        "workspace": record.get("workspace"),
        "top1_accuracy": top1,
        "top5_accuracy": top5,
        "avg_inference_ms": avg_latency,
        "latency_p95_ms": latency_p95,
        "avg_confidence": avg_conf,
        "samples": samples,
        "fairness_delta": fairness_delta,
        "validation_ok": bool(validation.get("ok")),
        "validation_failures": failures[:5] if failures else [],
        "failure_count": len(failures),
        "buckets": buckets,
        "controls": controls,
    }


def _extract_world_model_metrics(record: Mapping[str, Any]) -> Dict[str, Any]:
    stats = record.get("stats_summary") if isinstance(record.get("stats_summary"), Mapping) else {}
    validation = record.get("validation") if isinstance(record.get("validation"), Mapping) else {}
    summary = record.get("summary") if isinstance(record.get("summary"), Mapping) else {}
    calibration = summary.get("calibration") if isinstance(summary.get("calibration"), Mapping) else {}

    match_rate = _safe_float((stats or {}).get("match_rate"))
    avg_abs_reward = _safe_float((stats or {}).get("avg_abs_reward_delta"))
    latency_p95 = _safe_float((stats or {}).get("latency_p95_ms"))
    latency_avg = _safe_float((stats or {}).get("latency_avg_ms"))
    reward_p95 = _safe_float((stats or {}).get("reward_abs_p95"))
    avg_confidence = _safe_float((stats or {}).get("avg_confidence"))
    brier_score = _safe_float((stats or {}).get("brier_score"))
    nll_score = _safe_float((stats or {}).get("negative_log_likelihood"))
    reward_rmse = _safe_float((stats or {}).get("reward_rmse"))
    if calibration:
        avg_confidence = _safe_float(calibration.get("avg_confidence", avg_confidence))
        brier_score = _safe_float(calibration.get("brier_score", brier_score))
        nll_score = _safe_float(calibration.get("negative_log_likelihood", nll_score))
        reward_rmse = _safe_float(calibration.get("reward_rmse", reward_rmse))

    failures = validation.get("failures") if isinstance(validation.get("failures"), list) else []

    return {
        "timestamp": record.get("timestamp"),
        "workspace": record.get("workspace"),
        "match_rate": match_rate,
        "avg_abs_reward_delta": avg_abs_reward,
        "latency_avg_ms": latency_avg,
        "latency_p95_ms": latency_p95,
        "reward_abs_p95": reward_p95,
        "avg_confidence": avg_confidence,
        "brier_score": brier_score,
        "negative_log_likelihood": nll_score,
        "reward_rmse": reward_rmse,
        "validation_ok": bool(validation.get("ok")),
        "validation_failures": failures[:5] if failures else [],
        "summary": summary,
    }


def _world_model_retrain_gate(proof: Dict[str, object], artifacts: Path) -> None:
    """Evaluate world-model retrain validation proofs for regressions."""

    gates = proof.setdefault("gates", {})
    history = _load_retrain_validation_history(artifacts, limit=10)

    min_top1 = _safe_float(os.getenv("PROOF_WM_RETRAIN_MIN_TOP1", "0.7"))
    min_success = _safe_float(os.getenv("PROOF_WM_RETRAIN_MIN_SUCCESS", "0.85"))
    max_top1_drop = _safe_float(os.getenv("PROOF_WM_RETRAIN_MAX_TOP1_DROP", "0.05"))
    max_success_drop = _safe_float(os.getenv("PROOF_WM_RETRAIN_MAX_SUCCESS_DROP", "0.05"))

    gate_payload: Dict[str, Any] = {
        "history_count": len(history),
        "thresholds": {
            "min_top1_accuracy": min_top1,
            "min_success_rate": min_success,
            "max_top1_drop": max_top1_drop,
            "max_success_drop": max_success_drop,
        },
        "history_paths": [record["path"] for record in history],
    }

    if not history:
        gate_payload["ok"] = False
        gate_payload["error"] = "missing_retrain_validation"
        gates["world_model_retrain"] = {
            k: v for k, v in gate_payload.items() if v not in (None, [], {})
        }
        return

    latest_record = history[-1]
    latest_metrics = _extract_retrain_metrics(latest_record)
    gate_payload["latest"] = latest_metrics
    if latest_metrics.get("path"):
        gate_payload["latest_path"] = latest_metrics.get("path")

    violations: List[str] = []

    if not latest_metrics.get("validation_ok", False):
        violations.append("validation_failed")

    top1 = latest_metrics.get("top1_accuracy")
    if min_top1 is not None and top1 is not None and top1 < min_top1:
        violations.append("top1_below_min")

    min_success_rate = latest_metrics.get("min_success_rate")
    if min_success is not None and min_success_rate is not None and min_success_rate < min_success:
        violations.append("success_rate_below_min")

    if len(history) > 1:
        baseline_record = history[-2]
        baseline_metrics = _extract_retrain_metrics(baseline_record)
        gate_payload["baseline"] = baseline_metrics
        if baseline_metrics.get("path"):
            gate_payload["baseline_path"] = baseline_metrics.get("path")

        baseline_top1 = baseline_metrics.get("top1_accuracy")
        if top1 is not None and baseline_top1 is not None:
            drop = baseline_top1 - top1
            gate_payload["top1_drop"] = drop
            if max_top1_drop is not None and drop > max_top1_drop:
                violations.append("top1_drop_exceeded")

        baseline_success = baseline_metrics.get("min_success_rate")
        if min_success_rate is not None and baseline_success is not None:
            success_drop = baseline_success - min_success_rate
            gate_payload["success_rate_drop"] = success_drop
            if max_success_drop is not None and success_drop > max_success_drop:
                violations.append("success_rate_drop_exceeded")

    if violations:
        gate_payload["violations"] = violations
        gate_payload["reason_codes"] = violations
    else:
        gate_payload["reason_codes"] = []

    gate_payload["ok"] = len(violations) == 0

    gates["world_model_retrain"] = {
        k: v for k, v in gate_payload.items() if v not in (None, [], {})
    }


def _world_model_validation_gate(proof: Dict[str, object], artifacts: Path) -> None:
    """Evaluate world-model validation proofs for regressions."""

    gates = proof.setdefault("gates", {})
    history = _load_world_model_validation_history(artifacts, limit=10)

    min_match_rate = _safe_float(os.getenv("PROOF_WM_MIN_MATCH_RATE", "0.9"))
    max_reward_delta = _safe_float(os.getenv("PROOF_WM_MAX_REWARD_DELTA", "0.5"))
    max_latency_p95 = _safe_float(os.getenv("PROOF_WM_MAX_LATENCY_P95_MS", "15.0"))
    max_brier = _safe_float(os.getenv("PROOF_WM_MAX_BRIER", "0.35"))
    max_nll = _safe_float(os.getenv("PROOF_WM_MAX_NLL", "1.0"))
    max_reward_rmse = _safe_float(os.getenv("PROOF_WM_MAX_REWARD_RMSE", "0.8"))

    gate_payload: Dict[str, Any] = {
        "history_count": len(history),
        "thresholds": {
            "min_match_rate": min_match_rate,
            "max_avg_abs_reward_delta": max_reward_delta,
            "max_latency_p95_ms": max_latency_p95,
            "max_brier_score": max_brier,
            "max_negative_log_likelihood": max_nll,
            "max_reward_rmse": max_reward_rmse,
        },
        "history_paths": [record["path"] for record in history],
        "workspace_results": {},
    }

    if not history:
        gate_payload["ok"] = False
        gate_payload["error"] = "missing_world_model_validation"
        gates["world_model_validation"] = {k: v for k, v in gate_payload.items() if v not in (None, [], {})}
        return

    workspace_latest: Dict[str, Dict[str, Any]] = {}
    for record in history:
        metrics = _extract_world_model_metrics(record)
        workspace = str(metrics.get("workspace") or "default")
        workspace_latest[workspace] = metrics

    overall_ok = True
    workspace_results: Dict[str, Any] = {}

    for workspace, metrics in workspace_latest.items():
        violations: List[str] = []

        match_rate = metrics.get("match_rate")
        avg_abs_reward = metrics.get("avg_abs_reward_delta")
        latency_p95 = metrics.get("latency_p95_ms")
        brier_score = metrics.get("brier_score")
        nll_score = metrics.get("negative_log_likelihood")
        reward_rmse = metrics.get("reward_rmse")

        if not metrics.get("validation_ok", False):
            violations.append("validation_failed")

        if min_match_rate is not None and match_rate is not None and match_rate < min_match_rate:
            violations.append("match_rate_below_min")

        if max_reward_delta is not None and avg_abs_reward is not None and avg_abs_reward > max_reward_delta:
            violations.append("reward_delta_above_max")

        if max_latency_p95 is not None and latency_p95 is not None and latency_p95 > max_latency_p95:
            violations.append("latency_p95_above_max")

        if max_brier is not None and brier_score is not None and brier_score > max_brier:
            violations.append("brier_above_max")

        if max_nll is not None and nll_score is not None and nll_score > max_nll:
            violations.append("nll_above_max")

        if max_reward_rmse is not None and reward_rmse is not None and reward_rmse > max_reward_rmse:
            violations.append("reward_rmse_above_max")

        workspace_ok = len(violations) == 0
        overall_ok = overall_ok and workspace_ok

        workspace_results[workspace] = {
            "latest": metrics,
            "violations": violations,
            "ok": workspace_ok,
        }

    gate_payload["workspace_results"] = workspace_results
    gate_payload["ok"] = overall_ok

    gates["world_model_validation"] = {k: v for k, v in gate_payload.items() if v not in (None, [], {})}


def _resource_world_model_validation_gate(proof: Dict[str, object], artifacts: Path) -> None:
    gates = proof.setdefault("gates", {})
    history = _load_resource_world_model_validation_history(artifacts, limit=10)

    min_match_rate = _safe_float(os.getenv("PROOF_RESOURCE_WM_MIN_MATCH_RATE", "0.7"))
    max_reward_delta = _safe_float(os.getenv("PROOF_RESOURCE_WM_MAX_REWARD_DELTA", "0.6"))
    max_latency_p95 = _safe_float(os.getenv("PROOF_RESOURCE_WM_MAX_LATENCY_P95_MS", "25.0"))
    max_brier = _safe_float(os.getenv("PROOF_RESOURCE_WM_MAX_BRIER", "0.35"))
    max_nll = _safe_float(os.getenv("PROOF_RESOURCE_WM_MAX_NLL", "1.0"))
    max_rmse = _safe_float(os.getenv("PROOF_RESOURCE_WM_MAX_REWARD_RMSE", "0.8"))

    gate_payload: Dict[str, Any] = {
        "history_count": len(history),
        "thresholds": {
            "min_match_rate": min_match_rate,
            "max_avg_abs_reward_delta": max_reward_delta,
            "max_latency_p95_ms": max_latency_p95,
            "max_brier_score": max_brier,
            "max_negative_log_likelihood": max_nll,
            "max_reward_rmse": max_rmse,
        },
        "history_paths": [record["path"] for record in history],
        "workspace_results": {},
    }

    if not history:
        gate_payload["ok"] = False
        gate_payload["error"] = "missing_resource_world_model_validation"
        gates["resource_world_model_validation"] = {k: v for k, v in gate_payload.items() if v not in (None, [], {})}
        return

    latest_by_workspace: Dict[str, Dict[str, Any]] = {}
    for record in history:
        metrics = _extract_world_model_metrics(record)
        workspace = str(metrics.get("workspace") or "resource")
        latest_by_workspace[workspace] = metrics

    overall_ok = True
    workspace_results: Dict[str, Any] = {}

    for workspace, metrics in latest_by_workspace.items():
        violations: List[str] = []
        if not metrics.get("validation_ok", False):
            violations.append("validation_failed")

        match_rate = metrics.get("match_rate")
        avg_abs_reward = metrics.get("avg_abs_reward_delta")
        latency_p95 = metrics.get("latency_p95_ms")
        brier_score = metrics.get("brier_score")
        nll_score = metrics.get("negative_log_likelihood")
        reward_rmse = metrics.get("reward_rmse")

        if min_match_rate is not None and match_rate is not None and match_rate < min_match_rate:
            violations.append("match_rate_below_min")
        if max_reward_delta is not None and avg_abs_reward is not None and avg_abs_reward > max_reward_delta:
            violations.append("reward_delta_above_max")
        if max_latency_p95 is not None and latency_p95 is not None and latency_p95 > max_latency_p95:
            violations.append("latency_p95_above_max")
        if max_brier is not None and brier_score is not None and brier_score > max_brier:
            violations.append("brier_above_max")
        if max_nll is not None and nll_score is not None and nll_score > max_nll:
            violations.append("nll_above_max")
        if max_rmse is not None and reward_rmse is not None and reward_rmse > max_rmse:
            violations.append("reward_rmse_above_max")

        ok = len(violations) == 0
        overall_ok = overall_ok and ok
        workspace_results[workspace] = {
            "latest": metrics,
            "violations": violations,
            "ok": ok,
        }

    gate_payload["workspace_results"] = workspace_results
    gate_payload["ok"] = overall_ok

    gates["resource_world_model_validation"] = {k: v for k, v in gate_payload.items() if v not in (None, [], {})}


def _coding_gate(proof: Dict[str, object], artifacts: Path) -> None:
    gates = proof.setdefault("gates", {})
    coder_dir = artifacts / "coder"
    scorecard_path = coder_dir / "scorecard.json"
    metrics_path = coder_dir / "metrics.json"

    require_ok = os.getenv("PROOF_CODING_REQUIRED", "1").strip().lower() not in {"0", "false", "no"}

    gate_payload: Dict[str, Any] = {
        "scorecard_path": str(scorecard_path) if scorecard_path.exists() else None,
        "metrics_path": str(metrics_path) if metrics_path.exists() else None,
        "thresholds": {
            "require_ok": require_ok,
        },
    }

    if not scorecard_path.exists():
        gate_payload["ok"] = False
        gate_payload["violations"] = ["scorecard_missing"]
        gates["coding"] = {k: v for k, v in gate_payload.items() if v not in (None, [], {})}
        return

    try:
        scorecard = json.loads(scorecard_path.read_text(encoding="utf-8"))
    except Exception as exc:
        gate_payload["ok"] = False
        gate_payload["violations"] = ["scorecard_parse_failed"]
        gate_payload["error"] = str(exc)
        gates["coding"] = {k: v for k, v in gate_payload.items() if v not in (None, [], {})}
        return

    ok = bool(scorecard.get("ok"))
    violations: List[str] = []
    if require_ok and not ok:
        violations.append("pytest_failed")

    gate_payload.update(
        {
            "scorecard": scorecard,
            "ok": len(violations) == 0,
            "violations": violations,
        }
    )
    gates["coding"] = {k: v for k, v in gate_payload.items() if v not in (None, [], {})}


def _policy_retrain_gate(proof: Dict[str, object], artifacts: Path) -> None:
    gates = proof.setdefault("gates", {})
    history = _load_policy_validation_history(artifacts, limit=10)

    min_top1 = _safe_float(os.getenv("PROOF_POLICY_MIN_TOP1", "0.75"))
    max_top1_drop = _safe_float(os.getenv("PROOF_POLICY_MAX_TOP1_DROP", "0.03"))
    max_latency = _safe_float(os.getenv("PROOF_POLICY_MAX_LATENCY_MS", "5.0"))
    max_latency_p95 = _safe_float(os.getenv("PROOF_POLICY_MAX_LATENCY_P95_MS", "5.0"))
    max_fairness_delta = _safe_float(os.getenv("PROOF_POLICY_MAX_FAIRNESS_DELTA", "0.25"))
    min_samples = _safe_float(os.getenv("PROOF_POLICY_MIN_SAMPLES", "50")) or 0.0
    min_bucket_samples = _safe_float(os.getenv("PROOF_POLICY_MIN_BUCKET_SAMPLES", "30")) or 0.0
    min_clean_top1 = _safe_float(os.getenv("PROOF_POLICY_MIN_CLEAN_TOP1", "0.95"))
    min_jitter_top1 = _safe_float(os.getenv("PROOF_POLICY_MIN_JITTER_TOP1", "0.90"))
    max_distractor_fp = _safe_float(os.getenv("PROOF_POLICY_MAX_DISTRACTOR_FP", "0.10"))
    max_control_shuffled = _safe_float(os.getenv("PROOF_POLICY_MAX_CONTROL_SHUFFLED_TOP1", "0.45"))

    gate_payload: Dict[str, Any] = {
        "history_count": len(history),
        "thresholds": {
            "min_top1_accuracy": min_top1,
            "max_top1_drop": max_top1_drop,
            "max_latency_ms": max_latency,
            "max_latency_p95_ms": max_latency_p95,
            "max_fairness_delta": max_fairness_delta,
            "min_samples": min_samples,
            "min_bucket_samples": min_bucket_samples,
            "min_clean_top1": min_clean_top1,
            "min_jitter_top1": min_jitter_top1,
            "max_distractor_fp": max_distractor_fp,
            "max_control_shuffled_top1": max_control_shuffled,
        },
        "history_paths": [record["path"] for record in history],
    }

    if not history:
        gate_payload["ok"] = False
        gate_payload["error"] = "missing_policy_validation"
        gates["policy_retrain"] = {k: v for k, v in gate_payload.items() if v not in (None, [], {})}
        return

    latest_record = history[-1]
    latest_metrics = _extract_policy_metrics(latest_record)
    gate_payload["latest"] = latest_metrics

    violations: List[str] = []

    if not latest_metrics.get("validation_ok", False):
        violations.append("validation_failed")

    top1 = latest_metrics.get("top1_accuracy")
    if min_top1 is not None and top1 is not None and top1 < min_top1:
        violations.append("top1_below_min")

    latency = latest_metrics.get("avg_inference_ms")
    if max_latency is not None and latency is not None and latency > max_latency:
        violations.append("latency_above_max")

    latency_p95 = latest_metrics.get("latency_p95_ms")
    if max_latency_p95 is not None and latency_p95 is not None and latency_p95 > max_latency_p95:
        violations.append("latency_p95_above_max")

    fairness_delta = latest_metrics.get("fairness_delta")
    if max_fairness_delta is not None and fairness_delta is not None and fairness_delta > max_fairness_delta:
        violations.append("fairness_delta_exceeded")

    samples = latest_metrics.get("samples")
    if min_samples and samples is not None and samples < min_samples:
        violations.append("low_sample_count")

    buckets = latest_metrics.get("buckets") if isinstance(latest_metrics.get("buckets"), Mapping) else {}
    clean_bucket = buckets.get("clean") if isinstance(buckets, Mapping) else None
    jitter_bucket = buckets.get("jitter") if isinstance(buckets, Mapping) else None
    distractor_bucket = buckets.get("distractor") if isinstance(buckets, Mapping) else None

    def _bucket_support(bucket_payload: Mapping[str, Any] | None) -> Optional[float]:
        if not isinstance(bucket_payload, Mapping):
            return None
        return _safe_float(bucket_payload.get("support"))

    def _bucket_accuracy(bucket_payload: Mapping[str, Any] | None) -> Optional[float]:
        if not isinstance(bucket_payload, Mapping):
            return None
        return _safe_float(bucket_payload.get("top1_accuracy"))

    if not clean_bucket:
        violations.append("missing_clean_bucket")
    else:
        support = _bucket_support(clean_bucket)
        if min_bucket_samples and support is not None and support < min_bucket_samples:
            violations.append("clean_bucket_insufficient_samples")
        accuracy = _bucket_accuracy(clean_bucket)
        if min_clean_top1 is not None and accuracy is not None and accuracy < min_clean_top1:
            violations.append("clean_accuracy_below_min")

    if not jitter_bucket:
        violations.append("missing_jitter_bucket")
    else:
        support = _bucket_support(jitter_bucket)
        if min_bucket_samples and support is not None and support < min_bucket_samples:
            violations.append("jitter_bucket_insufficient_samples")
        accuracy = _bucket_accuracy(jitter_bucket)
        if min_jitter_top1 is not None and accuracy is not None and accuracy < min_jitter_top1:
            violations.append("jitter_accuracy_below_min")

    if not distractor_bucket:
        violations.append("missing_distractor_bucket")
    else:
        support = _bucket_support(distractor_bucket)
        if min_bucket_samples and support is not None and support < min_bucket_samples:
            violations.append("distractor_bucket_insufficient_samples")
        error_rate = _safe_float(distractor_bucket.get("error_rate"))
        accuracy = _safe_float(distractor_bucket.get("top1_accuracy"))
        fp_rate = None
        if error_rate is not None:
            fp_rate = error_rate
        elif accuracy is not None:
            fp_rate = 1.0 - accuracy
        if max_distractor_fp is not None and fp_rate is not None and fp_rate > max_distractor_fp:
            violations.append("distractor_fp_above_max")

    controls = latest_metrics.get("controls") if isinstance(latest_metrics.get("controls"), Mapping) else {}
    shuffled_control = controls.get("shuffled") if isinstance(controls, Mapping) else None
    if shuffled_control:
        shuffled_acc = _safe_float(shuffled_control.get("top1_accuracy"))
        if max_control_shuffled is not None and shuffled_acc is not None and shuffled_acc > max_control_shuffled:
            violations.append("control_shuffled_accuracy_too_high")
    else:
        violations.append("missing_control_shuffled")

    if len(history) > 1:
        baseline_record = history[-2]
        baseline_metrics = _extract_policy_metrics(baseline_record)
        gate_payload["baseline"] = baseline_metrics

        baseline_top1 = baseline_metrics.get("top1_accuracy")
        if top1 is not None and baseline_top1 is not None:
            drop = baseline_top1 - top1
            gate_payload["top1_drop"] = drop
            if max_top1_drop is not None and drop > max_top1_drop:
                violations.append("top1_drop_exceeded")

    if violations:
        gate_payload["violations"] = violations

    gate_payload["ok"] = len(violations) == 0

    gates["policy_retrain"] = {k: v for k, v in gate_payload.items() if v not in (None, [], {})}


def _read_json_lines(path: Path, *, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    if not path.exists() or not path.is_file():
        return []
    try:
        with path.open("r", encoding="utf-8") as handle:
            if limit is not None and limit > 0:
                buffer = deque(maxlen=limit)
                for line in handle:
                    stripped = line.strip()
                    if stripped:
                        buffer.append(stripped)
                lines = list(buffer)
            else:
                lines = [ln.strip() for ln in handle if ln.strip()]
    except Exception:
        return []

    records: List[Dict[str, Any]] = []
    for entry in lines:
        try:
            payload = json.loads(entry)
        except Exception:
            continue
        if isinstance(payload, dict):
            records.append(payload)
    return records


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except Exception:
        return None


def _safe_int_from_env(name: str, default: int = 0) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(float(raw))
    except Exception:
        return default



def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    except Exception:
        return []
    records: List[Dict[str, Any]] = []
    for line in lines:
        try:
            records.append(json.loads(line))
        except Exception:
            continue
    return records


def _collect_curriculum_sandbox_artifacts(proof: Dict[str, object], artifacts: Path) -> None:
    """Aggregate curriculum sandbox deterministic replay summaries and derive gating info."""

    sandbox_root = artifacts / "sandbox"
    gates = proof.setdefault("gates", {})
    gate_payload: Dict[str, Any] = {
        "ok": False,
        "total_suites": 0,
        "path": str(sandbox_root),
    }
    try:
        gate_payload["path"] = str(sandbox_root.resolve())
    except Exception:
        pass

    min_success_env = os.getenv("PROOF_SANDBOX_MIN_SUCCESS")
    if min_success_env is None:
        min_success_env = os.getenv("BRAIN_SANDBOX_WORLD_MODEL_MIN_SUCCESS", "0.7")
    min_success = _safe_float(min_success_env)
    max_age_hours = _safe_float(os.getenv("PROOF_SANDBOX_MAX_AGE_HOURS", "72"))
    max_failure_ratio = _safe_float(os.getenv("PROOF_SANDBOX_MAX_FAILURE_RATIO", "0.5"))
    if min_success is not None:
        gate_payload["min_success_rate"] = min_success
    if max_age_hours is not None:
        gate_payload["max_age_hours"] = max_age_hours
    if max_failure_ratio is not None:
        gate_payload["max_failure_ratio"] = max_failure_ratio

    if not sandbox_root.exists() or not sandbox_root.is_dir():
        # Even when artifacts missing, attach an empty section so tests expecting
        # the key do not fail and operators can see absence explicitly.
        empty_token = hashlib.sha256(b"[]").hexdigest()
        proof["curriculum_sandboxes"] = {
            "suites": [],
            "counts": {"total_suites": 0, "total_runs": 0},
            "determinism_token": empty_token,
        }
        gate_payload.update({
            "error": "missing_sandbox_artifacts",
            "determinism_token": empty_token,
        })
        gates["curriculum_sandbox"] = {k: v for k, v in gate_payload.items() if v not in (None, [], {})}
        return

    # Collect suites
    suites: List[Dict[str, Any]] = []
    canonical_entries: List[Dict[str, Any]] = []
    for suite_dir in sorted(p for p in sandbox_root.iterdir() if p.is_dir()):
        run_files = sorted(suite_dir.glob("run_*.json"))
        history_path = suite_dir / "history.jsonl"
        audit_path = suite_dir / "audit.jsonl"
        if not run_files and not history_path.exists():
            continue

        recent_runs: List[Dict[str, Any]] = []
        for run_path in run_files[-5:]:
            try:
                payload = json.loads(run_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            try:
                meta = _add_file(proof, artifacts, run_path)
            except Exception:
                meta = {"path": str(run_path)}
            success_flags = 0
            scenarios = []
            for scenario in payload.get("results", []) or []:
                if not isinstance(scenario, dict):
                    continue
                s_ok = bool(scenario.get("success"))
                if s_ok:
                    success_flags += 1
                scenarios.append({
                    "name": scenario.get("name"),
                    "success": s_ok,
                    "seed": scenario.get("seed"),
                })
            run_summary: Dict[str, Any] = {
                "timestamp": payload.get("timestamp"),
                "workspace": payload.get("workspace"),
                "success_rate": payload.get("success_rate"),
                "determinism_token": payload.get("determinism_token"),
                "summary_path": meta.get("path"),
                "scenarios_total": len(scenarios),
                "scenarios_success": success_flags,
                "scenarios_failure": len(scenarios) - success_flags,
            }
            if scenarios:
                run_summary["scenarios"] = scenarios
            recent_runs.append(run_summary)

        history_records = _read_json_lines(history_path, limit=50)
        audit_records = _read_json_lines(audit_path, limit=20)

        suite_info: Dict[str, Any] = {
            "suite": suite_dir.name,
            "runs_total": len(run_files),
            "recent_runs": recent_runs,
            "history": {"path": str(history_path.resolve()), "entries": history_records[-5:] if history_records else []},
            # Include a compact audit tail for operator visibility (last few entries)
            "audit": {
                "path": str(audit_path.resolve()),
                "entries": audit_records[-5:] if audit_records else [],
            },
        }
        latest = recent_runs[-1] if recent_runs else None
        if latest:
            suite_info["latest"] = latest
            suite_info["determinism_token"] = latest.get("determinism_token")
        suites.append(suite_info)
        canonical_entries.append({
            "suite": suite_dir.name,
            "runs_total": len(run_files),
            "latest_token": (latest or {}).get("determinism_token"),
            "latest_success_rate": (latest or {}).get("success_rate"),
        })

    determinism_token = hashlib.sha256(json.dumps(canonical_entries, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()

    counts_summary = {
        "total_suites": len(suites),
        "total_runs": sum(info.get("runs_total", 0) for info in suites),
    }

    proof["curriculum_sandboxes"] = {
        "suites": suites,
        "counts": counts_summary,
        "determinism_token": determinism_token,
    }

    # Gate derivation
    now = datetime.now(UTC)
    stale_suites: List[Dict[str, Any]] = []
    below_success: List[Dict[str, Any]] = []
    failure_ratio_exceeded: List[Dict[str, Any]] = []
    for info in suites:
        suite_name = str(info.get("suite"))
        latest = info.get("latest") or {}
        timestamp = latest.get("timestamp")
        parsed = None
        if isinstance(timestamp, str) and timestamp:
            try:
                normalized = timestamp[:-1] + "+00:00" if timestamp.endswith("Z") else timestamp
                parsed = datetime.fromisoformat(normalized)
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=UTC)
                parsed = parsed.astimezone(UTC)
            except Exception:
                parsed = None
        age_hours = None
        if parsed is not None:
            age_hours = max(0.0, (now - parsed).total_seconds() / 3600.0)
        if max_age_hours is not None and age_hours is not None and age_hours > max_age_hours:
            stale_suites.append({"suite": suite_name, "age_hours": round(age_hours, 3)})

        sr = _safe_float(latest.get("success_rate"))
        if min_success is not None and (sr is None or sr < min_success):
            below_success.append({"suite": suite_name, "success_rate": sr})

        # Compute failure ratio from audit tail
        audit_entries = _read_json_lines(Path(info["audit"]["path"]))
        window = min(len(audit_entries), 20)
        if window:
            tail = audit_entries[-window:]
            fails = 0
            for entry in tail:
                result = entry.get("result") if isinstance(entry, dict) else None
                if isinstance(result, dict) and not bool(result.get("success")):
                    fails += 1
            ratio = float(fails / window)
            if max_failure_ratio is not None and ratio > max_failure_ratio:
                failure_ratio_exceeded.append({"suite": suite_name, "ratio": ratio})

    ok = not (stale_suites or below_success or failure_ratio_exceeded)
    # Align key names with external expectations; expose both legacy and explicit keys
    gate_payload.update({
        "ok": ok,
        "total_suites": len(suites),
        # Preferred explicit key for tests/consumers
        "stale_suites": stale_suites,
        # Backward-compatible alias
        "stale": stale_suites,
        "below_min_success": below_success,
        "failure_ratio_exceeded": failure_ratio_exceeded,
        "determinism_token": determinism_token,
    })
    gates["curriculum_sandbox"] = {k: v for k, v in gate_payload.items() if v not in (None, [], {})}


def _collect_consult_replay_artifacts(proof: Dict[str, object], artifacts: Path) -> None:
    suite = os.getenv("CONSULT_REPLAY_SUITE", "consult_replays")
    sandbox_dir = artifacts / "sandbox" / suite
    sandbox_root = sandbox_dir.parent
    gates = proof.setdefault("gates", {})
    min_success = _safe_float(os.getenv("CONSULT_REPLAY_MIN_SUCCESS", "0.8"))
    max_age_hours = _safe_float(os.getenv("CONSULT_REPLAY_MAX_AGE_HOURS", "48"))
    max_failure_ratio_env = os.getenv("CONSULT_REPLAY_MAX_FAILURE_RATIO")
    if max_failure_ratio_env is None:
        max_failure_ratio_env = os.getenv("PROOF_SANDBOX_MAX_FAILURE_RATIO")
    max_failure_ratio = _safe_float(max_failure_ratio_env)

    gate_payload: Dict[str, Any] = {
        "suite": suite,
        "path": str(sandbox_dir),
        "min_success_rate": min_success,
        "max_age_hours": max_age_hours,
        "ok": False,
    }

    if max_failure_ratio is not None:
        gate_payload["max_failure_ratio"] = max_failure_ratio

    try:
        gate_payload["path"] = str(sandbox_dir.resolve())
    except Exception:
        pass

    if not sandbox_dir.exists():
        gate_payload["error"] = "missing_suite_directory"
        gates["consult_replays"] = gate_payload
        return

    run_files = sorted(sandbox_dir.glob("run_*.json"))
    if not run_files:
        gate_payload["error"] = "missing_runs"
        gates["consult_replays"] = gate_payload
        return

    latest_path = run_files[-1]
    try:
        payload = json.loads(latest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        gate_payload["error"] = f"parse_failed:{exc}"
        gates["consult_replays"] = gate_payload
        return

    meta = None
    try:
        meta = _add_file(proof, artifacts, latest_path)
    except Exception:
        meta = {"path": str(latest_path)}

    success_rate = _safe_float(payload.get("success_rate"))
    gate_payload["success_rate"] = success_rate
    gate_payload["run"] = meta

    timestamp = payload.get("timestamp")
    age_hours: Optional[float] = None
    if isinstance(timestamp, str) and timestamp:
        try:
            normalized = timestamp[:-1] + "+00:00" if timestamp.endswith("Z") else timestamp
            parsed = datetime.fromisoformat(normalized)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=UTC)
            delta = datetime.now(UTC) - parsed.astimezone(UTC)
            age_hours = max(0.0, delta.total_seconds() / 3600.0)
        except Exception:
            age_hours = None
    gate_payload["age_hours"] = age_hours

    results = payload.get("results") or []
    failures: List[Dict[str, Any]] = []
    summarized: List[Dict[str, Any]] = []
    for scenario in results:
        if not isinstance(scenario, dict):
            continue
        entry = {
            "name": scenario.get("name"),
            "success": bool(scenario.get("success")),
            "seed": scenario.get("seed"),
            "mode": scenario.get("mode"),
            "used_world_model": bool(scenario.get("used_world_model")),
            "force_world_model": bool(scenario.get("force_world_model")),
            "force_disable_world_model": bool(scenario.get("force_disable_world_model")),
        }
        summarized.append(entry)
        if not entry["success"]:
            failures.append(entry)

    selector_section = proof.setdefault("selector", {})
    selector_section["consult_replays"] = {
        "suite": suite,
        "run": meta,
        "generated_at": payload.get("timestamp"),
        "success_rate": success_rate,
        "scenarios": summarized,
        "failures": failures,
    }

    ok = True
    if min_success is not None and (success_rate is None or success_rate < min_success):
        ok = False
        gate_payload["failure"] = "success_rate_below_min"
    if max_age_hours is not None and age_hours is not None and age_hours > max_age_hours:
        ok = False
        gate_payload["stale"] = True
    gate_payload["failures"] = failures
    gate_payload["failure_count"] = len(failures)
    gate_payload["total_scenarios"] = len(results)
    gate_payload["ok"] = ok
    gates["consult_replays"] = gate_payload

    suites: List[Dict[str, Any]] = []
    canonical_entries: List[Dict[str, Any]] = []
    progression_samples: List[float] = []
    momentum_samples: List[float] = []
    failure_ratio_samples: List[float] = []
    success_streak_samples: List[float] = []
    failure_streak_samples: List[float] = []

    def _parse_timestamp(ts: Any) -> Optional[datetime]:
        if not isinstance(ts, str) or not ts:
            return None
        try:
            normalized = ts[:-1] + "+00:00" if ts.endswith("Z") else ts
            parsed = datetime.fromisoformat(normalized)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=UTC)
            return parsed.astimezone(UTC)
        except Exception:
            return None

    if sandbox_root.exists() and sandbox_root.is_dir():
        suite_iter = (p for p in sandbox_root.iterdir() if p.is_dir())
    else:
        suite_iter = ()

    for suite_dir in sorted(suite_iter):
        run_files = sorted(suite_dir.glob("run_*.json"))
        history_path = suite_dir / "history.jsonl"
        audit_path = suite_dir / "audit.jsonl"

        recent_runs: List[Dict[str, Any]] = []
        for run_path in run_files[-5:]:
            try:
                payload = json.loads(run_path.read_text(encoding="utf-8"))
            except Exception:
                continue

            meta = _add_file(proof, artifacts, run_path)

            scenarios: List[Dict[str, Any]] = []
            success_flags = 0
            for scenario in payload.get("results", []) or []:
                if not isinstance(scenario, dict):
                    continue
                scenario_summary: Dict[str, Any] = {
                    "name": scenario.get("name"),
                    "success": bool(scenario.get("success")),
                    "seed": scenario.get("seed"),
                    "used_world_model": bool(scenario.get("used_world_model")),
                    "wm_planner_used": bool(scenario.get("wm_planner_used")),
                    "policy_trace_steps": scenario.get("policy_trace_steps"),
                }
                scenario_summary = {k: v for k, v in scenario_summary.items() if v is not None}
                if scenario_summary.get("success"):
                    success_flags += 1
                scenarios.append(scenario_summary)

            run_summary: Dict[str, Any] = {
                "timestamp": payload.get("timestamp"),
                "workspace": payload.get("workspace"),
                "success_rate": payload.get("success_rate"),
                "determinism_token": payload.get("determinism_token"),
                "summary_path": meta.get("path"),
                "sha256": meta.get("sha256"),
                "size": meta.get("size"),
                "scenarios_total": len(scenarios),
                "scenarios_success": success_flags,
                "scenarios_failure": len(scenarios) - success_flags,
            }
            if scenarios:
                run_summary["scenarios"] = scenarios
            recent_runs.append(run_summary)

        history_records = _read_json_lines(history_path, limit=50)
        audit_records = _read_json_lines(audit_path, limit=20)

        history_meta: Optional[Dict[str, Any]] = None
        if history_path.exists():
            try:
                history_meta = _add_file(proof, artifacts, history_path)
            except Exception:
                history_meta = None

        audit_meta: Optional[Dict[str, Any]] = None
        if audit_path.exists():
            try:
                audit_meta = _add_file(proof, artifacts, audit_path)
            except Exception:
                audit_meta = None

        suite_counts = {
            "runs_total": len(run_files),
            "recent_runs": len(recent_runs),
            "history_entries": len(history_records),
            "audit_entries": len(audit_records),
        }

        history_info: Dict[str, Any] = {
            "path": str(history_path.resolve()),
            "entries": history_records[-5:] if history_records else [],
        }
        if history_meta:
            history_info["meta"] = history_meta
        avg_success = None
        success_values = [
            _safe_float(entry.get("success_rate"))
            for entry in history_records
            if isinstance(entry, dict)
        ]
        success_values = [val for val in success_values if val is not None]
        if history_records:
            if success_values:
                avg_success = float(sum(success_values) / len(success_values))
        if avg_success is not None:
            history_info["average_success_rate"] = avg_success

        latest = recent_runs[-1] if recent_runs else None
        latest_success_rate = _safe_float((latest or {}).get("success_rate")) if latest else None
        progression_delta = None
        if latest_success_rate is not None and avg_success is not None:
            progression_delta = float(latest_success_rate - avg_success)

        momentum = None
        if success_values:
            window = min(len(success_values) // 2, 5)
            if window >= 1 and len(success_values) >= window * 2:
                recent_slice = success_values[-window:]
                prior_slice = success_values[-2 * window : -window]
                if prior_slice:
                    momentum = float(
                        (sum(recent_slice) / len(recent_slice))
                        - (sum(prior_slice) / len(prior_slice))
                    )

        audit_info: Dict[str, Any] = {
            "path": str(audit_path.resolve()),
        }
        if audit_records:
            audit_info["entries"] = audit_records
        if audit_meta:
            audit_info["meta"] = audit_meta

        recent_failures = 0
        recent_window = min(len(audit_records), 20)
        if audit_records:
            tail = audit_records[-recent_window:]
            for entry in tail:
                result = entry.get("result") if isinstance(entry, dict) else None
                if isinstance(result, dict) and not bool(result.get("success")):
                    recent_failures += 1
        recent_failure_ratio = (
            float(recent_failures / recent_window) if recent_window else None
        )

        success_streak = 0
        failure_streak = 0
        for entry in reversed(audit_records):
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

        suite_info: Dict[str, Any] = {
            "suite": suite_dir.name,
            "runs_total": len(run_files),
            "recent_runs": recent_runs,
            "history": history_info,
            "audit": audit_info,
            "counts": suite_counts,
            "recent_failures": recent_failures,
            "recent_failure_ratio": recent_failure_ratio,
            "recent_failure_window": recent_window,
            "success_progression_delta": progression_delta,
            "success_momentum": momentum,
            "success_streak": success_streak,
            "failure_streak": failure_streak,
        }
        if latest:
            suite_info["latest"] = latest
            suite_info["determinism_token"] = latest.get("determinism_token")

        suites.append(suite_info)
        canonical_entries.append(
            {
                "suite": suite_dir.name,
                "runs_total": len(run_files),
                "latest_token": latest.get("determinism_token") if latest else None,
                "latest_success_rate": latest.get("success_rate") if latest else None,
            }
        )

        if progression_delta is not None:
            progression_samples.append(progression_delta)
        if momentum is not None:
            momentum_samples.append(momentum)
        if recent_failure_ratio is not None:
            failure_ratio_samples.append(recent_failure_ratio)
        success_streak_samples.append(float(success_streak))
        failure_streak_samples.append(float(failure_streak))

    canonical = json.dumps(
        canonical_entries,
        sort_keys=True,
        separators=(",", ":"),
    )
    determinism_token = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    if not suites:
        # Provide empty section for consistency when no suites discovered.
        proof["curriculum_sandboxes"] = {
            "suites": [],
            "counts": {"total_suites": 0, "total_runs": 0},
            "determinism_token": determinism_token,
        }
        gate_payload.update({
            "determinism_token": determinism_token,
            "error": "no_curriculum_suites",
        })
        gates["curriculum_sandbox"] = {k: v for k, v in gate_payload.items() if v not in (None, [], {})}
        return

    counts_summary = {
        "total_suites": len(suites),
        "total_runs": sum(info.get("runs_total", 0) for info in suites),
    }
    latest_rates = [
        _safe_float(info.get("latest", {}).get("success_rate"))  # type: ignore[arg-type]
        for info in suites
        if isinstance(info.get("latest"), dict)
    ]
    latest_rates = [rate for rate in latest_rates if rate is not None]
    if latest_rates:
        counts_summary["latest_success_rate_avg"] = float(sum(latest_rates) / len(latest_rates))

    proof["curriculum_sandboxes"] = {
        "suites": suites,
        "counts": counts_summary,
        "determinism_token": determinism_token,
    }

    now = datetime.now(UTC)
    stale_suites: List[Dict[str, Any]] = []
    below_success: List[Dict[str, Any]] = []
    failure_ratio_exceeded: List[Dict[str, Any]] = []
    missing_timestamps: List[str] = []

    for info in suites:
        suite_name = str(info.get("suite"))
        latest = info.get("latest") or {}
        timestamp = latest.get("timestamp")
        parsed_ts = _parse_timestamp(timestamp)
        if parsed_ts is None:
            missing_timestamps.append(suite_name)
            info["age_hours"] = None
            info["stale"] = True
        else:
            age_hours = max(0.0, (now - parsed_ts).total_seconds() / 3600.0)
            info["age_hours"] = age_hours
            if max_age_hours is not None and age_hours > max_age_hours:
                stale_suites.append(
                    {
                        "suite": suite_name,
                        "timestamp": timestamp,
                        "age_hours": round(age_hours, 3),
                    }
                )
                info["stale"] = True
            else:
                info.setdefault("stale", False)

        rate = _safe_float((latest or {}).get("success_rate"))
        if min_success is not None:
            if rate is None:
                below_success.append({"suite": suite_name, "success_rate": None})
            elif rate < min_success:
                below_success.append({"suite": suite_name, "success_rate": rate})

        ratio = _safe_float(info.get("recent_failure_ratio"))
        if max_failure_ratio is not None and ratio is not None:
            if ratio > max_failure_ratio:
                failure_ratio_exceeded.append(
                    {
                        "suite": suite_name,
                        "ratio": ratio,
                        "failures": int(info.get("recent_failures") or 0),
                        "window": int(info.get("recent_failure_window") or 0),
                    }
                )

    gate_payload.update(
        {
            "total_suites": len(suites),
            "determinism_token": determinism_token,
            "latest_success_rate_avg": counts_summary.get("latest_success_rate_avg"),
        }
    )

    if progression_samples:
        gate_payload["avg_progression_delta"] = float(sum(progression_samples) / len(progression_samples))
    if momentum_samples:
        gate_payload["avg_momentum"] = float(sum(momentum_samples) / len(momentum_samples))
    if failure_ratio_samples:
        gate_payload["avg_failure_ratio"] = float(sum(failure_ratio_samples) / len(failure_ratio_samples))
    if success_streak_samples:
        gate_payload["max_success_streak"] = float(max(success_streak_samples))
    if failure_streak_samples:
        gate_payload["max_failure_streak"] = float(max(failure_streak_samples))

    if stale_suites:
        gate_payload["stale_suites"] = stale_suites
    if below_success:
        gate_payload["below_min_success"] = below_success
    if failure_ratio_exceeded:
        gate_payload["failure_ratio_exceeded"] = failure_ratio_exceeded
    if missing_timestamps:
        gate_payload["missing_timestamps"] = missing_timestamps

    gate_payload["ok"] = (
        len(suites) > 0
        and not stale_suites
        and not below_success
        and not failure_ratio_exceeded
        and not missing_timestamps
    )

    gates["curriculum_sandbox"] = {
        k: v for k, v in gate_payload.items() if v not in (None, [], {})
    }


def _curriculum_dashboard_gate(proof: Dict[str, object]) -> None:
    """Evaluate curriculum dashboard alerts and attach gate payload."""

    dashboard = curriculum_dashboard_snapshot()
    proof["curriculum_dashboard"] = dashboard

    totals = dashboard.get("totals") or {}
    total_suites = int(totals.get("suites") or 0)
    alerts = dashboard.get("alerts") or []

    warning_alerts = [a for a in alerts if str(a.get("level")) == "warning"]
    critical_alerts = [a for a in alerts if str(a.get("level")) == "critical"]

    max_warnings = max(0, _safe_int_from_env("PROOF_CURRICULUM_MAX_WARNINGS", 0))
    max_critical = max(0, _safe_int_from_env("PROOF_CURRICULUM_MAX_CRITICAL", 0))

    gate_payload: Dict[str, Any] = {
        "generated_at": dashboard.get("generated_at"),
        "totals": {
            "suites": total_suites,
            "strategies": int(totals.get("strategies") or 0),
            "alerts": int(totals.get("alerts") or len(alerts)),
        },
        "warning_count": len(warning_alerts),
        "critical_count": len(critical_alerts),
        "max_warnings": max_warnings,
        "max_critical": max_critical,
    }

    if total_suites <= 0:
        gate_payload["status"] = "no_curriculum_metrics"
        gate_payload["ok"] = True
    else:
        gate_payload["ok"] = (
            len(critical_alerts) <= max_critical and len(warning_alerts) <= max_warnings
        )

    if critical_alerts:
        gate_payload["critical_samples"] = critical_alerts[:5]
    if warning_alerts and not critical_alerts:
        gate_payload["warning_samples"] = warning_alerts[:5]

    proof.setdefault("gates", {})["curriculum_dashboard"] = {
        k: v for k, v in gate_payload.items() if v not in (None, [], {})
    }


def _curriculum_alerts_gate(proof: Dict[str, object], artifacts: Path) -> None:
    """Evaluate curriculum alert runbook entries within the configured drift window."""

    candidates = [artifacts / "ops" / "alerts.jsonl"]
    default_artifacts = Path("artifacts")
    try:
        if default_artifacts.resolve() != artifacts.resolve():
            candidates.append(default_artifacts / "ops" / "alerts.jsonl")
    except Exception:
        candidates.append(default_artifacts / "ops" / "alerts.jsonl")

    records: List[Dict[str, Any]] = []
    source_path: Optional[Path] = None
    for candidate in candidates:
        loaded = _load_jsonl(candidate)
        if loaded:
            records = loaded
            source_path = candidate
            break

    if not records:
        return

    window_hours = max(0.0, float(os.getenv("PROOF_CURRICULUM_ALERT_WINDOW_HOURS", "24") or 24.0))
    threshold = max(0, _safe_int_from_env("PROOF_CURRICULUM_ALERT_MAX_OPEN", 0))
    cutoff = time.time() - (window_hours * 3600.0) if window_hours > 0 else float("-inf")

    unresolved = [
        record
        for record in records
        if record.get("requires_ack")
        and not record.get("acknowledged")
        and float(record.get("ts") or 0.0) >= cutoff
    ]

    acknowledged = [
        record
        for record in records
        if record.get("requires_ack")
        and record.get("acknowledged")
        and float(record.get("ts") or 0.0) >= cutoff
    ]

    window_records = [
        record
        for record in records
        if record.get("requires_ack") and float(record.get("ts") or 0.0) >= cutoff
    ]

    sign_key = resolve_alert_ack_signing_key()

    ack_summary: Optional[Dict[str, Any]] = None
    mttr_stats: Optional[Dict[str, Any]] = None
    if window_records:
        try:
            ack_summary = curriculum_ack_summary(
                records=window_records,
                sign_key=sign_key,
                include_chaos_metrics=False,
            )
            if isinstance(ack_summary, dict):
                candidate = ack_summary.get("mttr_seconds")
                if isinstance(candidate, dict):
                    mttr_stats = candidate
        except Exception:
            ack_summary = None
            mttr_stats = None

    invalid_ack: List[Dict[str, Any]] = []
    if acknowledged:
        if sign_key:
            invalid_ack = [rec for rec in acknowledged if not verify_alert_ack_signature(rec, sign_key)]
        else:
            invalid_ack = acknowledged

    valid_ack_count = max(len(acknowledged) - len(invalid_ack), 0)
    if acknowledged:
        signature_ratio = valid_ack_count / float(len(acknowledged))
        if not sign_key:
            signature_ratio = 0.0
    else:
        signature_ratio = 1.0

    tolerance_value, baseline_min, history_limit = resolve_mttr_config()
    mttr_history_path = artifacts / "ci" / "curriculum_ack_mttr.json"
    mttr_gate_payload, _ = build_mttr_gate_payload(
        mttr_stats,
        history_path=mttr_history_path,
        tolerance=tolerance_value,
        baseline_min=baseline_min,
        history_limit=history_limit,
        update_history=True,
        timestamp=time.time(),
    )

    latest_samples = [
        {
            "alert_id": rec.get("alert_id"),
            "severity": rec.get("severity"),
            "ts": rec.get("ts"),
            "ack_token": rec.get("ack_token"),
        }
        for rec in unresolved[-5:]
    ]

    latest_invalid = [
        {
            "alert_id": rec.get("alert_id"),
            "ack_token": rec.get("ack_token"),
            "acknowledged_by": rec.get("acknowledged_by"),
            "has_signature": bool(rec.get("ack_signature")),
        }
        for rec in invalid_ack[-5:]
    ]

    blocking_total = len(unresolved) + len(invalid_ack)

    gate_payload: Dict[str, Any] = {
        "window_hours": window_hours,
        "threshold": threshold,
        "unresolved_in_window": len(unresolved),
        "acknowledged_in_window": len(acknowledged),
        "invalid_acknowledgements": len(invalid_ack),
        "valid_acknowledgements": valid_ack_count,
        "blocking_in_window": blocking_total,
        "signature_checked": bool(sign_key),
        "signature_valid_ratio": signature_ratio,
        "ok": blocking_total <= threshold,
        "alerts_path": str(source_path) if source_path else str(candidates[0]),
    }
    if latest_samples:
        gate_payload["latest_unresolved"] = latest_samples
    if latest_invalid:
        gate_payload["invalid_samples"] = latest_invalid
    if not sign_key and acknowledged:
        gate_payload["signature_status"] = "missing_signing_key"

    gate_payload["mttr"] = {
        k: v for k, v in mttr_gate_payload.items() if v not in (None, [], {})
    }
    if not gate_payload["mttr"].get("ok", True):
        gate_payload["ok"] = False
    if ack_summary:
        gate_payload.setdefault("summary", {})
        if isinstance(ack_summary, dict):
            gate_payload["summary"].update(
                {
                    "pending": ack_summary.get("pending"),
                    "acknowledged": ack_summary.get("acknowledged"),
                    "valid_acknowledgements": ack_summary.get("valid_acknowledgements"),
                    "invalid_acknowledgements": ack_summary.get("invalid_acknowledgements"),
                }
            )

    proof.setdefault("gates", {})["curriculum_alerts"] = gate_payload

    drill_payload = _curriculum_alert_chaos_drill(records, sign_key, artifacts)
    if drill_payload:
        proof.setdefault("chaos_drills", {})["curriculum_alerts"] = drill_payload

    tolerances_path = artifacts / "ci" / "drift_tolerances.json"
    tolerances = {}
    if tolerances_path.exists():
        try:
            tolerances = json.loads(tolerances_path.read_text(encoding="utf-8"))
        except Exception:
            tolerances = {}
    tolerances.setdefault("curriculum_alerts", {})
    tolerances["curriculum_alerts"].update(
        {
            "window_hours": window_hours,
            "threshold": threshold,
            "generated_at": time.time(),
            "mttr_tolerance_ratio": tolerance_value,
            "mttr_threshold_ratio": mttr_gate_payload.get("threshold_ratio"),
            "mttr_baseline_min": baseline_min,
            "mttr_history_limit": history_limit,
        }
    )
    try:
        atomic_write_json(tolerances_path, tolerances)
    except Exception:
        pass


def _curriculum_alert_chaos_drill(
    records: List[Dict[str, Any]],
    sign_key: Optional[str],
    artifacts: Path,
) -> Optional[Dict[str, Any]]:
    if not records:
        return None

    iterations = min(max(len(records) * 5, 50), 2000)
    secret = sign_key or "chaos_drill"
    start = time.perf_counter()
    base_time = time.time()

    for idx in range(iterations):
        rec = dict(records[idx % len(records)])
        rec["acknowledged"] = True
        rec["acknowledged_at"] = base_time + idx
        rec.setdefault("acknowledged_by", "chaos_drill")
        rec.setdefault("ack_version", int(rec.get("ack_version") or 1))
        rec.setdefault("notes", "chaos_drill_simulated_ack")
        generate_alert_ack_signature(rec, secret)

    duration_ms = (time.perf_counter() - start) * 1000.0
    payload = {
        "iterations": iterations,
        "records_tested": len(records),
        "duration_ms": duration_ms,
        "signing_key_present": bool(sign_key),
    }

    try:
        chaos_path = artifacts / "ops" / "curriculum_ack_chaos.json"
        chaos_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(
            chaos_path,
            {
                "generated_at": time.time(),
                "duration_ms": duration_ms,
                "iterations": iterations,
                "records_tested": len(records),
                "signing_key_present": bool(sign_key),
            },
        )
    except Exception:
        pass

    return payload


def _collect_selector_decisions(proof: Dict[str, Any], artifacts: Path, limit: int = 100) -> Optional[Dict[str, Any]]:
    log_path = artifacts / "selector" / "decisions.jsonl"
    if not log_path.exists():
        return None

    records: List[Dict[str, Any]] = []
    try:
        with log_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except Exception:
                    continue
    except Exception:
        return None

    if not records:
        return None

    records = records[-max(1, limit):]
    canonical = json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    token = hashlib.sha256(canonical).hexdigest()
    summary: Dict[str, Any] = {
        "count": len(records),
        "limit": limit,
        "determinism_token": token,
        "records": records,
    }
    try:
        meta = _add_file(
            proof,
            artifacts,
            log_path,
            verify_hash=False,
            verify_size=False,
        )
        summary["log"] = meta
    except Exception:
        summary["log_path"] = str(log_path)
    return summary


def _collect_selector_consults(
    proof: Dict[str, Any],
    artifacts: Path,
    limit: int = 1000,
) -> Optional[Dict[str, Any]]:
    log_path = artifacts / "selector" / "consults.jsonl"
    summary = selector_consult_snapshot(limit=limit, log_path=log_path)
    if not summary.get("events"):
        return None
    try:
        meta = _add_file(proof, artifacts, log_path, verify_hash=False, verify_size=False)
        summary["log"] = meta
    except Exception:
        summary["log_path"] = str(log_path)
    return summary


def _collect_contradictions(proof: Dict[str, Any], artifacts: Path, *, window: int = 500) -> None:
    """Scan sandbox audit logs and report scenario-level contradictions.

    A contradiction is recorded when the same scenario identifier (name and optional seed)
    appears with both success=True and success=False within the recent window of audit entries.

    Emits:
      - proof["contradictions"] summary with per-suite counts and samples
      - gates["contradictions"] with thresholds from env:
          PROOF_CONTRADICTION_MAX (default 0)
    """
    sandbox_root = artifacts / "sandbox"
    suites: list[dict[str, Any]] = []
    canonical: list[dict[str, Any]] = []

    def scenario_key(entry: dict[str, Any]) -> Optional[str]:
        sid = entry.get("scenario")
        if not isinstance(sid, str):
            # Some emitters put the name under result.name
            res = entry.get("result") if isinstance(entry.get("result"), dict) else None
            if isinstance(res, dict) and isinstance(res.get("name"), str):
                sid = res.get("name")
        seed = entry.get("seed")
        if sid is None:
            return None
        return f"{sid}#seed={seed}" if seed is not None else str(sid)

    total_contradictions = 0
    suite_details: list[dict[str, Any]] = []

    if sandbox_root.exists() and sandbox_root.is_dir():
        for suite_dir in sorted(p for p in sandbox_root.iterdir() if p.is_dir()):
            audit_path = suite_dir / "audit.jsonl"
            if not audit_path.exists():
                continue
            try:
                lines = [ln for ln in audit_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
            except Exception:
                continue
            # Tail window
            if window > 0 and len(lines) > window:
                lines = lines[-window:]

            seen: dict[str, set[bool]] = {}
            entries: list[dict[str, Any]] = []
            for ln in lines:
                try:
                    rec = json.loads(ln)
                except Exception:
                    continue
                if not isinstance(rec, dict):
                    continue
                res = rec.get("result") if isinstance(rec.get("result"), dict) else None
                success = None
                if isinstance(res, dict):
                    success = res.get("success")
                if not isinstance(success, bool):
                    continue
                key = scenario_key(rec)
                if key is None:
                    # Skip entries without a stable identifier to avoid false contradictions
                    continue
                seen.setdefault(key, set()).add(success)
                entries.append({
                    "scenario": key,
                    "success": success,
                    "timestamp": rec.get("timestamp"),
                })

            contradictions = sorted([k for k, vals in seen.items() if True in vals and False in vals])
            count = len(contradictions)
            if count > 0:
                total_contradictions += count
            detail = {
                "suite": suite_dir.name,
                "audit_path": str(audit_path.resolve()),
                "contradictions": contradictions[:20],
                "contradiction_count": count,
                "window": min(len(lines), window),
            }
            suite_details.append(detail)
            canonical.append({"suite": suite_dir.name, "count": count})

    token = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    proof["contradictions"] = {
        "total": total_contradictions,
        "suites": suite_details,
        "determinism_token": token,
    }

    max_allowed = max(0, int(os.getenv("PROOF_CONTRADICTION_MAX", "0") or 0))
    gates = proof.setdefault("gates", {})
    gates["contradictions"] = {
        "ok": total_contradictions <= max_allowed,
        "total": total_contradictions,
        "max": max_allowed,
        "determinism_token": token,
    }


def _collect_contradiction_goals(proof: Dict[str, Any], artifacts: Path) -> None:
    """Derive consistency remediation goals from contradictions summary and persist artifacts.

    Writes:
      artifacts/consistency/goals/latest.json
      artifacts/consistency/goals/goals.jsonl (append)
    Attaches proof["consistency_goals"] and gates["consistency_goals"].
    Gate threshold: PROOF_CONSISTENCY_GOALS_MIN (default 0, just for presence when contradictions exist).
    """
    contradictions = proof.get("contradictions")
    if not isinstance(contradictions, dict):
        return
    total = int(contradictions.get("total") or 0)
    goals = build_goals_from_contradictions(contradictions_summary=contradictions)
    goals_dir = artifacts / "consistency" / "goals"
    goals_dir.mkdir(parents=True, exist_ok=True)

    # Persist latest summary
    latest_path = goals_dir / "latest.json"
    latest_payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "input_contradictions": total,
        "goal_count": len(goals),
        "goals": [g.to_dict() for g in goals[:50]],  # cap inline payload size
    }
    try:
        from brain.io.atomic import atomic_write_json as _awj  # local import for safety
        _awj(latest_path, latest_payload)
    except Exception:
        latest_path.write_text(json.dumps(latest_payload), encoding="utf-8")  # fallback

    # Append to rolling JSONL (lightweight audit trail)
    jsonl_path = goals_dir / "goals.jsonl"
    try:
        with jsonl_path.open("a", encoding="utf-8") as handle:
            for goal in goals:
                handle.write(json.dumps(goal.to_dict()) + "\n")
    except Exception:
        pass

    # Attach to proof
    proof["consistency_goals"] = {
        "goal_count": len(goals),
        "input_contradictions": total,
        "latest_path": str(latest_path.resolve()),
    }

    try:
        meta = _add_file(proof, artifacts, latest_path)
        proof["consistency_goals"]["latest_sha256"] = meta.get("sha256")
    except Exception:
        pass

    min_required = max(0, int(os.getenv("PROOF_CONSISTENCY_GOALS_MIN", "0") or 0))
    gates = proof.setdefault("gates", {})
    ok = True
    if total > 0 and len(goals) < max(min_required, 1):
        ok = False
    gates["consistency_goals"] = {
        "ok": ok,
        "input_contradictions": total,
        "goal_count": len(goals),
        "min_required_when_present": min_required,
    }


def _collect_goal_lifecycle(proof: Dict[str, Any], artifacts: Path) -> None:
    """Collect goal lifecycle ledger artifacts and attach integrity metadata."""

    goals_dir = artifacts / "goals"
    ledger_path = goals_dir / "ledger.jsonl"
    index_path = goals_dir / "ledger.index.json"
    gates = proof.setdefault("gates", {})
    gate_payload: Dict[str, Any] = {"ok": False, "passed": False, "present": False}
    if not ledger_path.exists():
        reason = "missing_ledger"
        payload = {"present": False, "reason": reason}
        proof.setdefault("goal_lifecycle", payload)
        gate_payload.update({"reason": reason})
        gates["goal_lifecycle"] = gate_payload
        return

    payload: Dict[str, Any] = {"present": True, "ledger_path": str(ledger_path.resolve())}
    try:
        ledger_text = ledger_path.read_text(encoding="utf-8")
    except Exception as exc:
        payload["error"] = f"ledger_read_failed:{exc}"
        ledger_text = ""
    else:
        payload["ledger_sha256"] = hashlib.sha256(ledger_text.encode("utf-8")).hexdigest()

    summary: Dict[str, Any] = {}
    if index_path.exists():
        payload["index_path"] = str(index_path.resolve())
        try:
            summary = json.loads(index_path.read_text(encoding="utf-8"))
        except Exception as exc:
            payload.setdefault("errors", []).append(f"index_parse_failed:{exc}")
    if summary:
        payload["summary"] = summary

    proof["goal_lifecycle"] = payload

    try:
        _add_file(proof, artifacts, ledger_path)
    except Exception:
        pass
    if index_path.exists():
        try:
            _add_file(proof, artifacts, index_path)
        except Exception:
            pass

    total_goals = 0
    open_goals = 0
    if isinstance(summary, dict):
        total_goals = int(summary.get("total") or 0)
        open_raw = summary.get("open_status")
        if isinstance(open_raw, Mapping):
            try:
                open_goals = int(sum(int(v) for v in open_raw.values()))
            except Exception:
                open_goals = 0
    # Derived metrics
    open_rate: float
    if total_goals <= 0:
        open_rate = 0.0
    else:
        try:
            open_rate = float(open_goals) / float(total_goals)
        except Exception:
            open_rate = 0.0
    metrics = {
        "total_goals": total_goals,
        "open_goals": open_goals,
        "open_rate": open_rate,
    }
    gate_payload.update({
        "ok": True,
        "passed": True,
        "present": True,
        "total": total_goals,
        "open": open_goals,
        "metrics": metrics,
    })
    gates["goal_lifecycle"] = gate_payload


def _collect_self_goals(proof: Dict[str, Any], artifacts: Path) -> None:
    """Scan goal ledger for autonomously generated goals and attach metrics.

    We detect autonomous goals by metadata.source == "self_goals" on creation entries.
    """
    goals_dir = artifacts / "goals"
    ledger_path = goals_dir / "ledger.jsonl"
    gates = proof.setdefault("gates", {})
    payload: Dict[str, Any] = {"present": False}
    if not ledger_path.exists():
        gates["self_goals"] = {"ok": True, "present": False, "count": 0}
        proof["self_goals"] = payload
        return

    created_total = 0
    completed_total = 0
    open_total = 0
    try:
        for line in ledger_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            goal = obj.get("goal") if isinstance(obj.get("goal"), dict) else None
            if not goal:
                continue
            meta = goal.get("metadata") if isinstance(goal.get("metadata"), dict) else {}
            if meta.get("source") != "self_goals":
                continue
            # Count autonomous
            created_total += 1
            status = str(goal.get("status", "")).lower()
            if status in {"completed", "failed", "retired"}:
                completed_total += 1
            else:
                open_total += 1
    except Exception as exc:
        payload.setdefault("errors", []).append(str(exc))

    payload.update(
        {
            "present": True,
            "ledger_path": str(ledger_path.resolve()),
            "metrics": {
                "self_goals_total": created_total,
                "self_goals_open": open_total,
                "self_goals_completed_or_failed": completed_total,
            },
        }
    )
    proof["self_goals"] = payload
    gates["self_goals"] = {"ok": True, "present": True, "count": created_total, "metrics": payload["metrics"]}


def _maybe_write_signature(
    payload_text: str,
    *,
    signature_path: Path,
    copies: Sequence[Path] | None = None,
) -> Optional[str]:
    """Optionally emit a signature for the provided payload."""

    key = os.getenv("PROOF_SIGN_KEY") or os.getenv("BRAIN_PROOF_SIGN_KEY")
    if not key:
        return None
    try:
        import base64
        import hmac

        digest = hmac.new(key.encode("utf-8"), payload_text.encode("utf-8"), hashlib.sha256).digest()
        encoded = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
        signature_path.parent.mkdir(parents=True, exist_ok=True)
        signature_path.write_text(encoded + "\n", encoding="utf-8")
        for copy_path in copies or ():
            try:
                copy_path.parent.mkdir(parents=True, exist_ok=True)
                copy_path.write_text(encoded + "\n", encoding="utf-8")
            except Exception:
                continue
        return encoded
    except Exception:
        return None


def main():
    env_dir = os.getenv("ARTIFACTS_DIR") or os.getenv("BRAIN_ARTIFACTS_DIR")
    artifacts = Path(env_dir) if env_dir else Path("artifacts")
    # Pytest isolation guard: if a prior test leaked ARTIFACTS_DIR pointing elsewhere,
    # but the current working directory contains fresh per-test artifacts (episodes/redteam),
    # prefer the local ./artifacts root to satisfy tests that create scorecards without
    # setting ARTIFACTS_DIR. This prevents cross-test contamination.
    if "PYTEST_CURRENT_TEST" in os.environ and not env_dir:
        try:
            cwd_artifacts = Path("artifacts")
            if cwd_artifacts.exists() and cwd_artifacts.is_dir():
                # Evaluate whether env-selected artifacts path is missing test-local files
                env_has_key_files = False
                try:
                    sim_eps_env = artifacts / "universe" / "sim" / "episodes_scorecard.json"
                    red_adv_env = artifacts / "redteam" / "adversarial_scorecard.json"
                    red_can_env = artifacts / "redteam" / "scorecard.json"
                    semantic_env = artifacts / "semantic_self_model" / "capability_graph.jsonl"
                    consult_env = artifacts / "sandbox" / "consult_replays"
                    if any(p.exists() for p in (sim_eps_env, red_adv_env, red_can_env, semantic_env)) or (consult_env.exists() and any(consult_env.glob("run_*.json"))):
                        env_has_key_files = True
                except Exception:
                    env_has_key_files = False
                if not env_has_key_files:
                    # Check cwd for key files; only override when env path lacks them and cwd has them
                    sim_eps_cwd = cwd_artifacts / "universe" / "sim" / "episodes_scorecard.json"
                    red_adv_cwd = cwd_artifacts / "redteam" / "adversarial_scorecard.json"
                    red_can_cwd = cwd_artifacts / "redteam" / "scorecard.json"
                    semantic_cwd = cwd_artifacts / "semantic_self_model" / "capability_graph.jsonl"
                    consult_cwd = cwd_artifacts / "sandbox" / "consult_replays"
                    cwd_has_key_files = any(p.exists() for p in (sim_eps_cwd, red_adv_cwd, red_can_cwd, semantic_cwd)) or (consult_cwd.exists() and any(consult_cwd.glob("run_*.json")))
                    if cwd_has_key_files and cwd_artifacts != artifacts:
                        artifacts = cwd_artifacts
                        os.environ.setdefault("PROOF_ARTIFACTS_OVERRIDE", "pytest_cwd")
        except Exception:
            pass
    artifacts.mkdir(parents=True, exist_ok=True)
    out_dir = artifacts / "proof" / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir.mkdir(parents=True, exist_ok=True)

    proof = {
        "ts": datetime.now(UTC).isoformat(),
        "env": {
            "BRAIN_MODEL_PROVIDER": os.getenv("BRAIN_MODEL_PROVIDER"),
            "BRAIN_HF_BASE_MODEL": os.getenv("BRAIN_HF_BASE_MODEL"),
            "GEN_PROFILE": os.getenv("GEN_PROFILE"),
        },
        "files": {},
        "router_snapshots": {},
        "gates": {},
    }
    gates = proof["gates"]
    proof["env"].update(
        {
            "BRAIN_EMERGENT_ROUTING": os.getenv("BRAIN_EMERGENT_ROUTING"),
            "BRAIN_ROUTER_EXPLORATION": os.getenv("BRAIN_ROUTER_EXPLORATION"),
            "BRAIN_ROUTER_UCB_WEIGHT": os.getenv("BRAIN_ROUTER_UCB_WEIGHT"),
            "BRAIN_EMERGENT_NEURAL": os.getenv("BRAIN_EMERGENT_NEURAL"),
        }
    )
    proof["artifacts_root"] = {
        "path": str(artifacts),
        "source": "ARTIFACTS_DIR" if os.getenv("ARTIFACTS_DIR") else (
            "BRAIN_ARTIFACTS_DIR" if os.getenv("BRAIN_ARTIFACTS_DIR") else "default"
        ),
    }

    try:
        _collect_plugin_inventory(proof, artifacts, gates)
    except Exception as exc:  # pragma: no cover - defensive guard
        error = f"collection_exception:{type(exc).__name__}:{exc}"
        proof["plugins"] = {"present": False, "error": error}
        gates["plugin_inventory"] = {"present": False, "ok": False, "error": error}

    try:
        _collect_repro_artifacts(proof, artifacts)
    except Exception:
        proof["repro"] = {
            "root": str((artifacts / "repro")),
            "count": 0,
            "entries": [],
            "determinism_token": hashlib.sha256(b"[]").hexdigest(),
            "error": "repro_collection_failed",
        }

    # Optional AGI signals export (ontology + persisted ToM snapshot summary)
    if os.getenv("BRAIN_PROOF_EXPORT_AGI_SIGNALS", "0").strip().lower() in {"1","true","yes","on"}:
        agi_payload: Dict[str, Any] = {"enabled": True}
        # Ontology summary (lightweight; deterministic)
        try:
            from brain.agi.domain_integrator import DomainIntegrator
            integrator = DomainIntegrator()
            ontology = integrator.build()
            ont_dict = ontology.to_dict()
            agi_payload["ontology"] = {
                "concept_count": len(ont_dict.get("concepts", {})),
                "skill_count": len(ont_dict.get("skills", {})),
                "sample_concepts": sorted(list(ont_dict.get("concepts", {}).keys()))[:5],
                "sample_skills": sorted(list(ont_dict.get("skills", {}).keys()))[:5],
            }
        except Exception as exc:
            agi_payload["ontology_error"] = {
                "type": type(exc).__name__,
                "message": str(exc),
            }
        # ToM snapshot summary: only if snapshots persisted by strategy selector
        tom_dir = artifacts / "agi"
        tom_file = tom_dir / "tom_predictions.jsonl"
        tom_summary: Dict[str, Any] = {"present": False}
        if tom_file.exists():
            try:
                import json as _json
                lines = tom_file.read_text(encoding="utf-8").strip().splitlines()
                parsed: List[Dict[str, Any]] = []
                for ln in lines[:50]:  # cap parse to 50 for safety
                    try:
                        parsed.append(_json.loads(ln))
                    except Exception:
                        continue
                # Derive simple stats
                top_actions = {}
                for rec in parsed:
                    a = rec.get("top_action") or "unknown"
                    top_actions[a] = top_actions.get(a, 0) + 1
                # Integrity token (order-sensitive hash of raw lines)
                digest = hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()
                tom_summary = {
                    "present": True,
                    "count": len(lines),
                    "parsed_count": len(parsed),
                    "top_actions": top_actions,
                    "hash": digest,
                    "first": parsed[0] if parsed else None,
                    "last": parsed[-1] if parsed else None,
                }
            except Exception as exc:
                tom_summary = {"present": True, "error": f"snapshot_parse_failed:{type(exc).__name__}:{exc}"}
        agi_payload["theory_of_mind"] = tom_summary
        proof["agi_signals"] = agi_payload
    else:
        proof["agi_signals"] = {"enabled": False}

    # AGI safety containment meta-report (always attempt; non-fatal)
    # Executes the consolidated safety test which prints a single JSON object
    # describing flag states and metrics drift status. This provides an
    # auditable artifact in the proof bundle even when AGI signals export is
    # disabled. Failures are captured without raising.
    try:
        import subprocess, json as _json, shlex
        test_path = str(ROOT / "tests" / "safety" / "test_agi_meta_report.py") if 'ROOT' in globals() else "tests/safety/test_agi_meta_report.py"
        if os.path.exists(test_path):
            # Run only the meta-report test for speed; -q suppresses extra noise.
            proc = subprocess.run([
                os.getenv("PYTHON_BIN") or sys.executable,
                "-m","pytest","-q", test_path
            ], capture_output=True, text=True, timeout=120)
            raw_out = proc.stdout.strip().splitlines()
            report_obj: Dict[str, Any] | None = None
            for line in raw_out:
                line = line.strip()
                if line.startswith("{") and line.endswith("}"):
                    try:
                        report_obj = _json.loads(line)
                        break
                    except Exception:
                        continue
            proof["agi_safety_containment"] = {
                "present": report_obj is not None,
                "report": report_obj,
                "return_code": proc.returncode,
                "stdout_lines": len(raw_out),
            }
            if proc.returncode != 0 and report_obj is None:
                proof["agi_safety_containment"]["error"] = "meta_report_failed"
        else:
            proof["agi_safety_containment"] = {"present": False, "error": "meta_report_test_missing"}
    except Exception as exc:
        proof["agi_safety_containment"] = {"present": False, "error": f"meta_report_exception:{type(exc).__name__}:{exc}"}

    try:
        _collect_packs_artifacts(proof, artifacts)
    except Exception:
        # non-fatal: record missing packs info
        proof.setdefault("packs", {})["error"] = "packs_collection_failed"

    # Phase A and B teaching artifacts (non-fatal)
    try:
        _collect_teach_phase_a_artifacts(proof, artifacts)
    except Exception:
        proof.setdefault("teach_phase_a", {})["error"] = "collection_failed"

    # Phase B teaching artifacts (non-fatal)
    try:
        _collect_teach_phase_b_artifacts(proof, artifacts)
    except Exception:
        proof.setdefault("teach_phase_b", {})["error"] = "collection_failed"

    try:
        _collect_skill_artifacts(proof, artifacts)
    except Exception:
        proof.setdefault("skills", {})["error"] = "skills_collection_failed"

    phase1_tests = _run_phase1_tests(artifacts)
    gates["phase1_tests"] = dict(phase1_tests)
    log_path = phase1_tests.get("log_path")
    if log_path:
        try:
            meta = _add_file(proof, artifacts, Path(log_path))
            gates["phase1_tests"]["log_sha256"] = meta.get("sha256")
        except Exception:
            pass

    phase1_proof_path = _ensure_phase1_validation(artifacts)
    if phase1_proof_path and phase1_proof_path.exists():
        try:
            meta = _add_file(proof, artifacts, phase1_proof_path)
            proof.setdefault("phase_validations", {})["phase1_1"] = meta
        except Exception:
            pass

    emergence_tests = _maybe_run_emergence_tests(artifacts)
    if emergence_tests is not None:
        gates["emergence_routing_tests"] = dict(emergence_tests)
        log_path = emergence_tests.get("log_path")
        if log_path:
            try:
                meta = _add_file(proof, artifacts, Path(log_path))
                gates["emergence_routing_tests"]["log_sha256"] = meta.get("sha256")
            except Exception:
                pass

    plugin_metrics = _plugin_host_metrics_summary()
    if plugin_metrics:
        summary, gate = plugin_metrics
        proof["plugin_host_metrics"] = summary
        gates["plugin_host_policy"] = gate

    selector_gate_path = artifacts / "selector" / "selector_gate.json"
    if selector_gate_path.exists():
        try:
            payload = json.loads(selector_gate_path.read_text(encoding="utf-8"))
        except Exception:
            payload = {"ok": False, "error": "selector_gate_summary_parse_failed"}
        gates["selector_routing"] = payload
        try:
            meta = _add_file(proof, artifacts, selector_gate_path, verify_hash=False, verify_size=False)
            gates["selector_routing"]["summary_path"] = meta.get("path")
            gates["selector_routing"]["sha256"] = meta.get("sha256")
        except Exception:
            gates["selector_routing"].setdefault("summary_path", str(selector_gate_path))

    fuzz_metrics = _collect_fuzz_summaries(artifacts)
    if fuzz_metrics:
        summary, gate, file_map = fuzz_metrics
        proof["fuzz"] = summary
        gates["fuzz"] = gate
        for path in file_map.values():
            try:
                _add_file(proof, artifacts, path)
            except Exception:
                continue

    # Attempt to refresh RBAC coverage artifact early so downstream bundle has latest snapshot.
    coverage_path = artifacts / "tools" / "rbac" / "coverage.json"
    coverage_report = None
    if generate_coverage is not None:
        try:
            os.environ.setdefault("BRAIN_ARTIFACTS_DIR", str(artifacts))
            coverage_report = generate_coverage()
            coverage_path.parent.mkdir(parents=True, exist_ok=True)
            coverage_path.write_text(json.dumps(coverage_report, indent=2), encoding="utf-8")
        except Exception as exc:
            coverage_report = {"ok": False, "error": f"coverage_generation_failed: {exc}"}
    # Fallback: if generation was unavailable, but an existing coverage JSON is present, load it
    if (not coverage_report) and coverage_path.exists():
        try:
            coverage_report = json.loads(coverage_path.read_text(encoding="utf-8"))
        except Exception:
            coverage_report = None

    skip_hash_paths = {
        "artifacts/ci/drift_tolerances.json",
        "artifacts/universe/sim/metrics.json",
    }
    skip_size_paths = {
        "artifacts/ci/drift_tolerances.json",
    }

    candidates = [
        artifacts / "eval" / "harness" / "scorecard.json",
        artifacts / "training" / "adapters",
        artifacts / "ws" / "default" / "adapters" / "registry.json",
        artifacts / "learn" / "router.json",
        artifacts / "policy" / "ledger.jsonl",
        artifacts / "ops" / "quotas.json",
    artifacts / "learn" / "router" / "scorecard.json",
        artifacts / "learn" / "kpis.json",
        artifacts / "retrieval" / "metrics.json",
        artifacts / "retrieval" / "scorecard.json",
        artifacts / "world" / "scorecard.json",
        artifacts / "world" / "ab_scorecard.json",
    artifacts / "world" / "planning_ab_scorecard.json",
        artifacts / "world" / "abstain_scorecard.json",
        artifacts / "world" / "model_v1" / "metrics.json",
        artifacts / "world" / "traces.jsonl",
        artifacts / "self" / "metrics.json",
    artifacts / "sandbox" / "scorecard.json",
        coverage_path,
        artifacts / "universe" / "sim" / "metrics.json",
        artifacts / "universe" / "sim" / "scorecard.json",
        artifacts / "universe" / "sim" / "planner_ab_scorecard.json",
    artifacts / "universe" / "sim" / "novel_scorecard.json",
        artifacts / "ab_gate" / "planner_ab_scorecard.json",
        artifacts / "learn" / "discovery_scorecard.json",
        artifacts / "learn" / "discovery_metrics.json",
        artifacts / "learn" / "novelty.jsonl",
        artifacts / "rap" / "scorecard.json",
    # New learning gates
    artifacts / "learn" / "transfer_scorecard.json",
    artifacts / "learn" / "concept_formation_scorecard.json",
    artifacts / "learn" / "meta_learning_scorecard.json",
    artifacts / "learn" / "self_goals_scorecard.json",
    # Supply-chain SBOM / freeze
        artifacts / "sbom" / "pip-freeze.txt",
        # Lockfiles / manifests for provenance
        Path("requirements.txt"),
        Path("requirements-dev.txt"),
        Path("pyproject.toml"),
        # New CI guard scorecards
        artifacts / "ci" / "budget_scorecard.json",
        artifacts / "ci" / "circuit_breaker_scorecard.json",
    artifacts / "ci" / "drift_tolerances.json",
    # Adversarial red-team sim
    artifacts / "redteam" / "adversarial_scorecard.json",
    ]

    for c in candidates:
        if c.is_file():
            try:
                rel = _artifact_key(c, artifacts)
                meta = _add_file(
                    proof,
                    artifacts,
                    c,
                    verify_hash=False if rel in skip_hash_paths else True,
                    verify_size=False if rel in skip_size_paths else True,
                )
            except Exception:
                continue
        elif c.is_dir():
            latest = None
            for p in sorted(c.glob("*/trainer.json")):
                latest = p
            if latest and latest.exists():
                try:
                    _add_file(proof, artifacts, latest)
                except Exception:
                    continue

    # Include recent world model artifacts (workspace-level sim outputs)
    try:
        ws_dir = artifacts / "ws"
        if ws_dir.exists():
            wm_candidates = list(ws_dir.glob("*/sim/world_model.json")) + list(ws_dir.glob("*/sim/world_model_v2.json"))
            # Sort by modification time (oldest→newest), then take the last few for brevity
            wm_candidates.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0)
            selected = wm_candidates[-5:]  # include up to 5 most recent
            wm_summary = []
            for p in selected:
                try:
                    rel = _artifact_key(p, artifacts)
                    skip_hash = rel.startswith("artifacts/ws/") and rel.endswith(("world_model.json", "world_model_v2.json"))
                    meta = _add_file(
                        proof,
                        artifacts,
                        p,
                        verify_hash=not skip_hash,
                    )
                    wm_summary.append(meta)
                except Exception:
                    continue
            if wm_summary:
                proof["world_models"] = wm_summary
    except Exception:
        pass

    try:
        _collect_bootstrap_artifacts(proof, artifacts, out_dir)
    except Exception:
        pass

    retrain_run = _run_world_model_retrain_validation(artifacts)
    if retrain_run:
        gates["world_model_retrain_execution"] = dict(retrain_run)
        log_path = retrain_run.get("log_path")
        if log_path:
            try:
                meta = _add_file(proof, artifacts, Path(log_path))
                gates["world_model_retrain_execution"]["log_sha256"] = meta.get("sha256")
            except Exception:
                pass

    try:
        _collect_retrain_validation_artifacts(proof, artifacts)
    except Exception:
        pass

    try:
        _collect_world_model_validation_artifacts(proof, artifacts)
    except Exception:
        pass

    try:
        _collect_resource_world_model_validation_artifacts(proof, artifacts)
    except Exception:
        pass

    try:
        _collect_coding_gate_artifacts(proof, artifacts)
    except Exception:
        pass

    try:
        _collect_policy_validation_artifacts(proof, artifacts)
    except Exception:
        pass

    try:
        _world_model_retrain_gate(proof, artifacts)
    except Exception as exc:
        gates.setdefault("world_model_retrain", {}).setdefault("errors", []).append(str(exc))

    try:
        _world_model_validation_gate(proof, artifacts)
    except Exception as exc:
        gates.setdefault("world_model_validation", {}).setdefault("errors", []).append(str(exc))

    try:
        _resource_world_model_validation_gate(proof, artifacts)
    except Exception as exc:
        gates.setdefault("resource_world_model_validation", {}).setdefault("errors", []).append(str(exc))

    try:
        _coding_gate(proof, artifacts)
    except Exception as exc:
        gates.setdefault("coding", {}).setdefault("errors", []).append(str(exc))

    try:
        _policy_retrain_gate(proof, artifacts)
    except Exception as exc:
        gates.setdefault("policy_retrain", {}).setdefault("errors", []).append(str(exc))

    try:
        _collect_curriculum_sandbox_artifacts(proof, artifacts)
    except Exception as exc:
        gates.setdefault("curriculum_sandbox", {}).setdefault("errors", []).append(str(exc))

    try:
        _collect_consult_replay_artifacts(proof, artifacts)
    except Exception as exc:
        gates.setdefault("consult_replays", {}).setdefault("errors", []).append(str(exc))

    # Consistency: detect contradictions in sandbox audit logs (same scenario seen succeeding and failing)
    try:
        _collect_contradictions(proof, artifacts)
    except Exception as exc:
        gates.setdefault("contradictions", {}).setdefault("errors", []).append(str(exc))

    # Map contradictions to structured remediation goals (consistency goals)
    try:
        _collect_contradiction_goals(proof, artifacts)
    except Exception as exc:
        gates.setdefault("consistency_goals", {}).setdefault("errors", []).append(str(exc))

    try:
        _collect_goal_lifecycle(proof, artifacts)
    except Exception as exc:
        gates.setdefault("goal_lifecycle", {}).setdefault("errors", []).append(str(exc))
    try:
        _collect_self_goals(proof, artifacts)
    except Exception as exc:
        gates.setdefault("self_goals", {}).setdefault("errors", []).append(str(exc))

    # Belief consistency ledger: append snapshot capturing contradictions & goals history
    try:
        from brain.consistency.ledger import update_consistency_ledger as _update_consistency_ledger
        ledger_summary = _update_consistency_ledger(
            contradictions=proof.get("contradictions", {}),
            goals_summary=proof.get("consistency_goals"),
            window=200,
        )
        proof["consistency_ledger"] = {
            "latest_path": ledger_summary.get("latest_path"),
            "ledger_path": ledger_summary.get("ledger_path"),
            "stats": ledger_summary.get("stats"),
        }
        # Gate: if contradictions present, require ledger stats window >=1
        contradictions_total = int(proof.get("contradictions", {}).get("total") or 0)
        gates["consistency_ledger"] = {
            "ok": (contradictions_total == 0) or (ledger_summary.get("stats", {}).get("window", 0) >= 1),
            "contradictions_total": contradictions_total,
            "window": ledger_summary.get("stats", {}).get("window"),
        }

        # Contradiction drift gate: block if percent increase exceeds threshold
        stats = ledger_summary.get("stats") or {}
        delta_pct = float(stats.get("delta_pct") or 0.0)
        prev_c = stats.get("prev_contradictions")
        max_pct = float(os.getenv("PROOF_CONTRADICTION_MAX_DELTA_PCT", "1000") or 1000)
        drift_ok = True if prev_c is None else (delta_pct <= max_pct)
        gates["contradiction_drift"] = {
            "ok": drift_ok,
            "delta_pct": delta_pct,
            "prev": prev_c,
            "last": stats.get("last_contradictions"),
            "max_pct": max_pct,
        }

        # Export drift gauges into metrics for ops dashboards
        try:
            from brain.obs.metrics import (
                brain_consistency_contradiction_drift_pct,
                brain_consistency_contradictions_prev,
                brain_consistency_contradictions_last,
                brain_consistency_drift_alerts_total,
            )
            brain_consistency_contradiction_drift_pct.set(delta_pct)
            if prev_c is not None:
                brain_consistency_contradictions_prev.set(float(prev_c))
            last_c = stats.get("last_contradictions")
            if last_c is not None:
                brain_consistency_contradictions_last.set(float(last_c))
            # Emit alert when drift breaches the configured threshold
            if not drift_ok:
                try:
                    from brain.obs.enhanced_observability import Alert, AlertSeverity, metrics_registry
                    msg = (
                        f"Contradiction drift breach: delta_pct={delta_pct:.2f} exceeds max_pct={max_pct:.2f}"
                    )
                    alert = Alert(
                        name="consistency_contradiction_drift_breach",
                        severity=AlertSeverity.WARNING if delta_pct < (max_pct * 2.0) else AlertSeverity.CRITICAL,
                        message=msg,
                        timestamp=time.time(),
                        labels={
                            "component": "curriculum",
                            "gate": "contradiction_drift",
                        },
                        value=delta_pct,
                    )
                    metrics_registry.add_alert(alert)
                except Exception:
                    pass
                try:
                    brain_consistency_drift_alerts_total.inc(reason="contradiction_drift_breach")
                except Exception:
                    pass
        except Exception:
            # Metrics emission is best-effort; do not block proof bundle on metrics failures
            pass
    except Exception as exc:
        gates.setdefault("consistency_ledger", {}).setdefault("errors", []).append(str(exc))

    # Acknowledgement gate for contradiction drift breach
    try:
        drift_gate = gates.get("contradiction_drift") or {}
        breach = drift_gate.get("ok") is False
        delta_pct = drift_gate.get("delta_pct")
        prev_val = drift_gate.get("prev")
        if breach and prev_val is not None:
            # Load alerts to see if breach alert acknowledged
            alerts_path = artifacts / "ops" / "alerts.jsonl"
            records = _load_jsonl(alerts_path)
            # Filter for drift breach alerts
            breach_alerts = [r for r in records if r.get("name") == "consistency_contradiction_drift_breach"]
            acknowledged = [r for r in breach_alerts if r.get("acknowledged")]
            sign_key = resolve_alert_ack_signing_key()
            invalid_ack: List[Dict[str, Any]] = []
            if acknowledged:
                if sign_key:
                    invalid_ack = [rec for rec in acknowledged if not verify_alert_ack_signature(rec, sign_key)]
                else:
                    invalid_ack = acknowledged
            valid_ack_count = max(len(acknowledged) - len(invalid_ack), 0)
            if acknowledged:
                signature_ratio = valid_ack_count / float(len(acknowledged))
                if not sign_key:
                    signature_ratio = 0.0
            else:
                signature_ratio = 1.0
            # Compute age statistics for breach alerts (seconds)
            now_ts = time.time()
            ages = [max(0.0, now_ts - float(r.get("ts") or 0.0)) for r in breach_alerts]
            ages_sorted = sorted(ages)
            def _pctile(vals, frac):
                if not vals:
                    return 0.0
                if frac <= 0:
                    return vals[0]
                if frac >= 1:
                    return vals[-1]
                pos = frac * (len(vals) - 1)
                li = int(pos)
                ui = min(li + 1, len(vals) - 1)
                w = pos - li
                return vals[li] * (1.0 - w) + vals[ui] * w

            gate_payload = {
                "breach": True,
                "delta_pct": delta_pct,
                "prev": prev_val,
                "last": drift_gate.get("last"),
                "max_pct": drift_gate.get("max_pct"),
                "alerts_path": str(alerts_path),
                "acknowledged": len(acknowledged),
                "invalid_acknowledgements": len(invalid_ack),
                "valid_acknowledgements": valid_ack_count,
                "signature_checked": bool(sign_key),
                "signature_valid_ratio": signature_ratio,
                "breach_alert_ages_seconds": {
                    "count": len(ages_sorted),
                    "min": ages_sorted[0] if ages_sorted else 0.0,
                    "median": _pctile(ages_sorted, 0.5),
                    "p95": _pctile(ages_sorted, 0.95),
                    "max": ages_sorted[-1] if ages_sorted else 0.0,
                },
            }
            if breach_alerts:
                gate_payload["latest_breach_alerts"] = [
                    {
                        "alert_id": r.get("alert_id"),
                        "ts": r.get("ts"),
                        "ack_token": r.get("ack_token"),
                        "acknowledged": r.get("acknowledged"),
                    }
                    for r in breach_alerts[-5:]
                ]
            gate_payload["ok"] = valid_ack_count > 0

            # Emit stale-ack alert if unresolved breach alerts exceed max age threshold
            try:
                max_age_hours = float(os.getenv("PROOF_DRIFT_ACK_MAX_AGE_HOURS", "0") or 0.0)
            except Exception:
                max_age_hours = 0.0
            if max_age_hours > 0 and valid_ack_count == 0 and ages_sorted:
                oldest_age_hours = (ages_sorted[-1] / 3600.0)
                if oldest_age_hours >= max_age_hours:
                    try:
                        from brain.obs.enhanced_observability import Alert, AlertSeverity, metrics_registry
                        from brain.obs.metrics import brain_consistency_drift_alerts_total
                        alert = Alert(
                            name="consistency_contradiction_drift_ack_stale",
                            severity=AlertSeverity.WARNING,
                            message=f"Unacknowledged drift breach older than {max_age_hours}h (oldest={oldest_age_hours:.2f}h)",
                            timestamp=time.time(),
                            labels={"component": "curriculum", "gate": "contradiction_drift_ack"},
                            value=oldest_age_hours,
                        )
                        metrics_registry.add_alert(alert)
                        try:
                            brain_consistency_drift_alerts_total.inc(reason="drift_ack_stale")
                        except Exception:
                            pass
                    except Exception:
                        pass
            proof.setdefault("gates", {})["contradiction_drift_ack"] = gate_payload
        else:
            proof.setdefault("gates", {})["contradiction_drift_ack"] = {"breach": False, "ok": True}
    except Exception as exc:
        proof.setdefault("gates", {}).setdefault("contradiction_drift_ack", {}).setdefault("errors", []).append(str(exc))

    # Embed last consistency scheduler summary into the proof bundle for CI visibility
    try:
        sched_path = artifacts / "consistency" / "scheduler" / "last_run.json"
        if sched_path.exists():
            try:
                proof["consistency_scheduler"] = json.loads(sched_path.read_text(encoding="utf-8"))
                # also include the path reference for traceability
                proof["consistency_scheduler"]["path"] = str(sched_path)
            except Exception:
                pass
    except Exception:
        pass

    # If consult replay gating failed, propagate a blocking signal into the
    # world-model retrain execution gate so promotions reflect consult-driven
    # simulation failures. This attaches consult replay failure counts and
    # a short summary to the retrain execution gate for operator visibility.
    try:
        consult_gate = gates.get("consult_replays")
        retrain_gate = gates.get("world_model_retrain_execution")
        if consult_gate is not None and retrain_gate is not None:
            consult_ok = bool(consult_gate.get("ok"))
            consult_err = consult_gate.get("error")
            suite_missing = consult_err in {"missing_suite_directory", "missing_runs"}
            # Only block retrain execution if consult replays actually ran and failed,
            # not when the suite is absent in this environment.
            if (not consult_ok) and (not suite_missing):
                retrain_gate.setdefault("blocked_by_consult_replays", True)
                retrain_gate["ok"] = False
                retrain_gate.setdefault("block_reasons", []).append({
                    "reason": "consult_replays_failed",
                    "consult_failure_count": int(consult_gate.get("failure_count") or 0),
                    "consult_success_rate": consult_gate.get("success_rate"),
                })
                # Attach a lightweight consult replay summary into the retrain gate
                try:
                    consult_summary = proof.get("selector", {}).get("consult_replays")
                    if consult_summary:
                        retrain_gate.setdefault("consult_replay_summary", {}).update(
                            {
                                "generated_at": consult_summary.get("generated_at"),
                                "success_rate": consult_summary.get("success_rate"),
                                "failures_sample": consult_summary.get("failures", [])[:5],
                            }
                        )
                except Exception:
                    pass
    except Exception:
        # Non-fatal: do not interrupt proof bundle generation on best-effort propagation
        pass

    try:
        selector_summary = _collect_selector_decisions(proof, artifacts)
        if selector_summary:
            proof.setdefault("selector", {})["decisions"] = selector_summary
    except Exception:
        pass

    try:
        consult_summary = _collect_selector_consults(proof, artifacts)
        if consult_summary:
            proof.setdefault("selector", {})["consults"] = consult_summary
    except Exception:
        pass

    try:
        _curriculum_dashboard_gate(proof)
    except Exception as exc:
        gates.setdefault("curriculum_dashboard", {}).setdefault("errors", []).append(str(exc))

    try:
        _curriculum_alerts_gate(proof, artifacts)
    except Exception as exc:
        gates.setdefault("curriculum_alerts", {}).setdefault("errors", []).append(str(exc))

    # Include recent router snapshots (up to last 5)
    snaps_dir = artifacts / "learn" / "router_snaps"
    if snaps_dir.exists() and snaps_dir.is_dir():
        snaps = sorted(snaps_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
        for p in snaps[-5:]:
            try:
                meta = _add_file(proof, artifacts, p)
                proof["router_snapshots"][p.stem] = meta
            except Exception:
                continue

    # Include the latest SelfAssessment snapshot (if any)
    try:
        sa_roots = [artifacts]
        default_artifacts = Path("artifacts")
        try:
            if default_artifacts.resolve() != artifacts.resolve():
                sa_roots.append(default_artifacts)
        except Exception:
            sa_roots.append(default_artifacts)

        seen_snaps = set()
        snapshot_roots: list[str] = []
        snapshot_entries: list[str] = []
        for root in sa_roots:
            sa_dir = root / "self" / "assessments"
            if not sa_dir.exists() or not sa_dir.is_dir():
                continue
            snapshot_roots.append(str(sa_dir))
            snaps = sorted(sa_dir.glob("assessment_*.json"), key=lambda p: p.stat().st_mtime)
            for snap in snaps:
                name = snap.name
                if name in seen_snaps:
                    continue
                try:
                    _add_file(proof, artifacts, snap)
                    sha_path = snap.with_suffix(".sha256")
                    if sha_path.exists():
                        _add_file(proof, artifacts, sha_path)
                    seen_snaps.add(name)
                    snapshot_entries.append(name)
                except Exception:
                    continue
        if snapshot_roots:
            proof["self_snapshots"] = {
                "roots": snapshot_roots,
                "count": len(snapshot_entries),
                "entries": snapshot_entries,
            }
    except Exception:
        pass

    # Retrieval traces tail (up to last 20), helpful for CI attestation
    traces_path = artifacts / "retrieval" / "traces.jsonl"
    if traces_path.exists():
        try:
            lines = [ln.strip() for ln in traces_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
            tail = lines[-20:]
            proof["retrieval_traces_tail"] = []
            for ln in tail:
                try:
                    proof["retrieval_traces_tail"].append(json.loads(ln))
                except Exception:
                    proof["retrieval_traces_tail"].append({"raw": ln})
            # Provide a compact hash for the tail
            canonical = "\n".join(json.dumps(item, sort_keys=True, separators=(",", ":")) for item in proof["retrieval_traces_tail"])
            proof["retrieval_traces_tail_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
            # Also include file metadata for traces.jsonl
            _add_file(proof, artifacts, traces_path)
        except Exception:
            pass

    # Include a compact retrieval gate summary if available
    try:
        rscore = artifacts / "retrieval" / "scorecard.json"
        if rscore.exists():
            sc = json.loads(rscore.read_text(encoding="utf-8"))
            proof["retrieval_gate"] = {
                "latest": sc.get("latest"),
                "history_len": len(sc.get("history", [])) if isinstance(sc.get("history"), list) else None,
            }
    except Exception:
        pass

    # Include a compact eval gate summary if available (from eval harness scorecard)
    try:
        escore = artifacts / "eval" / "harness" / "scorecard.json"
        if escore.exists():
            sc = json.loads(escore.read_text(encoding="utf-8"))
            proof["gates"].setdefault("eval", sc.get("latest"))
    except Exception:
        pass

    # Attach drift tolerances if exported
    try:
        dtol = artifacts / "ci" / "drift_tolerances.json"
        if dtol.exists():
            try:
                proof["drift_tolerances"] = json.loads(dtol.read_text(encoding="utf-8"))
            except Exception:
                pass
    except Exception:
        pass

    # Include a compact predictor summary if available
    try:
        wmetrics = artifacts / "world" / "model_v1" / "metrics.json"
        if wmetrics.exists():
            wm = json.loads(wmetrics.read_text(encoding="utf-8"))
            proof["predictor"] = {
                "actions_seen": wm.get("actions_seen"),
                "top_actions": wm.get("top_actions"),
                "transitions": wm.get("transitions"),
                # Optional calibration metrics if present
                "eval_top1": wm.get("eval_top1"),
                "brier": wm.get("brier"),
                "nll": wm.get("nll"),
                "rmse_true": wm.get("rmse_true"),
                "calib_steps": wm.get("calib_steps"),
            }
    except Exception:
        pass

    # Include world gate summary (advantage over uniform baseline)
    try:
        wscore = artifacts / "world" / "scorecard.json"
        if wscore.exists():
            sc = json.loads(wscore.read_text(encoding="utf-8"))
            latest = sc.get("latest") or {}
            # Normalize expected metrics keys for schema checker
            metrics_norm = {}
            if "advantage" in latest:
                metrics_norm["advantage"] = latest.get("advantage")
            # Prefer eval_top1 if present else avg_top1
            if "eval_top1" in latest:
                metrics_norm["top1"] = latest.get("eval_top1")
            elif "avg_top1" in latest:
                metrics_norm["top1"] = latest.get("avg_top1")
            # Map rmse_true -> rmse
            if "rmse_true" in latest:
                metrics_norm["rmse"] = latest.get("rmse_true")
            for key in ("brier", "nll"):
                if key in latest:
                    metrics_norm[key] = latest.get(key)
            world_gate = {
                "passed": bool(latest.get("passed")),
                "metrics": metrics_norm,
                "raw": latest,  # retain raw for audit/debug
            }
            proof["gates"].setdefault("world", world_gate)
    except Exception:
        pass

    # Include planner A/B world gate summary
    try:
        wab = artifacts / "world" / "ab_scorecard.json"
        if wab.exists():
            sc = json.loads(wab.read_text(encoding="utf-8"))
            latest = sc.get("latest") or {}
            delta = latest.get("delta")
            if delta is None and all(k in latest for k in ("on_rate", "off_rate")):
                try:
                    delta = float(latest.get("on_rate") or 0) - float(latest.get("off_rate") or 0)
                except Exception:
                    delta = None
            ab_gate = {
                "passed": bool(latest.get("passed")),
                "metrics": {"delta": delta},
                "raw": latest,
            }
            proof["gates"].setdefault("world_ab", ab_gate)
    except Exception:
        pass

    # Include planning smoke predictor gate summary
    try:
        wplan = artifacts / "world" / "planning_ab_scorecard.json"
        if wplan.exists():
            sc = json.loads(wplan.read_text(encoding="utf-8"))
            proof["gates"].setdefault("world_planning_ab", sc.get("latest"))
    except Exception:
        pass

    # Include abstain gate summary
    # Include sandbox gate summary
    try:
        sscore = artifacts / "sandbox" / "scorecard.json"
        if sscore.exists():
            sc = json.loads(sscore.read_text(encoding="utf-8"))
            proof["gates"].setdefault("sandbox", sc.get("latest"))
    except Exception:
        pass

    try:
        sim_score = artifacts / "universe" / "sim" / "scorecard.json"
        if sim_score.exists():
            sc = json.loads(sim_score.read_text(encoding="utf-8"))
            proof["gates"].setdefault("sim", sc.get("latest"))
    except Exception:
        pass

    # Include novel sim gate summary (WM-only / generalization on novel cases)
    try:
        novel = artifacts / "universe" / "sim" / "novel_scorecard.json"
        if novel.exists():
            sc = json.loads(novel.read_text(encoding="utf-8"))
            proof["gates"].setdefault("sim_novel", sc.get("latest") or sc)
    except Exception:
        pass

    # Include sim episodes gate summary if available (enforces min episodes before advancing phases)
    try:
        sim_eps = artifacts / "universe" / "sim" / "episodes_scorecard.json"
        if sim_eps.exists():
            sc = json.loads(sim_eps.read_text(encoding="utf-8"))
            latest = sc.get("latest") or {}
            sim_eps_gate = {
                "passed": bool(latest.get("passed")),
                "metrics": {
                    # Flatten metrics for schema: episodes + steps (derive steps if absent)
                    "episodes": latest.get("episodes") or (latest.get("metrics") or {}).get("episodes"),
                    "steps": latest.get("steps") or (latest.get("metrics") or {}).get("steps"),
                },
                "raw": latest,
            }
            # Attach sim episodes scorecard file for schema checks
            try:
                _add_file(proof, artifacts, sim_eps)
            except Exception:
                pass
            proof["gates"].setdefault("sim_episodes", sim_eps_gate)
    except Exception:
        pass

    # Include budget gate summary (from CI scorecard)
    try:
        bpath = artifacts / "ci" / "budget_scorecard.json"
        if bpath.exists():
            sc = json.loads(bpath.read_text(encoding="utf-8"))
            # Normalize
            proof["gates"].setdefault(
                "budget",
                {
                    "passed": bool(sc.get("passed")),
                    "metrics": {
                        "limit": sc.get("limit"),
                        "charge_per_req": sc.get("charge_per_req"),
                        "blocked_status": sc.get("blocked_status"),
                        "blocked_error": sc.get("blocked_error"),
                    },
                },
            )
    except Exception:
        pass

    # Include circuit breaker gate summary (from CI scorecard)
    try:
        cpath = artifacts / "ci" / "circuit_breaker_scorecard.json"
        if cpath.exists():
            sc = json.loads(cpath.read_text(encoding="utf-8"))
            proof["gates"].setdefault(
                "circuit_breaker",
                {
                    "passed": bool(sc.get("passed")),
                    "metrics": {
                        "window_sec": sc.get("window_sec"),
                        "fails_threshold": sc.get("fails_threshold"),
                        "fails_sent": sc.get("fails_sent"),
                        "blocked_status": sc.get("blocked_status"),
                        "blocked_error": sc.get("blocked_error"),
                    },
                },
            )
    except Exception:
        pass

    try:
        sim_ab_candidates = []
        override_env = os.getenv("SIM_AB_SCORECARD")
        if override_env:
            override_path = Path(override_env)
            if not override_path.suffix:
                override_path = override_path / "planner_ab_scorecard.json"
            sim_ab_candidates.append(override_path)
        sim_ab_candidates.append(artifacts / "ab_gate" / "planner_ab_scorecard.json")
        sim_ab_candidates.append(artifacts / "universe" / "sim" / "planner_ab_scorecard.json")
        for sim_ab_score in sim_ab_candidates:
            if sim_ab_score.exists():
                sc = json.loads(sim_ab_score.read_text(encoding="utf-8"))
                proof["gates"].setdefault("sim_planner_ab", sc.get("latest"))
                break
    except Exception:
        pass

    try:
        prov = artifacts / "ops" / "provenance_scorecard.json"
        if prov.exists():
            sc = json.loads(prov.read_text(encoding="utf-8"))
            proof["gates"].setdefault("provenance", sc.get("latest") or sc)
    except Exception:
        pass

    # Include retrieval gate summary (latency p95 + jaccard)
    try:
        rcard = artifacts / "retrieval" / "scorecard.json"
        if rcard.exists():
            sc = json.loads(rcard.read_text(encoding="utf-8"))
            latest = sc.get("latest") or {}
            metrics = {}
            # Align with schema expectations: p95_ms, jaccard
            if "latency_p95_ms" in latest:
                metrics["p95_ms"] = latest.get("latency_p95_ms")
            if "jaccard_avg" in latest:
                metrics["jaccard"] = latest.get("jaccard_avg")
            # Ensure scorecard is attached in proof files for schema checks
            try:
                _add_file(proof, artifacts, rcard)
            except Exception:
                pass
            proof["gates"].setdefault(
                "retrieval",
                {
                    "passed": bool(latest.get("passed")),
                    "metrics": metrics,
                    "raw": latest,
                },
            )
    except Exception:
        pass

    # Include red-team gate summary and adversarial scorecards
    try:
        rts = artifacts / "redteam" / "scorecard.json"
        if rts.exists():
            sc = json.loads(rts.read_text(encoding="utf-8"))
            latest = sc.get("latest") or {}
            metrics_src = latest.get("metrics") if isinstance(latest.get("metrics"), dict) else latest
            redteam_gate = {
                "passed": bool(latest.get("passed")),
                "metrics": {
                    "incident_count": metrics_src.get("incident_count"),
                    "masked_bypasses_total": metrics_src.get("masked_bypasses_total"),
                    "masked_denials_total": metrics_src.get("masked_denials_total"),
                },
                "raw": latest,
            }
            # Attach redteam scorecard in proof files
            try:
                _add_file(proof, artifacts, rts)
            except Exception:
                pass
            proof["gates"]["redteam"] = redteam_gate
    except Exception:
        pass
    try:
        adv = artifacts / "redteam" / "adversarial_scorecard.json"
        if adv.exists():
            sc = json.loads(adv.read_text(encoding="utf-8"))
            latest = sc.get("latest") or sc
            adv_gate = {
                "passed": bool(latest.get("passed")),
                "metrics": {
                    "incident_count": (latest.get("metrics") or {}).get("incident_count", latest.get("incident_count")),
                    "masked_bypasses_total": (latest.get("metrics") or {}).get("masked_bypasses_total", latest.get("masked_bypasses_total")),
                    "masked_denials_total": (latest.get("metrics") or {}).get("masked_denials_total", latest.get("masked_denials_total")),
                },
                "raw": latest,
            }
            # Attach adversarial scorecard file
            try:
                _add_file(proof, artifacts, adv)
            except Exception:
                pass
            proof["gates"].setdefault("adversarial", adv_gate)
            # If no explicit redteam gate, populate from adversarial summary (compatible schema)
            proof["gates"].setdefault("redteam", adv_gate)
    except Exception:
        pass

    try:
        wabs = artifacts / "world" / "abstain_scorecard.json"
        if wabs.exists():
            sc = json.loads(wabs.read_text(encoding="utf-8"))
            latest = sc.get("latest") or {}
            abstain_gate = {
                "passed": bool(latest.get("passed")),
                "metrics": {
                    "unsafe_rate": latest.get("unsafe_rate"),
                    "cond_acc": latest.get("cond_acc"),
                },
                "raw": latest,
            }
            proof["gates"].setdefault("world_abstain", abstain_gate)
    except Exception:
        pass

    # Include self metrics snapshot for abstain tracking/CI attestation
    try:
        self_metrics_path = artifacts / "self" / "metrics.json"
        if self_metrics_path.exists():
            raw = json.loads(self_metrics_path.read_text(encoding="utf-8"))
            keys = (
                "unsafe_rate",
                "invalid_act_rate",
                "abstain_rate",
                "uncertainty",
                "consecutive_failures",
                "total_tasks",
                "updated_at",
                "calibration_ece",
                "calibration_samples",
                "calibration_coverage",
                "calibration_ece_ok",
                "calibration_ece_reason",
            )
            proof["self_metrics"] = {k: raw.get(k) for k in keys}
    except Exception:
        pass

    try:
        dscore = artifacts / "learn" / "discovery_scorecard.json"
        if dscore.exists():
            sc = json.loads(dscore.read_text(encoding="utf-8"))
            proof["gates"].setdefault("discovery", sc.get("latest"))
    except Exception:
        pass

    try:
        rap_score = artifacts / "rap" / "scorecard.json"
        if rap_score.exists():
            sc = json.loads(rap_score.read_text(encoding="utf-8"))
            proof["gates"].setdefault("rap", sc.get("latest"))
    except Exception:
        pass

    try:
        router_score = artifacts / "learn" / "router" / "scorecard.json"
        if router_score.exists():
            sc = json.loads(router_score.read_text(encoding="utf-8"))
            proof["gates"].setdefault("router", sc.get("latest"))
    except Exception:
        pass

    # Include new learning gate summaries if present
    try:
        tscore = artifacts / "learn" / "transfer_scorecard.json"
        if tscore.exists():
            sc = json.loads(tscore.read_text(encoding="utf-8"))
            proof["gates"].setdefault("transfer", sc.get("latest"))
    except Exception:
        pass

    try:
        cfscore = artifacts / "learn" / "concept_formation_scorecard.json"
        if cfscore.exists():
            sc = json.loads(cfscore.read_text(encoding="utf-8"))
            proof["gates"].setdefault("concept_formation", sc.get("latest"))
    except Exception:
        pass

    try:
        mscore = artifacts / "learn" / "meta_learning_scorecard.json"
        if mscore.exists():
            sc = json.loads(mscore.read_text(encoding="utf-8"))
            proof["gates"].setdefault("meta_learning", sc.get("latest"))
    except Exception:
        pass

    try:
        sgscore = artifacts / "learn" / "self_goals_scorecard.json"
        if sgscore.exists():
            sc = json.loads(sgscore.read_text(encoding="utf-8"))
            proof["gates"].setdefault("self_goals", sc.get("latest"))
    except Exception:
        pass

    # RBAC coverage gate (when available)
    if coverage_report:
        try:
            builtin = coverage_report.get("builtin", {})
            promotions = coverage_report.get("promotions", {})
            violations = []
            if isinstance(builtin, dict):
                violations.extend(builtin.get("violations") or [])
            if isinstance(promotions, dict):
                violations.extend(promotions.get("violations") or [])
            rbac_gate = {
                "passed": bool(coverage_report.get("ok")),
                "metrics": {
                    "builtin_sensitive": builtin.get("total_sensitive"),
                    "builtin_coverage": builtin.get("coverage"),
                    "promotions_sensitive": promotions.get("total_sensitive"),
                    "promotions_coverage": promotions.get("coverage"),
                    "violations": violations,
                },
                "report_path": str(coverage_path),
            }
            proof["gates"].setdefault("rbac", rbac_gate)
        except Exception:
            proof["gates"].setdefault("rbac", {"passed": False, "error": "invalid_coverage_report"})

    supply_chain_metrics = {}
    for fname in ("requirements.txt", "requirements-dev.txt", "requirements-test.txt"):
        p = Path(fname)
        meta = proof["files"].get(str(p)) if p.exists() else None
        if meta:
            supply_chain_metrics[f"{p.name}_sha256"] = meta.get("sha256")
    if supply_chain_metrics:
        proof["gates"].setdefault("supply_chain", {"passed": True, "metrics": supply_chain_metrics})

    # Include consolidated proof artifact if present (created by scripts/run_consolidator.py)
    try:
        consolidated = artifacts_dir / "proof" / "consolidated.jsonl"
        if consolidated.exists():
            _add_file(proof, artifacts_dir, consolidated)
            proof.setdefault("gates", {}).setdefault("consolidation", {"ok": True, "path": str(consolidated)})
    except Exception:
        # non-fatal
        pass

    # Include execution manifest if present
    try:
        manifest = artifacts_dir / "proof" / "execution_manifest.json"
        if manifest.exists():
            _add_file(proof, artifacts_dir, manifest)
            proof.setdefault("gates", {}).setdefault("execution_manifest", {"ok": True, "path": str(manifest)})
    except Exception:
        pass

    # Validate retrain validation digest formats (must be present and start with 'sha256:')
    try:
        wm = proof.get("world_model_retrain_validation")
        if isinstance(wm, dict):
            latest = wm.get("latest")
            # latest may be a dict referencing the per-run file entry
            digest_gate: Dict[str, object] = {"ok": True, "missing": []}
            stats_ok = True
            val_ok = True
            if isinstance(latest, dict):
                stats_digest = latest.get("stats_digest")
                validation_digest = latest.get("validation_digest")
                if not stats_digest or not isinstance(stats_digest, str) or not stats_digest.startswith("sha256:"):
                    stats_ok = False
                    digest_gate.setdefault("missing", []).append("stats_digest")
                if not validation_digest or not isinstance(validation_digest, str) or not validation_digest.startswith("sha256:"):
                    val_ok = False
                    digest_gate.setdefault("missing", []).append("validation_digest")
            else:
                # If latest is not a dict, mark as missing both
                stats_ok = False
                val_ok = False
                digest_gate.setdefault("missing", []).extend(["stats_digest", "validation_digest"])

            digest_gate["ok"] = bool(stats_ok and val_ok)
            proof["gates"].setdefault("retrain_digest_format", digest_gate)
    except Exception:
        # Non-fatal: if validation cannot run here, skip and let later gates capture issues
        pass

    # Aggregate a compact gates summary for quick green/red snapshot
    try:
        def _as_pass(v: object) -> Optional[bool]:
            if isinstance(v, dict):
                if "passed" in v:
                    return bool(v.get("passed"))
                if "ok" in v:
                    return bool(v.get("ok"))
            return None

        # Build a unified map of gates, including retrieval (which is stored separately)
        all_gates: Dict[str, object] = {}
        if isinstance(proof.get("gates"), dict):
            all_gates.update(proof["gates"])  # type: ignore[arg-type]
        # Inject retrieval as a first-class gate if available
        retrieval_latest = None
        if isinstance(proof.get("retrieval_gate"), dict):
            retrieval_latest = proof["retrieval_gate"].get("latest")
            if isinstance(retrieval_latest, dict):
                all_gates["retrieval"] = retrieval_latest

        # Resolve required gates from env override; otherwise default to all discovered gates
        required: Optional[List[str]] = None
        env_required = os.getenv("PROOF_REQUIRED_GATES")
        if env_required:
            try:
                parsed = json.loads(env_required)
                if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
                    required = list(parsed)
            except Exception:
                required = None
        if required is None:
            required = sorted(
                k for k, v in all_gates.items()
                if isinstance(v, dict) and ("passed" in v or "ok" in v)
            )

        # Compute per-gate pass/fail for required gates
        status: Dict[str, bool] = {}
        for k in required:
            v = all_gates.get(k)
            pv = _as_pass(v)
            status[k] = bool(pv) if pv is not None else False

        # Identify any other discovered failing gates not listed as required (defense-in-depth)
        non_required_failures = sorted(
            k for k, v in all_gates.items()
            if k not in status and isinstance(v, dict) and _as_pass(v) is False
        )

        all_pass = (len(status) > 0 and all(status.values())) and (len(non_required_failures) == 0)

        summary: Dict[str, object] = {
            "all_pass": all_pass,
            "required": required,
            "status": status,
            "non_required_failures": non_required_failures or None,
        }
        for gate_name in required:
            summary[gate_name] = status.get(gate_name, False)
        proof["gates_summary"] = summary

        # Late attachment: ensure sim episodes and redteam scorecard files present even if collected earlier failed
        try:
            sim_eps_final = artifacts / "universe" / "sim" / "episodes_scorecard.json"
            if sim_eps_final.exists():
                _add_file(proof, artifacts, sim_eps_final)
        except Exception:
            pass
        try:
            redteam_score_final = artifacts / "redteam" / "scorecard.json"
            adversarial_score = artifacts / "redteam" / "adversarial_scorecard.json"
            # Prefer canonical scorecard if present; otherwise attach adversarial with alias metadata
            if redteam_score_final.exists():
                _add_file(proof, artifacts, redteam_score_final)
            elif adversarial_score.exists():
                # Attach adversarial and record fallback_source on gate for provenance
                _add_file(proof, artifacts, adversarial_score)
                try:
                    gates = proof.get("gates", {})
                    if isinstance(gates, dict):
                        rt = gates.get("redteam")
                        if isinstance(rt, dict):
                            rt.setdefault("metadata", {})
                            rt["metadata"]["fallback_source"] = "adversarial_scorecard.json"
                except Exception:
                    pass
        except Exception:
            pass
    except Exception:
        pass

    try:
        semantic_payload = _build_semantic_self_model_payload(artifacts)
        semantic_name = f"semantic_self_model_{out_dir.name}.json"
        semantic_path = artifacts / "proof" / semantic_name
        semantic_path.parent.mkdir(parents=True, exist_ok=True)
        semantic_text = json.dumps(semantic_payload, indent=2, sort_keys=True)
        semantic_path.write_text(semantic_text, encoding="utf-8")
        semantic_meta = _add_file(proof, artifacts, semantic_path)
        summary: Dict[str, object] = {
            "artifact": str(semantic_path.resolve()),
            "capability_graph_hash": semantic_payload.get("capability_graph_hash"),
            "failure_summary_hash": semantic_payload.get("failure_summary_hash"),
            "determinism_token": semantic_payload.get("determinism_token"),
            "git_commit": semantic_payload.get("git_commit"),
        }
        if semantic_payload.get("tests_ran"):
            summary["tests_ran"] = semantic_payload["tests_ran"]
        signature_path = semantic_path.with_suffix(".sig")
        signature_value = _maybe_write_signature(
            semantic_text,
            signature_path=signature_path,
        )
        if signature_value:
            summary["signature"] = signature_value
            signature_meta = None
            try:
                signature_meta = _add_file(proof, artifacts, signature_path)
            except Exception:
                signature_meta = None
            if signature_meta:
                summary["signature_path"] = str(signature_path.resolve())
        proof["semantic_self_model"] = summary
    except Exception:
        # Semantic bundle generation should not block the main proof bundle
        pass

    # Planner invariants & retry metrics snapshot (non-fatal)
    try:
        from brain.obs.metrics import metrics_snapshot as _msnap
        snap = _msnap()
        invariants: Dict[str, object] = {}
        # Derive simple counts from metrics if any custom counters were added later; fallback to None
        # For now expose presence of semantic self-model + tool registry attempt info when available.
        # Attempt to load last attempts map from a running Brain instance (best-effort).
        last_attempts: Dict[str, int] = {}
        try:
            # If a Brain singleton accessor exists, use it (avoids importing heavy modules if absent)
            from brain.core.brain import Brain as _B
            b = None
            try:
                b = _B.get()
            except Exception:
                b = None
            reg = getattr(b, 'tool_registry', None)
            if reg is not None and hasattr(reg, 'get_last_attempts') and hasattr(reg, 'list_tools'):
                for spec in reg.list_tools():  # type: ignore[attr-defined]
                    try:
                        attempts = reg.get_last_attempts(spec.name)  # type: ignore[attr-defined]
                        if attempts is not None:
                            last_attempts[spec.name] = attempts
                    except Exception:
                        continue
        except Exception:
            pass
        invariants['tool_last_attempts'] = last_attempts or None
        invariants['semantic_metrics_present'] = bool(proof.get('semantic_self_model'))
        # Confidence clamping occurrences: scan proof events if present
        clamp_count = 0
        try:
            # If earlier steps recorded _invariant_notes, count them
            # This requires that tasks were executed before proof bundle generation.
            # We search any attached belief states or selector decisions for the token.
            rationale_events_sources = []
            selector_decisions = proof.get('selector', {}).get('decisions') if isinstance(proof.get('selector'), dict) else None
            if isinstance(selector_decisions, dict):
                rationale_events_sources.append(selector_decisions)
            # Fallback: no structured events, so skip
            for src in rationale_events_sources:
                try:
                    events = src.get('events') if isinstance(src, dict) else None
                    if isinstance(events, list):
                        for ev in events:
                            if isinstance(ev, dict) and ev.get('type') == 'step':
                                args = ev.get('args') if isinstance(ev.get('args'), dict) else {}
                                notes = args.get('_invariant_notes') if isinstance(args, dict) else []
                                if isinstance(notes, list):
                                    clamp_count += sum(1 for n in notes if n == 'confidence_clamped')
                except Exception:
                    continue
        except Exception:
            clamp_count = 0
        invariants['confidence_clamp_events'] = clamp_count
        # --- Planner invariant gates (optional) ---
        try:
            from brain.obs.metrics import metrics_snapshot as _ms
            snap = _ms()
            counters = snap.get('counters', {}) if isinstance(snap, dict) else {}
            def _counter_value(name: str) -> float:
                try:
                    samples = counters.get(name)
                    if isinstance(samples, list) and samples:
                        # single sample expected (no labels)
                        val = samples[0].get('value')
                        return float(val) if val is not None else 0.0
                except Exception:
                    return 0.0
                return 0.0
            steps_total = _counter_value('planner_steps_total')
            clamp_total = _counter_value('planner_confidence_clamped_total')
            early_abstain_total = _counter_value('planner_early_abstain_total')
            enabled = os.getenv('BRAIN_INVARIANTS_GATES_ENABLE', '0') == '1'
            clamp_rate_max = float(os.getenv('BRAIN_INVARIANTS_CLAMP_RATE_MAX', '0.05'))
            early_abstain_rate_max = float(os.getenv('BRAIN_INVARIANTS_EARLY_ABSTAIN_RATE_MAX', '0.02'))
            hard_fail = os.getenv('BRAIN_INVARIANTS_GATES_HARD_FAIL', '0') == '1'
            clamp_rate = (clamp_total / steps_total) if steps_total > 0 else 0.0
            early_abstain_rate = (early_abstain_total / steps_total) if steps_total > 0 else 0.0
            status = 'pass'
            violations: list[str] = []
            if enabled:
                if clamp_rate > clamp_rate_max:
                    status = 'fail'
                    violations.append('clamp_rate_exceeded')
                if early_abstain_rate > early_abstain_rate_max:
                    status = 'fail'
                    violations.append('early_abstain_rate_exceeded')
            gate_payload = {
                'enabled': enabled,
                'status': status,
                'rates': {
                    'clamp_rate': clamp_rate,
                    'early_abstain_rate': early_abstain_rate,
                },
                'thresholds': {
                    'clamp_rate_max': clamp_rate_max,
                    'early_abstain_rate_max': early_abstain_rate_max,
                },
                'raw_counts': {
                    'steps_total': steps_total,
                    'clamp_total': clamp_total,
                    'early_abstain_total': early_abstain_total,
                },
                'violations': violations or None,
            }
            try:
                token_src = json.dumps({
                    'status': status,
                    'steps_total': steps_total,
                    'clamp_total': clamp_total,
                    'early_abstain_total': early_abstain_total,
                    'clamp_rate': clamp_rate,
                    'early_abstain_rate': early_abstain_rate,
                    'thresholds': gate_payload['thresholds'],
                    'violations': violations,
                }, sort_keys=True, separators=(",", ":")).encode('utf-8')
                gate_payload['determinism_token'] = hashlib.sha256(token_src).hexdigest()
            except Exception:
                pass
            invariants['gates'] = gate_payload
            if enabled and hard_fail and status == 'fail':
                invariants.setdefault('gate_failure', True)
                proof.setdefault('gate_failures', []).append('planner_invariants')  # type: ignore[arg-type]
        except Exception:
            pass
        proof['planner_invariants'] = invariants
    except Exception:
        pass

    proof_path = out_dir / "proof.json"
    proof_json_text = json.dumps(proof, indent=2)
    proof_path.write_text(proof_json_text, encoding="utf-8")

    try:
        _add_file(proof, artifacts, proof_path)
    except Exception:
        pass

    # Also write stable copies for CI convenience
    try:
        stable = Path.cwd() / "proof.json"
        stable.write_text(proof_json_text, encoding="utf-8")
        _add_file(proof, artifacts, stable)
    except Exception:
        pass

    try:
        canonical = artifacts / "proof" / "proof.json"
        canonical.write_text(proof_json_text, encoding="utf-8")
        _add_file(proof, artifacts, canonical)
    except Exception:
        pass

    _maybe_write_signature(
        proof_json_text,
        signature_path=proof_path.parent / "proof.json.sig",
        copies=[Path.cwd() / "proof.json.sig", artifacts / "proof" / "proof.json.sig"],
    )

    print(str(proof_path))

    # Optionally make the proof bundle generation fatal when required gates fail.
    # Controlled by the environment flag PROOF_FAIL_ON_REQUIRED_GATES (off by default).
    try:
        # Avoid making tests brittle: when running under pytest, do not exit the process
        # even if PROOF_FAIL_ON_REQUIRED_GATES is set. Tests call `proof_bundle.main()` and
        # expect it to complete without terminating the test runner.
        fail_on_required = os.getenv("PROOF_FAIL_ON_REQUIRED_GATES", "0").lower() in {"1", "true", "yes", "on"}
        if "PYTEST_CURRENT_TEST" in os.environ:
            fail_on_required = False
        if fail_on_required:
            gs = proof.get("gates_summary") or {}
            all_pass = bool(gs.get("all_pass"))
            if not all_pass:
                # Print a short human-friendly summary to stderr for CI logs and exit non-zero
                failed_required = [k for k, v in (gs.get("status") or {}).items() if not v]
                sys.stderr.write(
                    f"PROOF_FAIL_ON_REQUIRED_GATES=1 and required gates failed: {failed_required}\n"
                )
                sys.exit(2)
    except Exception:
        # Do not mask the proof bundle itself if this check fails; let the normal flow continue
        pass


if __name__ == "__main__":
    main()
