#!/usr/bin/env python3
"""
Proof schema checker

Validates that proof.json contains required gates and minimal schema for each gate.
Optionally verifies that expected scorecard files are included in the proof files.

Usage:
  python tools/ci/proof_schema_check.py [--proof path] [--required gate1,gate2,...] [--require-scorecards 0|1]

Exit status is non-zero on validation failure.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Dict, List, Set


# Map gate key -> expected scorecard relative path inside artifacts
EXPECTED_SCORECARDS: Dict[str, str] = {
    "retrieval": "retrieval/scorecard.json",
    "world": "world/scorecard.json",
    "world_ab": "world/ab_scorecard.json",
    "world_planning_ab": "world/planning_ab_scorecard.json",
    "world_abstain": "world/abstain_scorecard.json",
    "sandbox": "sandbox/scorecard.json",
    "sim": "universe/sim/scorecard.json",
    "sim_episodes": "universe/sim/episodes_scorecard.json",
    "sim_planner_ab": "universe/sim/planner_ab_scorecard.json",
    "discovery": "learn/discovery_scorecard.json",
    "rap": "rap/scorecard.json",
    "router": "learn/router/scorecard.json",
    "self_goals": "learn/self_goals_scorecard.json",
    "goal_lifecycle": "goals/ledger.jsonl",
    "transfer": "learn/transfer_scorecard.json",
    "redteam": "redteam/scorecard.json",
    # CI guards
    "budget": "ci/budget_scorecard.json",
    "circuit_breaker": "ci/circuit_breaker_scorecard.json",
}


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--proof", default="proof.json", help="Path to proof.json")
    ap.add_argument(
        "--required",
        default="",
        help="Comma-separated list of required gate keys found in proof['gates']",
    )
    ap.add_argument(
        "--require-scorecards",
        type=int,
        default=1,
        help="If 1, require expected scorecard files to be present for known gates",
    )
    ap.add_argument(
        "--strict-metrics",
        type=int,
        default=0,
        help="If 1, enforce expected metric keys for known gates (error if missing)",
    )
    ap.add_argument(
        "--require-sbom",
        type=int,
        default=0,
        help="If 1, require SBOM freeze (artifacts/sbom/pip-freeze.txt) to be present in proof files",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    js = load_json(args.proof)
    gates = js.get("gates", {})
    files = js.get("files", {})

    required: Set[str] = set(filter(None, [g.strip() for g in args.required.split(",")]))

    errors: List[str] = []
    warnings: List[str] = []

    # Basic gates_summary presence
    gs = js.get("gates_summary")
    if not isinstance(gs, dict) or "all_pass" not in gs:
        errors.append("gates_summary missing or malformed (expected dict with 'all_pass')")

    # Expected metric keys per known gate
    expected_metric_keys: Dict[str, List[str]] = {
        "retrieval": ["p95_ms", "jaccard"],
        "world": ["advantage", "top1", "brier", "nll", "rmse"],
        "world_ab": ["delta"],
    "world_planning_ab": ["baseline_success_rate", "candidate_success_rate", "delta", "candidate_max_steps"],
        "world_abstain": ["unsafe_rate", "cond_acc"],
        "sandbox": ["p95_ms"],
        "sim": [
            "overall_success_rate",
            "ema_reward",
            "curriculum_success_rate",
            "generalization_success_rate",
            "planner_world_model_episodes",
            "planner_heuristic_episodes",
            "planner_world_model_steps",
            "planner_heuristic_steps",
            "planner_world_model_fallback_episodes",
        ],
        "sim_episodes": ["episodes", "steps"],
        "sim_planner_ab": [
            "heuristic_success_rate",
            "world_model_success_rate",
            "delta",
            "world_model_steps_total",
            "world_model_total_steps",
            "heuristic_steps_total",
            "fallback_episodes",
            "world_model_episode_usage",
        ],
        "discovery": ["delta_heldout", "discovery_accept_rate", "discovery_unsafe_rate"],
    "self_goals": ["focus_gain", "selected_count", "improvement", "episodes_run"],
    "transfer": ["transfer_delta", "ml_score", "math_acc", "language_acc", "nav_overall_success"],
        "rap": ["rap_success_rate", "effect_size", "p_value", "rap_count"],
        "router": [
            "candidate_success_rate",
            "baseline_success_rate",
            "success_delta",
            "calibration_rmse",
            "calibration_bias",
            "calibration_bias_abs",
            "p_value",
            "sample_size",
        ],
        "goal_lifecycle": ["total_goals", "open_goals"],
        "redteam": ["incident_count", "masked_bypasses_total", "masked_denials_total"],
        # CI guard metrics
        "budget": [
            "limit",
            "charge_per_req",
            "blocked_status",
            "blocked_error",
        ],
        "circuit_breaker": [
            "window_sec",
            "fails_threshold",
            "fails_sent",
            "blocked_status",
            "blocked_error",
        ],
    }

    # Check required gates exist and have minimal shape
    for gate in sorted(required):
        if gate not in gates:
            errors.append(f"required gate missing: {gate}")
            continue
        entry = gates.get(gate) or {}
        if not isinstance(entry, dict):
            errors.append(f"gate '{gate}' malformed: expected object, got {type(entry).__name__}")
            continue
        if "passed" not in entry:
            errors.append(f"gate '{gate}' missing 'passed' field")
        # Require at least one of 'metrics' or 'thresholds' containers
        if not any(k in entry for k in ("metrics", "thresholds")):
            warnings.append(
                f"gate '{gate}' has no 'metrics' or 'thresholds' field (continuing, but consider including metrics)"
            )

        # Scorecard inclusion check for known gates
        if args.require_scorecards:
            exp = EXPECTED_SCORECARDS.get(gate)
            if exp:
                if not any(exp in k for k in files.keys()):
                    errors.append(f"expected scorecard not found in proof files for gate '{gate}': {exp}")
            else:
                warnings.append(f"no expected scorecard configured for gate '{gate}' (skipping)")

        # Expected metrics keys
        if args.strict_metrics:
            exp_keys = expected_metric_keys.get(gate, [])
            got_metrics = entry.get("metrics") if isinstance(entry.get("metrics"), dict) else {}
            missing_metrics = [k for k in exp_keys if k not in got_metrics]
            if missing_metrics:
                errors.append(
                    f"gate '{gate}' missing expected metric keys: {', '.join(missing_metrics)}"
                )

    # SBOM presence
    if args.require_sbom:
        required_sbom = "artifacts/sbom/pip-freeze.txt"
        if not any(required_sbom in k for k in files.keys()):
            errors.append(f"SBOM freeze missing from proof files: {required_sbom}")

    # Emit report
    if warnings:
        print("proof_schema_check warnings:")
        for w in warnings:
            print(" -", w)

    if errors:
        print("proof_schema_check errors:")
        for e in errors:
            print(" -", e)
        return 1

    print("proof_schema_check: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
