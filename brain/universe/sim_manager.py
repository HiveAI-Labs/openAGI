from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from datetime import UTC, datetime
import json
import math
import os
from pathlib import Path
from threading import Lock
import time
from typing import Any

from brain.world_model.proof import write_retrain_validation_proof

from .sim_env import KeysDoorsEnv, RoomsPlusEnv


@dataclass
class MetricsSnapshot:
    modes: dict[str, dict[str, float]]
    ema_reward: float
    overall_success_rate: float
    updated_at: str
    episodes: int


ActionKey = tuple[str, str | None, str | None]


def _normalize_optional(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_action_name(action: str | None) -> str:
    return (action or "").strip()


def _action_key(action: str | None, target: str | None, item: str | None) -> ActionKey:
    name = _normalize_action_name(action)
    return (name, _normalize_optional(target), _normalize_optional(item))


def _action_signature(key: ActionKey) -> str:
    name, target, item = key
    return "|".join([name, target or "", item or ""])


def _requires_target(action: str) -> bool:
    return action in {"move", "unlock", "open"}


def _requires_item(action: str) -> bool:
    return action in {"pickup"}


def _action_has_args(action: str, target: str | None, item: str | None) -> bool:
    if _requires_target(action) and not target:
        return False
    if _requires_item(action) and not item:
        return False
    return True


def _action_payload(key: ActionKey, count: int, total: int) -> dict[str, Any]:
    name, target, item = key
    payload: dict[str, Any] = {
        "name": name,
        "count": int(count),
        "target": target,
        "item": item,
    }
    if total:
        payload["confidence"] = round(float(count) / float(total), 6)
    return payload


def _env_int(name: str, default: int) -> int:
    try:
        raw = os.getenv(name)
        if raw is None or raw.strip() == "":
            return default
        return int(float(raw))
    except Exception:
        return default


# --- World Model v2: featureized policy and tiny planner ---

_ROOMS = ("start", "hall", "storage", "vault")

def _bool(val: Any) -> bool:
    return bool(val)

def _safe_len(x: Any) -> int:
    try:
        return len(x)
    except Exception:
        return 0

def _feature_key_from_state(state: dict[str, Any]) -> tuple[str, bool, bool, bool, bool, int, int]:
    """Return a normalized, compact feature key for WM v2.

    Schema: (room, has_key, door_unlocked, door_open, key_visible, steps_mod2, exits_count)
    """
    room = str(state.get("room", ""))
    inv = state.get("inventory") or []
    has_key = "key" in list(inv) if isinstance(inv, (list, tuple, set)) else False
    door = state.get("door", {}) if isinstance(state.get("door"), dict) else {}
    unlocked = _bool(door.get("unlocked"))
    open_flag = _bool(door.get("open"))
    visible_items = state.get("visible_items") or []
    try:
        key_visible = "key" in list(visible_items)
    except Exception:
        key_visible = False
    steps = int(state.get("steps", 0) or 0)
    steps_mod2 = steps % 2
    exits = state.get("exits") or []
    exits_count = _safe_len(exits)
    return (room, has_key, unlocked, open_flag, key_visible, steps_mod2, exits_count)

def _serialize_feat_key(key: tuple[str, bool, bool, bool, bool, int, int]) -> str:
    room, has_key, unlocked, open_flag, key_visible, steps_mod2, exits_count = key
    return f"room={room};hk={int(has_key)};u={int(unlocked)};o={int(open_flag)};kv={int(key_visible)};s2={steps_mod2};xc={exits_count}"

def _parse_signature(sig: str) -> ActionKey:
    parts = sig.split("|")
    name = (parts[0] if parts else sig).strip()
    target = (parts[1].strip() or None) if len(parts) > 1 else None
    item = (parts[2].strip() or None) if len(parts) > 2 else None
    return (name, target, item)


def _format_state_summary(
    state_key: tuple[str, str, bool, bool, bool, bool],
    counts: Counter[ActionKey],
) -> dict[str, Any]:
    if len(state_key) == 6:
        mode, room, unlocked, open_flag, has_key, key_visible = state_key
    else:
        mode, room, unlocked, open_flag, has_key = state_key  # legacy payloads
        key_visible = False
    total = int(sum(counts.values()))
    entry: dict[str, Any] = {
        "mode": mode,
        "room": room,
        "door_unlocked": bool(unlocked),
        "door_open": bool(open_flag),
        "has_key": bool(has_key),
    "key_visible": bool(key_visible),
        "total": total,
        "actions": [],
        "counts": {},
        "top_action": None,
        "top_action_signature": None,
    }
    if counts:
        top_key, top_count = counts.most_common(1)[0]
        entry["top_action"] = _action_payload(top_key, top_count, total)
        entry["top_action_signature"] = _action_signature(top_key)
    entry["actions"] = [_action_payload(k, c, total) for k, c in counts.most_common()]
    entry["counts"] = {_action_signature(k): int(v) for k, v in counts.items()}
    return entry


def _heuristic_decision(state: dict[str, Any], visited: set[str], env: KeysDoorsEnv | None = None) -> tuple[str, str | None, str | None]:
    room = state.get("room")
    inventory = set(state.get("inventory") or [])
    door = state.get("door", {}) if isinstance(state.get("door"), dict) else {}
    visible_items = state.get("visible_items") or []
    exits = state.get("exits") or []
    mode = str(state.get("mode", "curriculum"))

    if "key" in inventory:
        if room != "hall":
            return ("move", "hall", None)
        if not door.get("unlocked"):
            return ("unlock", "vault", None)
        if not door.get("open"):
            return ("open", "vault", None)
        return ("move", "vault", None)

    if mode == "generalization_plus" and "key" not in visible_items:
        # If hidden key mode and we're in the key room, actively search
        try:
            if env is not None and getattr(env, "hidden_key_mode", False) and room == getattr(env, "key_room", "storage"):
                return ("search", None, None)
        except Exception:
            pass
        # Otherwise bias navigation toward storage as a likely key location
        if room != "storage" and "storage" in exits:
            target = "storage"
            visited.add(target)
            return ("move", target, None)
        if room == "storage" and "hall" in exits:
            visited.add("hall")
            return ("move", "hall", None)

    if "key" in visible_items:
        return ("pickup", None, "key")

    target = "hall"
    for candidate in exits:
        if candidate == "vault":
            continue
        if candidate not in visited and candidate != room:
            target = candidate
            break
    if target == room or target not in exits:
        for candidate in exits:
            if candidate != room:
                target = candidate
                break
    visited.add(target)
    return ("move", target, None)


def _current_subgoal(state: dict[str, Any], env: KeysDoorsEnv | None = None) -> str:
    """Return the next unmet subgoal for recovery planning.

    Subgoals: acquire_key -> reach_hall -> unlock -> open -> enter_vault
    """
    room = state.get("room")
    inventory = set(state.get("inventory") or [])
    door = state.get("door", {}) if isinstance(state.get("door"), dict) else {}
    if "key" not in inventory:
        # If we're not seeing the key and not holding it, acquiring key is next
        return "acquire_key"
    if room != "hall":
        return "reach_hall"
    if not door.get("unlocked"):
        return "unlock"
    if not door.get("open"):
        return "open"
    return "enter_vault"


_CACHE_LOCK = Lock()
_SIM_CACHE: dict[tuple[str, str], WorkspaceSim] = {}


def get_workspace_sim(workspace: str, artifacts_dir: Path) -> WorkspaceSim:
    key = (workspace, str(artifacts_dir))
    with _CACHE_LOCK:
        sim = _SIM_CACHE.get(key)
        if sim is None:
            sim = WorkspaceSim(workspace=workspace, artifacts_dir=artifacts_dir)
            _SIM_CACHE[key] = sim
        return sim


def reset_cache() -> None:
    with _CACHE_LOCK:
        _SIM_CACHE.clear()


class WorkspaceSim:
    def __init__(self, workspace: str, artifacts_dir: Path) -> None:
        self.workspace = workspace
        self.artifacts_dir = artifacts_dir
        # Select environment variant via env var
        try:
            import os as _os
            _env_choice = (_os.getenv("BRAIN_SIM_ENV") or "base").strip().lower()
        except Exception:
            _env_choice = "base"
        if _env_choice in {"rooms_plus", "rooms+", "plus"}:
            self._env = RoomsPlusEnv()
        else:
            self._env = KeysDoorsEnv()
        self._mode = "curriculum"
        self.ws_dir = artifacts_dir / "ws" / workspace / "sim"
        self.ws_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.ws_dir / "state.json"
        self.episodes_path = artifacts_dir / "universe" / "sim" / "episodes.jsonl"
        self.metrics_path = artifacts_dir / "universe" / "sim" / "metrics.json"
        self.scorecard_path = artifacts_dir / "universe" / "sim" / "scorecard.json"
        self.episodes_path.parent.mkdir(parents=True, exist_ok=True)
        self._episodes_trim_lines = max(0, _env_int("BRAIN_SIM_EPISODES_MAX_LINES", 0))
        self._episodes_trim_interval = max(1, _env_int("BRAIN_SIM_EPISODES_TRIM_INTERVAL", 64))
        self._episodes_append_counter = 0
        self._load_state()

    def plan_with_world_model(
        self,
        state: dict[str, Any],
        *,
        goal_room: str | None = None,
        depth: int | None = None,
        beam: int | None = None,
    ) -> dict[str, Any]:
        """Return a tiny WM v2 plan from the given state.

        Uses the persisted world_model_v2.json if present. Honors the same
        environment knobs as run_scripted_episode:
        - BRAIN_WM_PLANNER_{DEPTH,BEAM,PRIORS,LOOP_PENALTY,HEURISTIC_SCALE}
        - BRAIN_WM_V2_LAPLACE_ALPHA
        """
        wm_v2 = self.load_world_model_v2()
        if not isinstance(wm_v2, dict) or not wm_v2:
            # Optional neural fallback: attempt a 1-step neural prediction
            try:
                import os as _os
                _neural_on = (_os.getenv("BRAIN_WM_NEURAL") or "").strip().lower() in {"1","true","yes","on"}
            except Exception:
                _neural_on = False
            if _neural_on:
                # Ask env for allowed signatures
                allowed = None
                try:
                    allowed = _allowed_action_sigs_from_env()
                except Exception:
                    allowed = None
                sig = self._neural_predict_action_sig(state, allowed)
                if sig:
                    name, target, item = _parse_signature(sig)
                    return {"ok": True, "action_signature": sig, "action": name, "target": target, "item": item, "path": [sig], "source": "neural"}
            return {"ok": False, "error": "no_v2_model"}
        try:
            import os as _os
            _alpha = float(wm_v2.get("alpha", 1.0))
            max_depth = depth if depth is not None else int(float(_os.getenv("BRAIN_WM_PLANNER_DEPTH") or "3"))
            max_depth = max(1, max_depth)
            beam_w = beam if beam is not None else int(float(_os.getenv("BRAIN_WM_PLANNER_BEAM") or "3"))
            beam_w = max(1, beam_w)
            _priors_on = ((_os.getenv("BRAIN_WM_PLANNER_PRIORS") or "1").strip().lower() in {"1","true","yes","on"})
            _loop_penalty = float((_os.getenv("BRAIN_WM_PLANNER_LOOP_PENALTY") or "0.3").strip())
            _hscale = float((_os.getenv("BRAIN_WM_PLANNER_HEURISTIC_SCALE") or "1.0").strip())
        except Exception:
            _alpha = float(wm_v2.get("alpha", 1.0))
            max_depth = 3
            beam_w = 3
            _priors_on = True
            _loop_penalty = 0.3
            _hscale = 1.0

        if goal_room is None:
            goal_room = getattr(self._env, "goal_room", "vault")

        policy = wm_v2.get("policy") or {}
        transitions = wm_v2.get("transitions") or {}
        if not isinstance(policy, dict) or not isinstance(transitions, dict):
            return {"ok": False, "error": "incomplete_v2_model"}

        start_fk = _serialize_feat_key(_feature_key_from_state(state))

        def best_actions_for_fk(fk: str):
            e = policy.get(f"f:{fk}")
            if not isinstance(e, dict):
                counts = {}
                total = 0
            else:
                counts = e.get("counts") if isinstance(e, dict) else None
                total = int(e.get("total", 0))
            if not isinstance(counts, dict):
                counts = {}
            if _priors_on:
                flags = {p.split("=",1)[0]: p.split("=",1)[1] for p in fk.split(";") if "=" in p}
                room = flags.get("room", "")
                has_key = flags.get("hk", "0") == "1"
                unlocked = flags.get("u", "0") == "1"
                open_flag = flags.get("o", "0") == "1"
                key_visible = flags.get("kv", "0") == "1"
                # Generic exploration priors
                if not has_key and not key_visible:
                    counts.setdefault("search||", 0)
                    counts["search||"] += int(max(1, _alpha))
                if key_visible and not has_key:
                    counts.setdefault("pickup||key", 0)
                    counts["pickup||key"] += int(max(1, 2 * _alpha))
                # Goal-aware at the hub
                if room == "hall":
                    if goal_room == "vault":
                        if not unlocked:
                            counts.setdefault("unlock|vault|", 0)
                            counts["unlock|vault|"] += int(max(1, 2 * _alpha))
                        elif not open_flag:
                            counts.setdefault("open|vault|", 0)
                            counts["open|vault|"] += int(max(1, 2 * _alpha))
                        else:
                            counts.setdefault("move|vault|", 0)
                            counts["move|vault|"] += int(max(1, _alpha))
                        if not has_key:
                            counts.setdefault("move|storage|", 0)
                            counts["move|storage|"] += int(max(1, _alpha))
                    else:
                        counts.setdefault(f"move|{goal_room}|", 0)
                        # Strongly bias direct move to the goal from the hub when goal != vault
                        counts[f"move|{goal_room}|"] += int(max(3, 5 * _alpha))
                # Move back to hub from leaf rooms (except goal room)
                if room not in ("hall", goal_room):
                    counts.setdefault("move|hall|", 0)
                    counts["move|hall|"] += int(max(1, _alpha))
            k = len(counts)
            denom = float(total + (_alpha * k)) if k else 1.0
            pairs = []
            for sig, c in counts.items():
                try:
                    c_int = int(c)
                except Exception:
                    c_int = 0
                p = (c_int + _alpha) / denom
                pairs.append((sig, p))
            pairs.sort(key=lambda x: x[1], reverse=True)
            # Ensure goal-directed move from the hub is considered when goal != vault
            try:
                flags = {p.split("=",1)[0]: p.split("=",1)[1] for p in fk.split(";") if "=" in p}
                room = flags.get("room", "")
            except Exception:
                room = ""
            if room == "hall" and goal_room != "vault":
                sig_goal = f"move|{goal_room}|"
                if all(s != sig_goal for s,_ in pairs):
                    # compute an approximate prob for insertion
                    c_val = counts.get(sig_goal, 0)
                    try:
                        c_int = int(c_val)
                    except Exception:
                        c_int = 0
                    p_ins = (c_int + _alpha) / (float(total + (_alpha * max(1, len(counts)))))
                    pairs.append((sig_goal, p_ins))
                    pairs.sort(key=lambda x: x[1], reverse=True)
            return pairs[:beam_w]

        def room_from_fk(fk: str) -> str:
            try:
                for part in fk.split(";"):
                    if part.startswith("room="):
                        return part.split("=", 1)[1]
            except Exception:
                pass
            return ""

        def flags_from_fk(fk: str) -> dict[str, Any]:
            out = {"room": room_from_fk(fk), "has_key": False, "unlocked": False, "open": False, "key_visible": False}
            try:
                for part in fk.split(";"):
                    if part.startswith("hk="):
                        out["has_key"] = (part.split("=",1)[1] == "1")
                    elif part.startswith("u="):
                        out["unlocked"] = (part.split("=",1)[1] == "1")
                    elif part.startswith("o="):
                        out["open"] = (part.split("=",1)[1] == "1")
                    elif part.startswith("kv="):
                        out["key_visible"] = (part.split("=",1)[1] == "1")
            except Exception:
                pass
            return out

        def heuristic_bonus(fk: str) -> float:
            f = flags_from_fk(fk)
            bonus = 0.0
            # Reward being at the goal
            if f.get("room") == goal_room:
                bonus += 1.0
            # Only apply door progression shaping when vault is the target
            if goal_room == "vault":
                if f.get("has_key"):
                    bonus += 0.7
                if f.get("unlocked"):
                    bonus += 0.3
                if f.get("open"):
                    bonus += 0.3
                if f.get("key_visible") and not f.get("has_key"):
                    bonus += 0.1
            return _hscale * bonus

        Node = tuple[str, float, list[str], tuple[str, ...]]
        frontier: list[Node] = [(start_fk, 0.0, [], (start_fk,))]
        for _ in range(max_depth):
            new_frontier: list[Node] = []
            for fk, score, path, visited_fks in frontier:
                if room_from_fk(fk) == goal_room:
                    first = path[0] if path else None
                    name, target, item = _parse_signature(first) if first else (None, None, None)  # type: ignore
                    return {
                        "ok": True,
                        "action_signature": first,
                        "action": name,
                        "target": target,
                        "item": item,
                        "path": path,
                    }
                for sig, p in best_actions_for_fk(fk):
                    tentry = transitions.get(f"f:{fk}|{sig}")
                    if not isinstance(tentry, dict):
                        continue
                    nexts = tentry.get("next") if isinstance(tentry.get("next"), dict) else None
                    if not nexts:
                        continue
                    for nfk, cnt in sorted(nexts.items(), key=lambda kv: int(kv[1]) if kv[1] is not None else 0, reverse=True)[:3]:
                        step_cost = -math.log(max(p, 1e-6))
                        hcost = -heuristic_bonus(nfk)
                        loop_cost = _loop_penalty if nfk in visited_fks else 0.0
                        new_score = score + step_cost + hcost + loop_cost
                        new_frontier.append((nfk, new_score, path + [sig], tuple(list(visited_fks) + [nfk])))
            if not new_frontier:
                break
            new_frontier.sort(key=lambda n: n[1])
            frontier = new_frontier[:beam_w]
        # fallback: 1-step pick
        first = (best_actions_for_fk(start_fk) or [(None, 0.0)])[0][0]
        if first:
            name, target, item = _parse_signature(first)
            return {"ok": True, "action_signature": first, "action": name, "target": target, "item": item, "path": [first]}
        # Neural fallback if planner couldn't propose
        try:
            import os as _os
            _neural_on = (_os.getenv("BRAIN_WM_NEURAL") or "").strip().lower() in {"1","true","yes","on"}
        except Exception:
            _neural_on = False
        if _neural_on:
            allowed = None
            try:
                allowed = _allowed_action_sigs_from_env()
            except Exception:
                allowed = None
            sig = self._neural_predict_action_sig(state, allowed)
            if sig:
                n, t, it = _parse_signature(sig)
                return {"ok": True, "action_signature": sig, "action": n, "target": t, "item": it, "path": [sig], "source": "neural"}
        return {"ok": False, "error": "no_plan"}

    def reset(self, mode: str = "curriculum", seed: int | None = None) -> dict[str, Any]:
        obs = self._env.reset(mode=mode, seed=seed)
        self._mode = mode or "curriculum"
        self._persist_state()
        self._append_event(
            {
                "event": "reset",
                "mode": self._mode,
                "seed": seed,
                "state": obs["state"],
            },
        )
        return obs

    def observe(self) -> dict[str, Any]:
        obs = self._env.observe()
        self._persist_state()
        self._append_event({"event": "observe", "mode": self._mode, "state": obs["state"]})
        return obs

    def act(self, action: str, *, target: str | None = None, item: str | None = None) -> dict[str, Any]:
        result = self._env.step(action, target=target, item=item)
        self._persist_state()
        # Capture and clear optional alternative flag set by planner
        try:
            was_alt = bool(getattr(self, "_action_was_alternative", False))
        except Exception:
            was_alt = False
        try:
            desired_for_alt = getattr(self, "_desired_signature_for_alt", None)
            if desired_for_alt is not None:
                desired_for_alt = str(desired_for_alt)
        except Exception:
            desired_for_alt = None
        try:
            if hasattr(self, "_action_was_alternative"):
                delattr(self, "_action_was_alternative")
            if hasattr(self, "_desired_signature_for_alt"):
                delattr(self, "_desired_signature_for_alt")
        except Exception:
            pass
        # If an alternative was executed, bump a masking metric counter
        if was_alt:
            try:
                raw = self._read_metrics_dict()
                if not isinstance(raw, dict):
                    raw = {}
                md = raw.setdefault("masking", {})
                md["alternative_actions_recorded"] = int(md.get("alternative_actions_recorded", 0) or 0) + 1
                self.metrics_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception:
                pass
        entry = {
            "event": "act",
            "mode": self._mode,
            "action": action,
            "target": target,
            "item": item,
            "state": result["state"],
            "reward": result["reward"],
            "done": result["done"],
            "success": result["success"],
            "message": result["message"],
            "action_was_alternative": was_alt,
            "desired_signature": desired_for_alt,
        }
        self._append_event(entry)
        if result["done"]:
            self._update_metrics(success=bool(result["success"]), reward=float(result["reward"]))
        return result

    def write_scorecard(self, thresholds: dict[str, float], extra: dict[str, Any] | None = None) -> dict[str, Any]:
        metrics = self._read_metrics()
        modes = metrics.modes
        threshold_floats = {k: float(v) for k, v in thresholds.items()}
        entry_metrics: dict[str, Any] = {
            "overall_success_rate": float(metrics.overall_success_rate),
            "ema_reward": float(metrics.ema_reward),
        }
        entry_modes: dict[str, Any] = {}
        passes = True
        # Include metrics for all observed modes so proof drift has stable keys.
        observed_modes = set(modes.keys()) | set(threshold_floats.keys())
        for mode in sorted(observed_modes):
            stats = modes.get(mode, {})
            rate = float(stats.get("success_rate", 0.0))
            count = int(stats.get("count", 0))
            entry_metrics[f"{mode}_success_rate"] = rate
            entry_metrics[f"{mode}_count"] = count
            threshold = threshold_floats.get(mode)
            entry_modes[mode] = {
                "success_rate": rate,
                "count": count,
                "threshold": threshold,
            }
            if threshold is not None and rate < threshold:
                passes = False

        raw_dict = self._read_metrics_dict()
        world_section = raw_dict.get("world_model") if isinstance(raw_dict, dict) else None
        if isinstance(world_section, dict):
            top1 = world_section.get("top1_accuracy")
            samples = world_section.get("samples")
            if top1 is not None:
                entry_metrics["world_model_top1"] = float(top1)
            if samples is not None:
                entry_metrics["world_model_samples"] = int(samples)

        planner_section = raw_dict.get("planner_usage") if isinstance(raw_dict, dict) else None
        planner_section_payload: dict[str, Any] | None = None
        if isinstance(planner_section, dict):
            episodes_info = planner_section.get("episodes") or {}
            steps_info = planner_section.get("steps") or {}
            try:
                entry_metrics["planner_world_model_episodes"] = int(episodes_info.get("world_model", 0))
            except (TypeError, ValueError):
                entry_metrics["planner_world_model_episodes"] = 0
            try:
                entry_metrics["planner_heuristic_episodes"] = int(episodes_info.get("heuristic", 0))
            except (TypeError, ValueError):
                entry_metrics["planner_heuristic_episodes"] = 0
            try:
                entry_metrics["planner_world_model_steps"] = int(steps_info.get("world_model", 0))
            except (TypeError, ValueError):
                entry_metrics["planner_world_model_steps"] = 0
            try:
                entry_metrics["planner_heuristic_steps"] = int(steps_info.get("heuristic", 0))
            except (TypeError, ValueError):
                entry_metrics["planner_heuristic_steps"] = 0
            fallbacks_info = planner_section.get("fallbacks") or {}
            try:
                entry_metrics["planner_world_model_fallback_episodes"] = int(fallbacks_info.get("count", 0))
            except (TypeError, ValueError):
                entry_metrics["planner_world_model_fallback_episodes"] = 0
            planner_section_payload = planner_section

        entry: dict[str, Any] = {
            "ts": datetime.now(UTC).isoformat(),
            "updated_at": metrics.updated_at,
            "passed": passes,
            "thresholds": threshold_floats,
            "metrics": entry_metrics,
            "modes": entry_modes,
        }
        if planner_section_payload is not None:
            entry["planner_usage"] = planner_section_payload
        if isinstance(world_section, dict):
            entry["world_model"] = world_section
        if extra:
            entry.update(extra)

        card: dict[str, Any] = {"history": []}
        if self.scorecard_path.exists():
            try:
                current = json.loads(self.scorecard_path.read_text(encoding="utf-8"))
                if isinstance(current, dict):
                    card = current
            except Exception:
                card = {"history": []}

        history = card.get("history")
        if not isinstance(history, list):
            history = []
        history.append(entry)
        card["history"] = history[-25:]
        card["latest"] = entry
        card["thresholds"] = threshold_floats

        self.scorecard_path.write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")
        return card

    def record_world_model_metrics(
        self,
        stats: dict[str, Any],
        *,
        ema_reward: float | None = None,
        overall_success_rate: float | None = None,
    ) -> dict[str, Any]:
        raw = self._read_metrics_dict()
        if ema_reward is not None:
            raw["ema_reward"] = float(ema_reward)
        if overall_success_rate is not None:
            raw["overall_success_rate"] = float(overall_success_rate)
        world_section = raw.get("world_model")
        if not isinstance(world_section, dict):
            world_section = {}
        world_section.update(stats)
        world_section["updated_at"] = datetime.now(UTC).isoformat()
        raw["world_model"] = world_section
        raw["updated_at"] = world_section["updated_at"]
        self.metrics_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
        return raw

    def retrain_world_model(self) -> dict[str, Any]:
        # Local progress iterator: wraps iterable with tqdm when available; otherwise returns as-is.
        def _piter(iterable, desc: str | None = None, total: int | None = None):
            try:
                from tqdm import tqdm  # type: ignore
                return tqdm(iterable, desc=desc, total=total, leave=False)
            except Exception:
                return iterable

        events = self._load_workspace_events()
        dataset_all, episodes = self._build_dataset(events)
        # Select training dataset: by default, prefer successful episodes only.
        # When BRAIN_WM_TRAIN_ON_ALL=1, include all transitions (success and failure).
        try:
            import os as _os
            _train_all = (_os.getenv("BRAIN_WM_TRAIN_ON_ALL") or "").strip().lower() in {"1","true","yes","on"}
        except Exception:
            _train_all = False
        if _train_all:
            dataset = dataset_all
        else:
            training_dataset: list[dict[str, Any]] = []
            for episode in episodes:
                transitions = episode.get("transitions", [])
                if not transitions:
                    continue
                if bool(transitions[-1].get("success")):
                    training_dataset.extend(transitions)
            dataset = training_dataset if training_dataset else dataset_all
        state_counts: dict[tuple[str, str, bool, bool, bool, bool], Counter[ActionKey]] = defaultdict(Counter)

        # Optional: apply action-quality weighting for classic (v1) model too
        try:
            import os as _os
            _quality_on = (_os.getenv("BRAIN_WM_ACTION_QUALITY") or "1").strip().lower() in {"1","true","yes","on"}
        except Exception:
            _quality_on = True
        def _v1_quality_weight(prev: dict[str, Any], sig: str, nxt: dict[str, Any]) -> int:
            if not _quality_on:
                return 1
            try:
                name, target, item = _parse_signature(sig)
                room = str(prev.get("room", ""))
                if name == "move" and target == room:
                    return 0
                if room == "hall" and name == "move" and target == "start":
                    return 0
                # Reward progress by doubling counts
                if _progress(prev, nxt):
                    return 2
            except Exception:
                return 1
            return 1
        for sample in _piter(dataset, desc="WM v1: count policy/transitions", total=len(dataset) if isinstance(dataset, list) else None):
            key = self._state_key(sample["mode"], sample["state"])
            ak = _action_key(sample.get("action"), sample.get("target"), sample.get("item"))
            if ak[0]:
                sig = _action_signature(ak)
                w = _v1_quality_weight(sample.get("state") or {}, sig, sample.get("next_state") or {})
                if w > 0:
                    state_counts[key][ak] += w

        total_samples = len(dataset)
        correct = 0
        mode_accum: dict[str, dict[str, int]] = defaultdict(lambda: {"samples": 0, "correct": 0})
        for sample in _piter(dataset, desc="WM v1: evaluate top1", total=len(dataset) if isinstance(dataset, list) else None):
            key = self._state_key(sample["mode"], sample["state"])
            counts = state_counts[key]
            if not counts:
                continue
            prediction = counts.most_common(1)[0][0]
            mode_accum[sample["mode"]]["samples"] += 1
            sample_key = _action_key(sample.get("action"), sample.get("target"), sample.get("item"))
            if sample_key == prediction:
                correct += 1
                mode_accum[sample["mode"]]["correct"] += 1

        top1_accuracy = (correct / total_samples) if total_samples else None

        episode_stats: dict[str, dict[str, float]] = defaultdict(lambda: {
            "episodes": 0,
            "success": 0,
            "steps": 0,
            "reward_sum": 0.0,
        })
        total_success = 0
        reward_total = 0.0
        for episode in _piter(episodes, desc="WM v1: episode stats", total=len(episodes) if isinstance(episodes, list) else None):
            transitions = episode.get("transitions", [])
            if not transitions:
                continue
            mode = episode.get("mode", "curriculum")
            info = episode_stats[mode]
            info["episodes"] += 1
            info["steps"] += len(transitions)
            ep_reward = sum(float(t.get("reward", 0.0)) for t in transitions)
            info["reward_sum"] += ep_reward
            reward_total += ep_reward
            success = bool(transitions[-1].get("success"))
            if success:
                info["success"] += 1
                total_success += 1

        avg_reward = (reward_total / len(episodes)) if episodes else None
        prev_raw = self._read_metrics_dict()
        prev_ema = 0.0
        if isinstance(prev_raw, dict):
            try:
                prev_ema = float(prev_raw.get("ema_reward", 0.0))
            except (TypeError, ValueError):
                prev_ema = 0.0
        if avg_reward is not None:
            alpha = 0.3
            ema_reward = alpha * avg_reward + (1.0 - alpha) * prev_ema
        else:
            ema_reward = prev_ema
        if episodes:
            overall_success_rate = total_success / len(episodes)
        elif isinstance(prev_raw, dict):
            try:
                overall_success_rate = float(prev_raw.get("overall_success_rate", 0.0))
            except (TypeError, ValueError):
                overall_success_rate = None
        else:
            overall_success_rate = None

        per_mode_summary: dict[str, dict[str, Any]] = {}
        mode_keys = set(mode_accum.keys()) | set(episode_stats.keys())
        for mode in sorted(mode_keys):
            ma = mode_accum.get(mode, {"samples": 0, "correct": 0})
            es = episode_stats.get(mode, {"episodes": 0, "success": 0, "steps": 0, "reward_sum": 0.0})
            samples = int(ma.get("samples", 0))
            correct_mode = int(ma.get("correct", 0))
            episodes_count = int(es.get("episodes", 0))
            per_mode_summary[mode] = {
                "samples": samples,
                "top1_accuracy": (correct_mode / samples) if samples else None,
                "episodes": episodes_count,
                "success_rate": (es["success"] / episodes_count) if episodes_count else None,
                "avg_steps": (es["steps"] / episodes_count) if episodes_count else None,
                "avg_reward": (es["reward_sum"] / episodes_count) if episodes_count else None,
            }

        timestamp = datetime.now(UTC).isoformat()
        state_summaries = [_format_state_summary(key, counts) for key, counts in state_counts.items()]

        world_model_stats: dict[str, Any] = {
            "trained_at": timestamp,
            "samples": total_samples,
            "states": len(state_counts),
            "top1_accuracy": top1_accuracy,
            "per_mode": per_mode_summary,
            "episodes": len(episodes),
            "avg_reward": avg_reward,
            "state_details": state_summaries,
        }

        self.record_world_model_metrics(world_model_stats, ema_reward=ema_reward, overall_success_rate=overall_success_rate)
        self._persist_world_model(state_counts, timestamp)

        # Also train featureized WM v2 (policy + transitions)
        try:
            v2_stats = self._retrain_world_model_v2(dataset)
            if isinstance(v2_stats, dict):
                world_model_stats["v2"] = {
                    "policy_entries": v2_stats.get("policy_entries"),
                    "transition_entries": v2_stats.get("transition_entries"),
                    "alpha": v2_stats.get("alpha"),
                }
        except Exception:
            pass

        # Optional: train tiny neural model when enabled
        try:
            import os as _os
            _neural_on = (_os.getenv("BRAIN_WM_NEURAL") or "").strip().lower() in {"1","true","yes","on"}
        except Exception:
            _neural_on = False
        if _neural_on:
            try:
                neural_stats = self._train_neural_world_model(dataset)
                if isinstance(neural_stats, dict):
                    world_model_stats["neural"] = {
                        "samples": neural_stats.get("samples"),
                        "top1_accuracy": neural_stats.get("top1_accuracy"),
                        "train_top1_accuracy": neural_stats.get("train_top1_accuracy"),
                        "test_top1_accuracy": neural_stats.get("test_top1_accuracy"),
                    }
            except Exception:
                pass

        validation_payload: dict[str, Any] | None = None
        try:
            validation_payload = self.validate_world_model_quality()
        except Exception as exc:
            validation_payload = {"ok": False, "error": str(exc)}

        proof_path: Path | None = None
        if validation_payload is not None:
            try:
                stats_payload = json.loads(
                    json.dumps(world_model_stats, sort_keys=True, separators=(",", ":"), ensure_ascii=True),
                )
                validation_payload = json.loads(
                    json.dumps(validation_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True),
                )
                proof_path = write_retrain_validation_proof(
                    artifacts_dir=self.artifacts_dir,
                    workspace=self.workspace,
                    stats=stats_payload,
                    validation=validation_payload,
                )
            except Exception:
                proof_path = None

        if validation_payload is not None:
            world_model_stats["validation"] = validation_payload
        if proof_path is not None:
            world_model_stats["validation_proof"] = {"path": str(proof_path)}

        return world_model_stats

    def _persist_world_model(
        self,
        state_counts: dict[tuple[str, str, bool, bool, bool, bool], Counter[ActionKey]],
        timestamp: str,
    ) -> None:
        model_entries: list[dict[str, Any]] = []
        for key, counts in state_counts.items():
            entry = _format_state_summary(key, counts)
            model_entries.append(entry)
        payload = {
            "trained_at": timestamp,
            "workspace": self.workspace,
            "states": model_entries,
        }
        model_path = self.ws_dir / "world_model.json"
        model_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _retrain_world_model_v2(self, dataset: list[dict[str, Any]]) -> dict[str, Any]:
        """Train a featureized policy distribution with Laplace smoothing and a simple transition model.

        Persists to world_model_v2.json. Returns basic stats.
        """
        # Local progress iterator: wraps iterable with tqdm when available; otherwise returns as-is.
        def _piter(iterable, desc: str | None = None, total: int | None = None):
            try:
                from tqdm import tqdm  # type: ignore
                return tqdm(iterable, desc=desc, total=total, leave=False)
            except Exception:
                return iterable
        alpha = 1.0
        try:
            import os as _os
            _alpha = _os.getenv("BRAIN_WM_V2_LAPLACE_ALPHA")
            if _alpha is not None and _alpha.strip() != "":
                alpha = max(0.0, float(_alpha))
        except Exception:
            alpha = 1.0

        # Optional action-quality weighting controls
        try:
            import os as _os
            _quality_on = (_os.getenv("BRAIN_WM_ACTION_QUALITY") or "1").strip().lower() in {"1","true","yes","on"}
        except Exception:
            _quality_on = True

        def _quality_multiplier(sig: str, st: dict[str, Any], ns: dict[str, Any] | None = None) -> float:
            if not _quality_on:
                return 1.0
            try:
                name, target, item = _parse_signature(sig)
                room = str(st.get("room", ""))
                inv = set(st.get("inventory") or [])
                has_key = ("key" in inv)
                door = st.get("door") or {}
                unlocked = bool(door.get("unlocked"))
                open_flag = bool(door.get("open"))
                # Penalize regressive/no-op moves
                if name == "move" and target == room:
                    return 0.2
                if room == "hall" and name == "move" and target == "start":
                    return 0.3
                # Encourage key acquisition path when not holding key
                if not has_key:
                    if name == "search":
                        return 1.2
                    if name == "move" and target in {"storage", "hall"} and room != target:
                        return 1.1
                # Encourage vault progression when holding key
                if has_key and room == "hall":
                    if not unlocked and name == "unlock" and target == "vault":
                        return 1.3
                    if unlocked and not open_flag and name == "open" and target == "vault":
                        return 1.3
                    if unlocked and open_flag and name == "move" and target == "vault":
                        return 1.3
                # Mild bonus if actual next state shows progress
                if ns is not None:
                    try:
                        if _progress(st, ns):
                            return 1.25
                    except Exception:
                        pass
            except Exception:
                return 1.0
            return 1.0

        policy_counts: dict[str, Counter[str]] = defaultdict(Counter)
        policy_totals: dict[str, int] = defaultdict(int)
        trans_counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)  # (feat, act_sig) -> next_feat counts
        trans_totals: dict[tuple[str, str], int] = defaultdict(int)

        for sample in _piter(dataset, desc="WM v2: build counts", total=len(dataset) if isinstance(dataset, list) else None):
            s = sample.get("state") or {}
            ns = sample.get("next_state") or {}
            ak = _action_key(sample.get("action"), sample.get("target"), sample.get("item"))
            if not ak[0]:
                continue
            sig = _action_signature(ak)
            fk = _serialize_feat_key(_feature_key_from_state(s))
            nfk = _serialize_feat_key(_feature_key_from_state(ns))
            # Apply action-quality weighting: skip clearly regressive actions; boost progress
            w = _quality_multiplier(sig, s, ns)
            if w <= 0.0:
                continue
            inc = 1
            if w >= 1.2:
                inc = 2
            policy_counts[fk][sig] += inc
            policy_totals[fk] += inc
            trans_counts[(fk, sig)][nfk] += inc
            trans_totals[(fk, sig)] += inc

        # Persist
        timestamp = datetime.now(UTC).isoformat()
        payload: dict[str, Any] = {
            "trained_at": timestamp,
            "alpha": alpha,
            "rooms": list(_ROOMS),
            "policy": {},
            "transitions": {},
            "features_schema": [
                "room",
                "has_key",
                "door_unlocked",
                "door_open",
                "key_visible",
                "steps_mod2",
                "exits_count",
            ],
        }

        for fk, counts in policy_counts.items():
            payload["policy"][f"f:{fk}"] = {
                "counts": {sig: int(c) for sig, c in counts.items()},
                "total": int(policy_totals.get(fk, 0)),
            }
        for (fk, sig), ncounts in trans_counts.items():
            payload["transitions"][f"f:{fk}|{sig}"] = {
                "next": {nfk: int(c) for nfk, c in ncounts.items()},
                "total": int(trans_totals.get((fk, sig), 0)),
            }

        # Optional: inject minimal movement priors for underrepresented but valid moves.
        # Controlled by env var BRAIN_WM_V2_INJECT_PRIORS in {1,true,yes,on}.
        try:
            import os as _os
            _inject_raw = (_os.getenv("BRAIN_WM_V2_INJECT_PRIORS") or "").strip().lower()
            _inject = _inject_raw in {"1", "true", "yes", "on"}
        except Exception:
            _inject = False

        if _inject:
            def _parse_fk(fk_str: str) -> dict[str, str]:
                parts = {}
                for p in fk_str.split(";"):
                    if "=" in p:
                        k, v = p.split("=", 1)
                        parts[k] = v
                return parts
            def _ser_fk(d: dict[str, str]) -> str:
                # Preserve canonical order
                return ";".join([
                    f"room={d.get('room','start')}",
                    f"hk={d.get('hk','0')}",
                    f"u={d.get('u','0')}",
                    f"o={d.get('o','0')}",
                    f"kv={d.get('kv','0')}",
                    f"s2={d.get('s2','0')}",
                    f"xc={d.get('xc','1')}",
                ])
            # Helper to bump policy counts
            def _policy_inc(entry: dict[str, Any], sig: str, inc: int = 1) -> None:
                counts = entry.setdefault("counts", {})
                counts[sig] = int(counts.get(sig, 0) or 0) + inc
                entry["total"] = int(entry.get("total", 0) or 0) + inc
            # Helper to bump transitions
            def _trans_inc(fk: str, sig: str, nfk: str, inc: int = 1) -> None:
                key = f"f:{fk}|{sig}"
                ten = payload["transitions"].setdefault(key, {"next": {}, "total": 0})
                nxt = ten.setdefault("next", {})
                nxt[nfk] = int(nxt.get(nfk, 0) or 0) + inc
                ten["total"] = int(ten.get("total", 0) or 0) + inc
            # Helper to get/create a policy entry for a feature key
            def _get_or_create_policy_entry(fk: str) -> dict[str, Any]:
                key = f"f:{fk}"
                entry = payload["policy"].get(key)
                if entry is None:
                    entry = {"counts": {}, "total": 0}
                    payload["policy"][key] = entry
                return entry

            for key, entry in list(payload["policy"].items()):
                if not isinstance(entry, dict) or not key.startswith("f:"):
                    continue
                fk = key[2:]
                d = _parse_fk(fk)
                room = d.get("room", "")
                hk = d.get("hk", "0")
                u = d.get("u", "0")
                o = d.get("o", "0")
                kv = d.get("kv", "0")
                s2 = d.get("s2", "0")
                # Vault progression priors when agent holds the key in hall
                if room == "hall" and hk == "1":
                    if u == "0":
                        _policy_inc(entry, "unlock|vault|")
                        d_next = dict(d)
                        d_next["u"] = "1"
                        d_next["s2"] = "1" if s2 == "0" else "0"
                        nfk = _ser_fk(d_next)
                        _trans_inc(fk, "unlock|vault|", nfk)
                    elif u == "1" and o == "0":
                        _policy_inc(entry, "open|vault|")
                        d_next = dict(d)
                        d_next["o"] = "1"
                        d_next["s2"] = "1" if s2 == "0" else "0"
                        nfk = _ser_fk(d_next)
                        _trans_inc(fk, "open|vault|", nfk)
                    elif u == "1" and o == "1":
                        _policy_inc(entry, "move|vault|")
                        d_next = dict(d)
                        d_next["room"] = "vault"
                        d_next["s2"] = "1" if s2 == "0" else "0"
                        # Arriving in vault typically reduces exits_count; keep kv=0
                        d_next["xc"] = d.get("xc", "1")
                        nfk = _ser_fk(d_next)
                        _trans_inc(fk, "move|vault|", nfk)
                # Inject start -> hall when not holding key (covers both kv=0 and kv=1 states)
                if room == "start" and hk == "0":
                    _policy_inc(entry, "move|hall|")
                    # Add a minimal transition to hall (toggle s2, adjust exits_count, clear key visibility)
                    d_next = dict(d)
                    d_next["room"] = "hall"
                    d_next["kv"] = "0"
                    d_next["s2"] = "1" if s2 == "0" else "0"
                    d_next["xc"] = "3"
                    nfk = _ser_fk(d_next)
                    _trans_inc(fk, "move|hall|", nfk)
                # Inject hall -> storage bias even when holding key (does not break validity, captures exploration)
                if room == "hall" and hk == "1":
                    _policy_inc(entry, "move|storage|")
                    d_next = dict(d)
                    d_next["room"] = "storage"
                    d_next["s2"] = "1" if s2 == "0" else "0"
                    d_next["xc"] = "1"
                    nfk = _ser_fk(d_next)
                    _trans_inc(fk, "move|storage|", nfk)

                # RoomsPlus priors: when hall has many exits (xc >= 5), encourage exploring office and lab
                # This is gated by the observed feature fk (xc) so base envs (xc<=3) are unaffected.
                try:
                    xc_val = int(d.get("xc", "0"))
                except Exception:
                    xc_val = 0
                if room == "hall" and xc_val >= 5:
                    for dest in ("office", "lab"):
                        _policy_inc(entry, f"move|{dest}|")
                        d_next = dict(d)
                        d_next["room"] = dest
                        d_next["kv"] = "0"  # default to not-visible on arrival
                        d_next["s2"] = "1" if s2 == "0" else "0"
                        d_next["xc"] = "1"  # leaf rooms typically have one exit (back to hall)
                        nfk = _ser_fk(d_next)
                        _trans_inc(fk, f"move|{dest}|", nfk)

            # Additionally, if no hall entries were present (common in new RoomsPlus workspaces), synthesize minimal hall entries
            # so the prior applies. We create entries for hk in {0,1} and s2 in {0,1} with xc=5, u=1, o=1, kv=0.
            hall_seeded = any(
                k.startswith("f:room=hall;") and ";xc=5" in k for k in payload["policy"].keys()
            )
            # Allow forcing hall seeding even if some hall entries already exist
            try:
                import os as _os
                _force_hall_seed = (_os.getenv("BRAIN_WM_V2_FORCE_HALL_SEED") or "").strip().lower() in {"1","true","yes","on"}
            except Exception:
                _force_hall_seed = False
            if (not hall_seeded) or _force_hall_seed:
                for hk in ("0", "1"):
                    for s2 in ("0", "1"):
                        d = {"room": "hall", "hk": hk, "u": "1", "o": "1", "kv": "0", "s2": s2, "xc": "5"}
                        fk = _ser_fk(d)
                        entry = _get_or_create_policy_entry(fk)
                        for dest in ("office", "lab"):
                            _policy_inc(entry, f"move|{dest}|")
                            d_next = dict(d)
                            d_next["room"] = dest
                            d_next["kv"] = "0"
                            d_next["s2"] = "1" if s2 == "0" else "0"
                            d_next["xc"] = "1"
                            nfk = _ser_fk(d_next)
                            _trans_inc(fk, f"move|{dest}|", nfk)
                        # Also seed vault move prior when already open+unlocked and holding key
                        if hk == "1":
                            _policy_inc(entry, "move|vault|")
                            d_next2 = dict(d)
                            d_next2["room"] = "vault"
                            d_next2["s2"] = "1" if s2 == "0" else "0"
                            nfk2 = _ser_fk(d_next2)
                            _trans_inc(fk, "move|vault|", nfk2)

            # Seed base environment hall entries when priors enabled but no base hall+hk=1 entry exists
            # Check specifically for xc=3 (base env) to avoid confusion with RoomsPlus xc=5 entries
            base_hall_hk1_exists = any(
                k.startswith("f:room=hall;hk=1;") and ";xc=3" in k for k in payload["policy"].keys()
            )
            if not base_hall_hk1_exists:
                # Create minimal hall;hk=1 entries for base env (xc=3)
                for s2 in ("0", "1"):
                    d = {"room": "hall", "hk": "1", "u": "0", "o": "0", "kv": "0", "s2": s2, "xc": "3"}
                    fk = _ser_fk(d)
                    entry = _get_or_create_policy_entry(fk)
                    # Inject hall->storage prior
                    _policy_inc(entry, "move|storage|")
                    d_next = dict(d)
                    d_next["room"] = "storage"
                    d_next["s2"] = "1" if s2 == "0" else "0"
                    d_next["xc"] = "1"
                    nfk = _ser_fk(d_next)
                    _trans_inc(fk, "move|storage|", nfk)

            # Seed reverse moves for leaf rooms in RoomsPlus: office/lab -> hall, and basic search priors in leaf rooms when no key
            def _seed_leaf_room(room_name: str) -> None:
                # Create minimal leaf entries for hk in {0,1}, s2 in {0,1}, with xc=1, u/o inherit as 1
                for hk in ("0", "1"):
                    for s2 in ("0", "1"):
                        d = {"room": room_name, "hk": hk, "u": "1", "o": "1", "kv": "0", "s2": s2, "xc": "1"}
                        fk = _ser_fk(d)
                        entry = _get_or_create_policy_entry(fk)
                        # reverse move to hall
                        _policy_inc(entry, "move|hall|")
                        d_next = dict(d)
                        d_next["room"] = "hall"
                        d_next["kv"] = "0"
                        d_next["s2"] = "1" if s2 == "0" else "0"
                        d_next["xc"] = "5"  # back to hall fan-out in RoomsPlus
                        nfk = _ser_fk(d_next)
                        _trans_inc(fk, "move|hall|", nfk)
                        # minimal search prior when not holding key and key not visible
                        if hk == "0":
                            _policy_inc(entry, "search||")
                            # keep kv unchanged to avoid fabricating reveal semantics
                            n2 = dict(d)
                            n2["s2"] = "1" if s2 == "0" else "0"
                            nfk2 = _ser_fk(n2)
                            _trans_inc(fk, "search||", nfk2)

            # Always seed office/lab leaf priors to ensure planner can traverse and return
            _seed_leaf_room("office")
            _seed_leaf_room("lab")
            # Also seed storage similarly to help lab_key sequences that route through storage
            _seed_leaf_room("storage")

        model_path = self.ws_dir / "world_model_v2.json"
        model_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        # Unconditional minimal coverage repair: ensure basic policies exist even if priors are disabled
        try:
            def _p_get(entry: dict[str, Any]) -> dict[str, int]:
                c = entry.get("counts")
                return c if isinstance(c, dict) else {}
            def _p_set(entry: dict[str, Any], counts: dict[str, int]) -> None:
                entry["counts"] = {k: int(v) for k, v in counts.items()}
                entry["total"] = int(sum(int(v) for v in counts.values()))
            def _ser(d: dict[str, str]) -> str:
                return ";".join([
                    f"room={d.get('room','start')}",
                    f"hk={d.get('hk','0')}",
                    f"u={d.get('u','0')}",
                    f"o={d.get('o','0')}",
                    f"kv={d.get('kv','0')}",
                    f"s2={d.get('s2','0')}",
                    f"xc={d.get('xc','1')}",
                ])
            def _ensure_policy_sig(fk: str, sig: str, inc: int = 1) -> None:
                key = f"f:{fk}"
                entry = payload["policy"].get(key)
                if not isinstance(entry, dict):
                    entry = {"counts": {}, "total": 0}
                    payload["policy"][key] = entry
                counts = _p_get(entry)
                counts[sig] = int(counts.get(sig, 0) or 0) + inc
                _p_set(entry, counts)
            def _ensure_transition(fk: str, sig: str, nfk: str, inc: int = 1) -> None:
                key = f"f:{fk}|{sig}"
                ten = payload["transitions"].get(key)
                if not isinstance(ten, dict):
                    ten = {"next": {}, "total": 0}
                    payload["transitions"][key] = ten
                nxt = ten.get("next") if isinstance(ten.get("next"), dict) else {}
                nxt[nfk] = int(nxt.get(nfk, 0) or 0) + inc
                ten["next"] = nxt
                ten["total"] = int(ten.get("total", 0) or 0) + inc

            # start_no_key -> move|hall|
            for s2 in ("0", "1"):
                d = {"room": "start", "hk": "0", "u": "0", "o": "0", "kv": "0", "s2": s2, "xc": "1"}
                fk = _ser(d)
                n = dict(d); n["room"] = "hall"; n["kv"] = "0"; n["s2"] = ("1" if s2 == "0" else "0"); n["xc"] = "3"
                _ensure_policy_sig(fk, "move|hall|", 1)
                _ensure_transition(fk, "move|hall|", _ser(n), 1)

            # hall_no_key -> move|storage|
            for s2 in ("0", "1"):
                d = {"room": "hall", "hk": "0", "u": "0", "o": "0", "kv": "0", "s2": s2, "xc": "3"}
                fk = _ser(d)
                n = dict(d); n["room"] = "storage"; n["kv"] = "0"; n["s2"] = ("1" if s2 == "0" else "0"); n["xc"] = "1"
                _ensure_policy_sig(fk, "move|storage|", 1)
                _ensure_transition(fk, "move|storage|", _ser(n), 1)

            # storage_no_key and key hidden/visible
            for s2 in ("0", "1"):
                # hidden key (kv=0) -> search|| (reveals key)
                d0 = {"room": "storage", "hk": "0", "u": "0", "o": "0", "kv": "0", "s2": s2, "xc": "1"}
                fk0 = _ser(d0)
                n0 = dict(d0); n0["s2"] = ("1" if s2 == "0" else "0")  # reveal modeled via step toggle only
                _ensure_policy_sig(fk0, "search||", 1)
                _ensure_transition(fk0, "search||", _ser(n0), 1)
                # key visible (kv=1) -> pickup||key
                d1 = {"room": "storage", "hk": "0", "u": "0", "o": "0", "kv": "1", "s2": s2, "xc": "1"}
                fk1 = _ser(d1)
                n1 = dict(d1); n1["hk"] = "1"; n1["kv"] = "0"; n1["s2"] = ("1" if s2 == "0" else "0")
                _ensure_policy_sig(fk1, "pickup||key", 1)
                _ensure_transition(fk1, "pickup||key", _ser(n1), 1)
        except Exception:
            pass

        # Re-write after coverage repair
        model_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return {
            "alpha": alpha,
            "policy_entries": len(policy_counts),
            "transition_entries": len(trans_counts),
        }

    # --- Optional tiny neural classifier (tabular features -> action signature) ---

    def _neural_action_vocab(self) -> list[str]:
        """Return a fixed small action vocabulary for navigation tasks.

        This keeps the model tiny and deterministic. Extend as new rooms/actions are introduced.
        """
        vocab = [
            "move|hall|",
            "move|start|",
            "move|storage|",
            "move|vault|",
            "search||",
            "pickup||key",
            "unlock|vault|",
            "open|vault|",
            # RoomsPlus extras
            "move|office|",
            "move|lab|",
        ]
        return vocab

    def _neural_rooms(self) -> list[str]:
        # Include RoomsPlus rooms; harmless in base env (one-hots remain zero)
        return ["start", "hall", "storage", "vault", "office", "lab"]

    def _encode_state_for_neural(self, st: dict[str, Any]) -> list[float]:
        # One-hot rooms + expanded boolean/numeric flags
        rooms = self._neural_rooms()
        room = str(st.get("room", ""))
        inv = set(st.get("inventory") or [])
        has_key = 1.0 if ("key" in inv) else 0.0
        door = st.get("door") or {}
        unlocked = 1.0 if door.get("unlocked") else 0.0
        open_flag = 1.0 if door.get("open") else 0.0
        vis = set(st.get("visible_items") or [])
        key_visible = 1.0 if ("key" in vis) else 0.0
        steps_mod2 = float(int(st.get("steps", 0)) % 2)
        exits = st.get("exits") or []
        try:
            exits_count = float(len(exits))
        except Exception:
            exits_count = 0.0
        onehot = [1.0 if room == r else 0.0 for r in rooms]
        # Extras: specific exits, door negations, mode flags, visible items count
        in_exits = set(exits) if isinstance(exits, (list, tuple, set)) else set()
        vault_in_exits = 1.0 if "vault" in in_exits else 0.0
        storage_in_exits = 1.0 if "storage" in in_exits else 0.0
        start_in_exits = 1.0 if "start" in in_exits else 0.0
        hall_in_exits = 1.0 if "hall" in in_exits else 0.0
        door_locked = 1.0 - unlocked
        door_closed = 1.0 - open_flag
        mode = str(st.get("mode", ""))
        is_curriculum = 1.0 if mode == "curriculum" else 0.0
        is_generalization_plus = 1.0 if mode == "generalization_plus" else 0.0
        try:
            vis_count = float(len(vis))
        except Exception:
            vis_count = 0.0
        # Composite features (low-risk, informative):
        # - vault_unlocked_and_open: captures final-door-ready condition
        vault_unlocked_and_open = 1.0 if (unlocked == 1.0 and open_flag == 1.0) else 0.0
        # - room_change_possible: whether there's any exit different from current room
        try:
            _has_alt_exit = any((str(e) != room) for e in in_exits) if in_exits else False
        except Exception:
            _has_alt_exit = False
        room_change_possible = 1.0 if _has_alt_exit else 0.0
        # - key_visible_and_door_locked: useful for prioritizing key pickup when vault locked
        key_visible_and_door_locked = 1.0 if (key_visible == 1.0 and door_locked == 1.0) else 0.0
        feats = onehot + [
            has_key, unlocked, open_flag, key_visible,
            steps_mod2, exits_count,
            vault_in_exits, storage_in_exits, start_in_exits, hall_in_exits,
            door_locked, door_closed,
            is_curriculum, is_generalization_plus,
            vis_count,
            # appended composites
            vault_unlocked_and_open,
            room_change_possible,
            key_visible_and_door_locked,
        ]
        return feats

    def _softmax(self, z: list[float]) -> list[float]:
        import math
        m = max(z) if z else 0.0
        exps = [math.exp(v - m) for v in z]
        s = sum(exps) or 1.0
        return [v / s for v in exps]

    def _neural_predict_action_sig(self, state: dict[str, Any], allowed: set[str] | None = None) -> str | None:
        model_path = self.ws_dir / "neural_wm.json"
        if not model_path.exists():
            return None
        try:
            payload = json.loads(model_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None
        W = payload.get("W"); b = payload.get("b"); vocab = payload.get("vocab"); rooms = payload.get("rooms")
        if not (isinstance(W, list) and isinstance(b, list) and isinstance(vocab, list) and isinstance(rooms, list)):
            return None
        # Ensure encoder matches saved rooms length
        x = self._encode_state_for_neural(state)
        D = len(x); K = len(b)
        # Matmul x @ W (W stored as D x K)
        if not W or len(W) != D:
            return None
        scores = [0.0 for _ in range(K)]
        for i in range(D):
            wi = W[i]
            if not isinstance(wi, list):
                return None
            xi = float(x[i])
            for k in range(K):
                try:
                    scores[k] += xi * float(wi[k])
                except Exception:
                    return None
        for k in range(K):
            try:
                scores[k] += float(b[k])
            except Exception:
                pass
        probs = self._softmax(scores)
        # Rank actions by prob; apply allowed mask if provided
        ranked = sorted(range(K), key=lambda k: probs[k], reverse=True)
        for idx in ranked:
            sig = str(vocab[idx])
            if allowed is not None and sig not in allowed:
                continue
            return sig
        return None

    def _train_neural_world_model(self, dataset: list[dict[str, Any]]) -> dict[str, Any]:
        """Tiny multiclass logistic regression (softmax) via SGD on tabular features.

        Saves to neural_wm.json with fields {rooms, vocab, W, b, samples, top1_accuracy,
        train_top1_accuracy, test_top1_accuracy}. Returns basic stats.
        Controlled by env var BRAIN_WM_NEURAL.
        """
        # Gather samples
        vocab = self._neural_action_vocab()
        index = {a: i for i, a in enumerate(vocab)}
        X: list[list[float]] = []
        Y: list[int] = []
        for s in dataset:
            ak = _action_key(s.get("action"), s.get("target"), s.get("item"))
            sig = _action_signature(ak)
            if sig not in index:
                continue
            x = self._encode_state_for_neural(s.get("state") or {})
            X.append(x)
            Y.append(index[sig])
        N = len(X)
        if N == 0:
            return {"samples": 0, "top1_accuracy": None}
        D = len(X[0]); K = len(vocab)
        # Deterministic split (80/20)
        import random
        rng = random.Random(1337)
        idxs_all = list(range(N))
        rng.shuffle(idxs_all)
        split = max(1, int(0.8 * N))
        idx_tr = idxs_all[:split]
        idx_te = idxs_all[split:] if split < N else []
        # Initialize weights
        W = [[0.0 for _ in range(K)] for _ in range(D)]
        b = [0.0 for _ in range(K)]
        lr = 0.1
        # Small L2 regularization (weight decay) for generalization; env override allowed
        l2 = 1e-4
        try:
            import os as _os
            _l2 = _os.getenv("BRAIN_WM_NEURAL_L2")
            if _l2 is not None and _l2.strip() != "":
                l2 = max(0.0, float(_l2))
        except Exception:
            l2 = 1e-4
        # Allow more optimization steps on larger datasets; still bounded for determinism
        # Cap can be overridden by env var BRAIN_WM_NEURAL_EPOCHS_MAX
        max_cap = 300
        try:
            import os as _os
            _cap = _os.getenv("BRAIN_WM_NEURAL_EPOCHS_MAX")
            if _cap is not None and _cap.strip() != "":
                max_cap = max(1, int(float(_cap)))
        except Exception:
            max_cap = 300
        epochs = max(10, min(max_cap, 10 + N // 10))
        # Train only on training split
        for _ in range(epochs):
            rng.shuffle(idx_tr)
            for j in idx_tr:
                x = X[j]; y = Y[j]
                # forward
                scores = [0.0 for _ in range(K)]
                for i in range(D):
                    wi = W[i]; xi = x[i]
                    for k in range(K):
                        scores[k] += wi[k] * xi
                for k in range(K):
                    scores[k] += b[k]
                # softmax
                probs = self._softmax(scores)
                # gradients and update (cross-entropy)
                for k in range(K):
                    grad = probs[k] - (1.0 if k == y else 0.0)
                    b[k] -= lr * grad
                for i in range(D):
                    xi = X[j][i]
                    for k in range(K):
                        grad = probs[k] - (1.0 if k == Y[j] else 0.0)
                        # L2 penalty applied to weights (not biases)
                        W[i][k] -= lr * (grad * xi + (l2 * W[i][k]))
        # Evaluate top-1 on train and test
        def _acc(indices: list[int]) -> float | None:
            if not indices:
                return None
            correct = 0
            for j in indices:
                x = X[j]
                scores = [0.0 for _ in range(K)]
                for i in range(D):
                    wi = W[i]; xi = x[i]
                    for k in range(K):
                        scores[k] += wi[k] * xi
                for k in range(K):
                    scores[k] += b[k]
                pred = max(range(K), key=lambda k: scores[k])
                if pred == Y[j]:
                    correct += 1
            return (correct / len(indices)) if indices else None

        acc_train = _acc(idx_tr)
        acc_test = _acc(idx_te)
        # Back-compat top1_accuracy = full-dataset accuracy
        correct_all = 0
        for j in range(N):
            x = X[j]
            scores = [0.0 for _ in range(K)]
            for i in range(D):
                wi = W[i]; xi = x[i]
                for k in range(K):
                    scores[k] += wi[k] * xi
            for k in range(K):
                scores[k] += b[k]
            pred = max(range(K), key=lambda k: scores[k])
            if pred == Y[j]:
                correct_all += 1
        acc = (correct_all / N) if N else None
        # Persist
        payload = {
            "rooms": self._neural_rooms(),
            "vocab": vocab,
            "W": W,
            "b": b,
            "trained_at": datetime.now(UTC).isoformat(),
            "samples": N,
            "top1_accuracy": acc,
            "train_top1_accuracy": acc_train,
            "test_top1_accuracy": acc_test,
        }
        (self.ws_dir / "neural_wm.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"samples": N, "top1_accuracy": acc, "train_top1_accuracy": acc_train, "test_top1_accuracy": acc_test}

    def load_world_model_v2(self) -> dict[str, Any]:
        model_path = self.ws_dir / "world_model_v2.json"
        if not model_path.exists():
            return {}
        try:
            return json.loads(model_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def validate_world_model_quality(self) -> dict[str, Any]:
        """Lightweight sanity checks to catch pathological policies.

        Returns a dict with {ok: bool, failures: [..]} and records metrics under world_model.validation.
        """
        failures: list[str] = []
        wm2 = self.load_world_model_v2()
        policy = wm2.get("policy") if isinstance(wm2, dict) else None
        if not isinstance(policy, dict) or not policy:
            return {"ok": False, "failures": ["no_v2_policy"]}
        # Canonical states to test
        def _mk(room: str, *, hk: int = 0, u: int = 0, o: int = 0, kv: int = 0, xc: int = 3, s2: int = 0) -> dict[str, Any]:
            door = {"unlocked": bool(u), "open": bool(o)}
            inv = (["key"] if hk else [])
            vis = (["key"] if kv else [])
            exits = ["start", "storage", "vault"] if room == "hall" else (["hall"] if room in {"start", "storage"} else [])
            return {"room": room, "inventory": inv, "visible_items": vis, "exits": exits, "door": door, "mode": "generalization_plus", "steps": 0}
        cases = [
            ("hall_no_key", _mk("hall", hk=0, u=0, o=0, kv=0, xc=3), {"forbid": {"move|start|"}}),
            ("hall_have_key_lock", _mk("hall", hk=1, u=0, o=0, kv=0, xc=3), {"require_any": {"unlock|vault|", "move|storage|"}}),
            ("storage_key_visible", _mk("storage", hk=0, kv=1, xc=1), {"require_any": {"pickup||key", "search||"}}),
        ]
        bad = []
        for name, st, rule in cases:
            ranked = []
            try:
                # Use internal ranker through planner context
                ranked = []
                try:
                    ranked = []
                    # Inline ranker (must mirror _v2_rank_action_sigs logic minimally)
                    fk = _serialize_feat_key(_feature_key_from_state(st))
                    entry = policy.get(f"f:{fk}")
                    if isinstance(entry, dict):
                        counts = entry.get("counts") if isinstance(entry.get("counts"), dict) else {}
                        total = int(entry.get("total", 0))
                        k = len(counts)
                        denom = float(total + (wm2.get("alpha", 1.0) * k)) if k else 1.0
                        for sig, c in counts.items():
                            try:
                                c_int = int(c)
                            except Exception:
                                c_int = 0
                            p = (c_int + wm2.get("alpha", 1.0)) / denom
                            ranked.append((sig, p))
                        ranked.sort(key=lambda x: x[1], reverse=True)
                except Exception:
                    ranked = []
                top = ranked[0][0] if ranked else None
            except Exception:
                top = None
            if not top:
                bad.append(f"{name}: no_top")
                continue
            forbid = set(rule.get("forbid", []))
            require_any = set(rule.get("require_any", []))
            if top in forbid:
                bad.append(f"{name}: regressive_top={top}")
            if require_any and top not in require_any:
                bad.append(f"{name}: unexpected_top={top}")
        ok = not bad
        # Record under metrics
        raw = self._read_metrics_dict()
        if not isinstance(raw, dict):
            raw = {}
        wmsec = raw.setdefault("world_model", {})
        wmsec["validation"] = {"ok": ok, "failures": bad, "ts": datetime.now(UTC).isoformat()}
        raw["world_model"] = wmsec
        raw["updated_at"] = wmsec["validation"]["ts"]
        self.metrics_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"ok": ok, "failures": bad}

    def repair_world_model_v2_policy(self) -> dict[str, Any]:
        """Repair missing basic policies in an already persisted v2 model using the same minimal coverage rules.

        Returns {ok: bool, repaired: int}
        """
        model_path = self.ws_dir / "world_model_v2.json"
        if not model_path.exists():
            return {"ok": False, "repaired": 0, "error": "no_model"}
        try:
            payload = json.loads(model_path.read_text(encoding="utf-8"))
        except Exception as e:
            return {"ok": False, "repaired": 0, "error": str(e)}
        if not isinstance(payload, dict):
            return {"ok": False, "repaired": 0, "error": "invalid_payload"}
        if not isinstance(payload.get("policy"), dict):
            payload["policy"] = {}
        if not isinstance(payload.get("transitions"), dict):
            payload["transitions"] = {}

        repaired = 0
        try:
            def _p_get(entry: dict[str, Any]) -> dict[str, int]:
                c = entry.get("counts")
                return c if isinstance(c, dict) else {}
            def _p_set(entry: dict[str, Any], counts: dict[str, int]) -> None:
                entry["counts"] = {k: int(v) for k, v in counts.items()}
                entry["total"] = int(sum(int(v) for v in counts.values()))
            def _ser(d: dict[str, str]) -> str:
                return ";".join([
                    f"room={d.get('room','start')}",
                    f"hk={d.get('hk','0')}",
                    f"u={d.get('u','0')}",
                    f"o={d.get('o','0')}",
                    f"kv={d.get('kv','0')}",
                    f"s2={d.get('s2','0')}",
                    f"xc={d.get('xc','1')}",
                ])
            def _ensure_policy_sig(fk: str, sig: str, inc: int = 1) -> None:
                nonlocal repaired
                key = f"f:{fk}"
                entry = payload["policy"].get(key)
                if not isinstance(entry, dict):
                    entry = {"counts": {}, "total": 0}
                    payload["policy"][key] = entry
                counts = _p_get(entry)
                before = int(counts.get(sig, 0) or 0)
                counts[sig] = before + inc
                _p_set(entry, counts)
                if before == 0:
                    repaired += 1
            def _ensure_transition(fk: str, sig: str, nfk: str, inc: int = 1) -> None:
                key = f"f:{fk}|{sig}"
                ten = payload["transitions"].get(key)
                if not isinstance(ten, dict):
                    ten = {"next": {}, "total": 0}
                    payload["transitions"][key] = ten
                nxt = ten.get("next") if isinstance(ten.get("next"), dict) else {}
                nxt[nfk] = int(nxt.get(nfk, 0) or 0) + inc
                ten["next"] = nxt
                ten["total"] = int(ten.get("total", 0) or 0) + inc

            # Basic repairs identical to training-time coverage
            for s2 in ("0", "1"):
                d = {"room": "start", "hk": "0", "u": "0", "o": "0", "kv": "0", "s2": s2, "xc": "1"}
                fk = _ser(d)
                n = dict(d); n["room"] = "hall"; n["kv"] = "0"; n["s2"] = ("1" if s2 == "0" else "0"); n["xc"] = "3"
                _ensure_policy_sig(fk, "move|hall|", 1)
                _ensure_transition(fk, "move|hall|", _ser(n), 1)
            for s2 in ("0", "1"):
                d = {"room": "hall", "hk": "0", "u": "0", "o": "0", "kv": "0", "s2": s2, "xc": "3"}
                fk = _ser(d)
                n = dict(d); n["room"] = "storage"; n["kv"] = "0"; n["s2"] = ("1" if s2 == "0" else "0"); n["xc"] = "1"
                _ensure_policy_sig(fk, "move|storage|", 1)
                _ensure_transition(fk, "move|storage|", _ser(n), 1)
            for s2 in ("0", "1"):
                d0 = {"room": "storage", "hk": "0", "u": "0", "o": "0", "kv": "0", "s2": s2, "xc": "1"}
                fk0 = _ser(d0)
                n0 = dict(d0); n0["s2"] = ("1" if s2 == "0" else "0")
                _ensure_policy_sig(fk0, "search||", 1)
                _ensure_transition(fk0, "search||", _ser(n0), 1)
                d1 = {"room": "storage", "hk": "0", "u": "0", "o": "0", "kv": "1", "s2": s2, "xc": "1"}
                fk1 = _ser(d1)
                n1 = dict(d1); n1["hk"] = "1"; n1["kv"] = "0"; n1["s2"] = ("1" if s2 == "0" else "0")
                _ensure_policy_sig(fk1, "pickup||key", 1)
                _ensure_transition(fk1, "pickup||key", _ser(n1), 1)
        except Exception as e:
            return {"ok": False, "repaired": repaired, "error": str(e)}
        # Persist
        try:
            model_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            return {"ok": False, "repaired": repaired, "error": str(e)}
        return {"ok": True, "repaired": repaired}

    def _load_workspace_events(self) -> list[dict[str, Any]]:
        if not self.episodes_path.exists():
            return []
        events: list[dict[str, Any]] = []
        try:
            with self.episodes_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if data.get("workspace") == self.workspace:
                        events.append(data)
        except Exception:
            return []
        return events

    def _build_dataset(
        self, events: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        dataset: list[dict[str, Any]] = []
        episodes: list[dict[str, Any]] = []
        current: list[dict[str, Any]] = []
        last_state: dict[str, Any] | None = None
        current_mode = "curriculum"
        for ev in events:
            mode = ev.get("mode", current_mode)
            event = ev.get("event")
            if event == "reset":
                if current:
                    episodes.append({"mode": current_mode, "transitions": current})
                    current = []
                last_state = ev.get("state")
                current_mode = mode
                continue
            if event == "observe":
                last_state = ev.get("state")
                current_mode = mode
                continue
            if event == "act":
                prev_state = last_state
                action = _normalize_action_name(ev.get("action"))
                next_state = ev.get("state")
                reward = float(ev.get("reward", 0.0))
                success = bool(ev.get("success"))
                done = bool(ev.get("done"))
                if prev_state is not None and action:
                    sample = {
                        "mode": mode,
                        "state": prev_state,
                        "next_state": next_state,
                        "action": action,
                        "target": _normalize_optional(ev.get("target")),
                        "item": _normalize_optional(ev.get("item")),
                        "reward": reward,
                        "success": success,
                    }
                    dataset.append(sample)
                    current.append(sample)
                last_state = next_state
                current_mode = mode
                if done:
                    if current:
                        episodes.append({"mode": mode, "transitions": current})
                        current = []
                continue
        if current:
            episodes.append({"mode": current_mode, "transitions": current})
        return dataset, episodes

    @staticmethod
    def _state_key(mode: str, state: dict[str, Any]) -> tuple[str, str, bool, bool, bool, bool]:
        door = state.get("door", {}) if isinstance(state, dict) else {}
        inventory = state.get("inventory", []) if isinstance(state, dict) else []
        visible_items = state.get("visible_items", []) if isinstance(state, dict) else []
        room = str(state.get("room")) if isinstance(state, dict) else ""
        unlocked = bool(door.get("unlocked"))
        open_flag = bool(door.get("open"))
        has_key = "key" in list(inventory) if isinstance(inventory, (list, tuple, set)) else False
        try:
            visible_iterable = list(visible_items) if isinstance(visible_items, (list, tuple, set)) else []
        except TypeError:
            visible_iterable = []
        key_visible = "key" in visible_iterable
        return (mode, room, unlocked, open_flag, has_key, key_visible)

    def _append_event(self, payload: dict[str, Any]) -> None:
        data = {
            "ts": time.time(),
            "workspace": self.workspace,
            **payload,
        }
        with self.episodes_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(data, ensure_ascii=False) + "\n")
        self._maybe_trim_episodes()

    def _maybe_trim_episodes(self) -> None:
        if not self._episodes_trim_lines:
            return
        self._episodes_append_counter += 1
        if (self._episodes_append_counter % self._episodes_trim_interval) != 0:
            return
        path = self.episodes_path
        if not path.exists() or not path.is_file():
            return
        try:
            buffer: deque[str] = deque(maxlen=self._episodes_trim_lines)
            total = 0
            with path.open("r", encoding="utf-8", errors="ignore") as handle:
                for line in handle:
                    buffer.append(line)
                    total += 1
        except OSError:
            return
        if total <= len(buffer):
            return
        tmp = path.with_suffix(path.suffix + ".trim")
        try:
            with tmp.open("w", encoding="utf-8") as handle:
                handle.writelines(buffer)
            tmp.replace(path)
            self._episodes_append_counter = 0
        except OSError:
            try:
                tmp.unlink(missing_ok=True)  # pragma: no cover
            except Exception:
                pass

    def _update_metrics(self, *, success: bool, reward: float) -> None:
        raw = self._read_metrics_dict()
        modes = raw.setdefault("modes", {})
        stats = modes.setdefault(self._mode, {"count": 0, "success": 0, "success_rate": 0.0})
        stats["count"] = int(stats.get("count", 0)) + 1
        if success:
            stats["success"] = int(stats.get("success", 0)) + 1
        stats["success_rate"] = (stats.get("success", 0) / stats["count"]) if stats["count"] else 0.0
        alpha = 0.3
        prev = float(raw.get("ema_reward", 0.0))
        raw["ema_reward"] = alpha * reward + (1.0 - alpha) * prev
        total_success = sum(int(m.get("success", 0)) for m in modes.values())
        total_count = sum(int(m.get("count", 0)) for m in modes.values())
        raw["overall_success_rate"] = (total_success / total_count) if total_count else 0.0
        raw["episodes"] = int(raw.get("episodes", 0)) + 1
        raw["updated_at"] = datetime.now(UTC).isoformat()
        self.metrics_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")

    def _read_metrics(self) -> MetricsSnapshot:
        raw = self._read_metrics_dict()
        return MetricsSnapshot(
            modes={k: {**v} for k, v in raw.get("modes", {}).items()},
            ema_reward=float(raw.get("ema_reward", 0.0)),
            overall_success_rate=float(raw.get("overall_success_rate", 0.0)),
            updated_at=str(raw.get("updated_at", datetime.now(UTC).isoformat())),
            episodes=int(raw.get("episodes", 0)),
        )

    def _read_metrics_dict(self) -> dict[str, Any]:
        if not self.metrics_path.exists():
            return {}
        try:
            return json.loads(self.metrics_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}

    def record_planner_usage(
        self,
        *,
        mode: str,
        planner_source: str,
        used_world_model: bool,
        total_steps: int,
        world_model_steps: int,
        heuristic_steps: int,
        fallback_triggered: bool = False,
        fallback_reason: str | None = None,
    ) -> None:
        raw = self._read_metrics_dict()
        if not isinstance(raw, dict):
            raw = {}

        planner_usage = raw.get("planner_usage")
        if not isinstance(planner_usage, dict):
            planner_usage = {}

        source = planner_source if planner_source in {"world_model", "heuristic"} else "heuristic"

        episodes = planner_usage.get("episodes")
        if not isinstance(episodes, dict):
            episodes = {}
        episodes[source] = int(episodes.get(source, 0) or 0) + 1
        planner_usage["episodes"] = episodes

        modes = planner_usage.get("modes")
        if not isinstance(modes, dict):
            modes = {}
        mode_entry = modes.get(mode)
        if not isinstance(mode_entry, dict):
            mode_entry = {"world_model": 0, "heuristic": 0}
        mode_entry[source] = int(mode_entry.get(source, 0) or 0) + 1
        modes[mode] = mode_entry
        planner_usage["modes"] = modes

        steps_info = planner_usage.get("steps")
        if not isinstance(steps_info, dict):
            steps_info = {}
        steps_info["world_model"] = int(steps_info.get("world_model", 0) or 0) + int(world_model_steps)
        steps_info["heuristic"] = int(steps_info.get("heuristic", 0) or 0) + int(heuristic_steps)
        planner_usage["steps"] = steps_info

        planner_usage["last"] = {
            "mode": mode,
            "source": source,
            "used_world_model": bool(used_world_model),
            "world_model_steps": int(world_model_steps),
            "heuristic_steps": int(heuristic_steps),
            "total_steps": int(total_steps),
            "ts": datetime.now(UTC).isoformat(),
        }

        if fallback_triggered:
            fallbacks = planner_usage.get("fallbacks")
            if not isinstance(fallbacks, dict):
                fallbacks = {"count": 0, "by_reason": {}}
            fallbacks["count"] = int(fallbacks.get("count", 0) or 0) + 1
            by_reason = fallbacks.get("by_reason")
            if not isinstance(by_reason, dict):
                by_reason = {}
            reason_key = (fallback_reason or "unknown").strip() or "unknown"
            by_reason[reason_key] = int(by_reason.get(reason_key, 0) or 0) + 1
            fallbacks["by_reason"] = by_reason
            planner_usage["fallbacks"] = fallbacks

        raw["planner_usage"] = planner_usage
        self.metrics_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")

    def load_world_model_policy(self) -> dict[tuple[str, str, bool, bool, bool, bool], dict[str, Any]]:
        model_path = self.ws_dir / "world_model.json"
        if not model_path.exists():
            return {}
        try:
            raw = json.loads(model_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
        entries = raw.get("states", [])
        if not isinstance(entries, list):
            return {}
        policy: dict[tuple[str, str, bool, bool, bool, bool], dict[str, Any]] = {}
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            key_visible_flag = bool(entry.get("key_visible", False))
            key = (
                str(entry.get("mode", "curriculum")),
                str(entry.get("room", "")),
                bool(entry.get("door_unlocked", False)),
                bool(entry.get("door_open", False)),
                bool(entry.get("has_key", False)),
                key_visible_flag,
            )
            top_action = entry.get("top_action")
            if isinstance(top_action, dict):
                action_name = _normalize_action_name(top_action.get("name"))
                if not action_name:
                    continue
                target = _normalize_optional(top_action.get("target"))
                item = _normalize_optional(top_action.get("item"))
                count = int(top_action.get("count", 0))
                confidence_val: float | None
                confidence_raw = top_action.get("confidence")
                if confidence_raw is None:
                    total = entry.get("total")
                    try:
                        total_val = float(total) if total is not None else 0.0
                    except (TypeError, ValueError):
                        total_val = 0.0
                    confidence_val = (float(count) / total_val) if total_val else None
                else:
                    try:
                        confidence_val = float(confidence_raw)
                    except (TypeError, ValueError):
                        confidence_val = None
                policy[key] = {
                    "action": action_name,
                    "target": target,
                    "item": item,
                    "confidence": confidence_val,
                    "count": count,
                }
                continue
            top_sig = entry.get("top_action_signature") or entry.get("top_action")
            if isinstance(top_sig, str) and top_sig.strip():
                parts = top_sig.split("|")
                action_name = _normalize_action_name(parts[0] if parts else top_sig)
                target = _normalize_optional(parts[1]) if len(parts) > 1 else None
                item = _normalize_optional(parts[2]) if len(parts) > 2 else None
                policy[key] = {
                    "action": action_name,
                    "target": target,
                    "item": item,
                    "confidence": None,
                    "count": int(entry.get("top_count", 0)),
                }
        return policy

    def _persist_state(self) -> None:
        data = self._env.serialize()
        data["mode"] = self._mode
        with self.state_path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)

    def _load_state(self) -> None:
        if not self.state_path.exists():
            self._persist_state()
            return
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
            self._env.load_state(data)
            self._mode = str(data.get("mode", "curriculum"))
        except (json.JSONDecodeError, OSError):
            self._mode = "curriculum"
            self._env.reset(mode=self._mode)
            self._persist_state()


def run_scripted_episode(
    *,
    workspace: str,
    mode: str,
    seed: int | None,
    artifacts_dir: Path,
    force_disable_world_model: bool = False,
    # When True, aggressively prefer and attempt to use the world model (v2 if available).
    # This does not absolutely forbid heuristic fallback on hard failures, but will try
    # to salvage WM usage and pick a WM-driven action when possible. Intended for
    # WM-engaged recording and WM-only evaluation in curriculum tooling.
    force_world_model: bool = False,
    env_mutator: callable | None = None,
) -> dict[str, Any]:
    controller = get_workspace_sim(workspace, artifacts_dir)
    controller.reset(mode=mode, seed=seed)
    # Allow caller to mutate environment (for diagnostic cases) after reset but before observe/loop
    try:
        if env_mutator:
            env_mutator(controller._env)  # type: ignore[attr-defined]
    except Exception:
        pass
    obs = controller.observe()
    state = obs["state"]
    # Seed recent room history with the initial room
    try:
        recent_rooms = [str(state.get("room", ""))]
    except Exception:
        recent_rooms = []
    visited: set[str] = set()
    # Allow overriding max steps per episode via env
    try:
        import os as _os
        _ms_env = (_os.getenv("SIM_MAX_STEPS") or _os.getenv("BRAIN_WM_MAX_STEPS") or "").strip()
        max_steps = max(1, int(float(_ms_env))) if _ms_env else 16
    except Exception:
        max_steps = 16
    policy_lookup = controller.load_world_model_policy()
    # Optional featureized WM v2
    wm_v2 = controller.load_world_model_v2()
    policy_trace: list[dict[str, Any]] = []
    used_world_model = False
    fallback_triggered = False
    fallback_reason: str | None = None
    world_model_disabled = bool(force_disable_world_model)
    world_model_forced_off = bool(force_disable_world_model)
    last_signature: str | None = None
    consecutive_signature_count = 0
    # Defaults for loop guards when using world model
    max_world_model_repeats = 3
    state_repeat_counts: dict[tuple[str, str, bool, bool, bool, bool], int] = defaultdict(int)
    max_world_model_state_repeats = 3
    result: dict[str, Any] | None = None
    _metrics_counted = False
    used_wm_planner = False

    # Optional per-step sleep to enforce real-time pacing when desired
    try:
        import os as _os
        _sleep_ms_env = _os.getenv("SIM_STEP_SLEEP_MS")
        _sleep_ms = float(_sleep_ms_env) if (_sleep_ms_env is not None and _sleep_ms_env.strip() != "") else 0.0
        if _sleep_ms < 0:
            _sleep_ms = 0.0
    except Exception:
        _sleep_ms = 0.0

    # Preference knobs to increase world model usage when desired
    try:
        import os as _os
        _prefer_wm_raw = (_os.getenv("BRAIN_PREFER_WORLD_MODEL") or "").strip().lower()
        _prefer_world_model = _prefer_wm_raw in {"1", "true", "yes", "y", "on"}
        _wm_conf_env = (_os.getenv("BRAIN_WORLD_MODEL_CONFIDENCE_THRESHOLD") or _os.getenv("BRAIN_HEURISTIC_CONFIDENCE_THRESHOLD") or "").strip()
        _wm_conf_threshold = float(_wm_conf_env) if _wm_conf_env != "" else 0.0
        _wm_rep_env = (_os.getenv("BRAIN_WORLD_MODEL_MAX_REPEATS") or "").strip()
        _wm_state_rep_env = (_os.getenv("BRAIN_WORLD_MODEL_MAX_STATE_REPEATS") or "").strip()
        # Loop-detection toggles
        _disable_loop_raw = (_os.getenv("BRAIN_DISABLE_LOOP_DETECTION") or "").strip().lower()
        _disable_loop_detection = _disable_loop_raw in {"1", "true", "yes", "y", "on"}
        _loop_thresh_env = (_os.getenv("BRAIN_LOOP_DETECTION_THRESHOLD") or "").strip()
        if _wm_rep_env:
            try:
                max_world_model_repeats = max(1, int(float(_wm_rep_env)))
            except (TypeError, ValueError):
                pass
        if _wm_state_rep_env:
            try:
                max_world_model_state_repeats = max(1, int(float(_wm_state_rep_env)))
            except (TypeError, ValueError):
                pass
        if _loop_thresh_env:
            try:
                _loop_t = max(1, int(float(_loop_thresh_env)))
                max_world_model_repeats = _loop_t
                max_world_model_state_repeats = _loop_t
            except (TypeError, ValueError):
                pass
        # If explicitly preferring WM and no custom limits provided, relax guards a bit
        if _prefer_world_model:
            if not _wm_rep_env:
                max_world_model_repeats = max_world_model_repeats + 3
            if not _wm_state_rep_env:
                max_world_model_state_repeats = max_world_model_state_repeats + 3
    except Exception:
        _prefer_world_model = False
        _wm_conf_threshold = 0.0
        _disable_loop_detection = False

    # Planner flags for WM v2
    try:
        import os as _os
        _wm_v2_env = (_os.getenv("BRAIN_WORLD_MODEL_V2") or _os.getenv("BRAIN_WORLD_MODEL_VERSION") or "").strip().lower()
        _use_wm_v2 = (_wm_v2_env in {"2", "v2", "true", "1", "on", "yes"}) and bool(wm_v2)
        _planner_on = ((_os.getenv("BRAIN_WM_PLANNER") or "").strip().lower() in {"1","true","yes","on"}) and _use_wm_v2
        _alpha = float(wm_v2.get("alpha", 1.0)) if isinstance(wm_v2, dict) else 1.0
        _force_wm = ((_os.getenv("BRAIN_FORCE_WORLD_MODEL") or "").strip().lower() in {"1","true","yes","on"}) and _use_wm_v2
    except Exception:
        _use_wm_v2 = False
        _planner_on = False
        _alpha = 1.0
        _force_wm = False

    # Allow explicit parameter to force WM usage regardless of env var.
    # Only meaningful when WM v2 is available and selected.
    if force_world_model and _use_wm_v2 and not _force_wm:
        _force_wm = True

    def _v2_predict_action_sig(_state: dict[str, Any]) -> str | None:
        if not _use_wm_v2 or not isinstance(wm_v2, dict):
            return None
        policy = wm_v2.get("policy") or {}
        if not isinstance(policy, dict):
            return None
        fk = _serialize_feat_key(_feature_key_from_state(_state))
        entry = policy.get(f"f:{fk}")
        def pick_from(entry: dict[str, Any]) -> str | None:
            counts = entry.get("counts") if isinstance(entry, dict) else None
            total = int(entry.get("total", 0)) if isinstance(entry, dict) else 0
            if not isinstance(counts, dict) or not counts:
                return None
            k = len(counts)
            best_sig = None
            best_score = -1.0
            denom = float(total + (_alpha * k)) if k else 1.0
            for sig, c in counts.items():
                try:
                    c_int = int(c)
                except Exception:
                    c_int = 0
                p = (c_int + _alpha) / denom
                # Apply a quality multiplier to discourage regressions
                try:
                    name, target, _ = _parse_signature(sig)
                    room = str(_state.get("room", ""))
                    mult = 1.0
                    if name == "move" and target == room:
                        mult = 0.2
                    if room == "hall" and name == "move" and target == "start":
                        mult = 0.3
                    score = p * mult
                except Exception:
                    score = p
                if score > best_score:
                    best_score = score
                    best_sig = sig
            return best_sig
        sig = pick_from(entry) if entry else None
        if sig is None:
            # Tolerant toggles: try key_visible flips and steps_mod2 flips
            room, has_key, unlocked, open_flag, key_visible, steps_mod2, exits_count = _feature_key_from_state(_state)
            for kv in (False, True):
                alt = (room, has_key, unlocked, open_flag, kv, steps_mod2, exits_count)
                alt_e = policy.get(f"f:{_serialize_feat_key(alt)}")
                sig = pick_from(alt_e) if alt_e else None
                if sig:
                    break
        if sig is None:
            for sm2 in (0, 1):
                alt = (room, has_key, unlocked, open_flag, key_visible, sm2, exits_count)
                alt_e = policy.get(f"f:{_serialize_feat_key(alt)}")
                sig = pick_from(alt_e) if alt_e else None
                if sig:
                    break
        # If still none and forcing WM, try picking top-ranked candidate from any matching tolerant entry
        if sig is None and _force_wm:
            ranked = _v2_rank_action_sigs(_state)
            if ranked:
                sig = ranked[0]
        return sig

    def _v2_rank_action_sigs(_state: dict[str, Any]) -> list[str]:
        """Return action signatures ranked by smoothed probability for current state (fallbacks try kv/s2 toggles)."""
        if not _use_wm_v2 or not isinstance(wm_v2, dict):
            return []
        policy = wm_v2.get("policy") or {}
        if not isinstance(policy, dict):
            return []
        def pairs_from(entry: dict[str, Any]) -> list[tuple[str, float]]:
            if not isinstance(entry, dict):
                return []
            counts = entry.get("counts") if isinstance(entry.get("counts"), dict) else {}
            total = int(entry.get("total", 0))
            if not isinstance(counts, dict) or not counts:
                return []
            k = len(counts)
            denom = float(total + (_alpha * k)) if k else 1.0
            pairs: list[tuple[str, float]] = []
            for sig, c in counts.items():
                try:
                    c_int = int(c)
                except Exception:
                    c_int = 0
                p = (c_int + _alpha) / denom
                # Apply quality multiplier similar to predict()
                try:
                    name, target, _ = _parse_signature(sig)
                    room = str(_state.get("room", ""))
                    mult = 1.0
                    if name == "move" and target == room:
                        mult = 0.2
                    if room == "hall" and name == "move" and target == "start":
                        mult = 0.3
                    p *= mult
                except Exception:
                    pass
                pairs.append((sig, p))
            pairs.sort(key=lambda x: x[1], reverse=True)
            return pairs

        room, has_key, unlocked, open_flag, key_visible, steps_mod2, exits_count = _feature_key_from_state(_state)
        for kv in (key_visible, not key_visible):
            fk = _serialize_feat_key((room, has_key, unlocked, open_flag, kv, steps_mod2, exits_count))
            entry = policy.get(f"f:{fk}")
            pairs = pairs_from(entry)
            if pairs:
                return [s for s, _ in pairs]
        for s2 in (steps_mod2, 1 - steps_mod2):
            fk = _serialize_feat_key((room, has_key, unlocked, open_flag, key_visible, s2, exits_count))
            entry = policy.get(f"f:{fk}")
            pairs = pairs_from(entry)
            if pairs:
                return [s for s, _ in pairs]
        return []

    def _allowed_action_sigs_from_env() -> set[str] | None:
        """Ask env for allowed actions for the next step, as signatures.

        Conventions:
        - If env exposes get_allowed_actions(state) and returns ["*"] or None: unrestricted.
        - If it returns []: no actions allowed (mask all).
        - Otherwise: treat returned strings as allowed signatures.
        """
        try:
            getter = getattr(controller._env, "get_allowed_actions", None)  # type: ignore[attr-defined]
            if callable(getter):
                allowed = getter(state)
                if allowed is None:
                    return None
                if isinstance(allowed, (list, tuple, set)):
                    s = set(str(x) for x in allowed)
                    if "*" in s:
                        return None
                    return set(s)
        except Exception:
            return None
        return None

    def _v2_plan_action_sig(_state: dict[str, Any], goal_room: str, max_depth: int = 3, beam: int = 3) -> str | None:
        if not _planner_on or not isinstance(wm_v2, dict):
            return _v2_predict_action_sig(_state)
        policy = wm_v2.get("policy") or {}
        transitions = wm_v2.get("transitions") or {}
        if not isinstance(policy, dict) or not isinstance(transitions, dict):
            return _v2_predict_action_sig(_state)
        # Planner knobs
        try:
            import os as _os
            _depth_env = (_os.getenv("BRAIN_WM_PLANNER_DEPTH") or "").strip()
            _beam_env = (_os.getenv("BRAIN_WM_PLANNER_BEAM") or "").strip()
            if _depth_env:
                max_depth_local = max(1, int(float(_depth_env)))
            else:
                max_depth_local = max_depth
            if _beam_env:
                beam_local = max(1, int(float(_beam_env)))
            else:
                beam_local = beam
            _priors_on = ((_os.getenv("BRAIN_WM_PLANNER_PRIORS") or "1").strip().lower() in {"1","true","yes","on"})
            _loop_penalty = float((_os.getenv("BRAIN_WM_PLANNER_LOOP_PENALTY") or "0.3").strip())
            _hscale = float((_os.getenv("BRAIN_WM_PLANNER_HEURISTIC_SCALE") or "1.0").strip())
        except Exception:
            max_depth_local = max_depth
            beam_local = beam
            _priors_on = True
            _loop_penalty = 0.3
            _hscale = 1.0

        start_fk = _serialize_feat_key(_feature_key_from_state(_state))

        def best_actions_for_fk(fk: str) -> list[tuple[str, float]]:
            e = policy.get(f"f:{fk}")
            if not isinstance(e, dict):
                counts = {}
                total = 0
            else:
                counts = e.get("counts") if isinstance(e, dict) else None
                total = int(e.get("total", 0))
            if not isinstance(counts, dict):
                counts = {}

            # Optionally inject simple priors to encourage key acquisition & door progression
            if _priors_on:
                # parse feature flags from fk
                flags = {p.split("=",1)[0]: p.split("=",1)[1] for p in fk.split(";") if "=" in p}
                room = flags.get("room", "")
                has_key = flags.get("hk", "0") == "1"
                unlocked = flags.get("u", "0") == "1"
                open_flag = flags.get("o", "0") == "1"
                key_visible = flags.get("kv", "0") == "1"
                # Encourage searching when key not visible and not held
                if not has_key and not key_visible:
                    counts.setdefault("search||", 0)
                    counts["search||"] += int(max(1, _alpha))
                # Encourage pickup when key visible
                if key_visible and not has_key:
                    counts.setdefault("pickup||key", 0)
                    counts["pickup||key"] += int(max(1, 2 * _alpha))
                # Hub priors are goal-aware now
                if room == "hall":
                    if goal_room == "vault":
                        if not unlocked:
                            counts.setdefault("unlock|vault|", 0)
                            counts["unlock|vault|"] += int(max(1, 2 * _alpha))
                        elif not open_flag:
                            counts.setdefault("open|vault|", 0)
                            counts["open|vault|"] += int(max(1, 2 * _alpha))
                        else:
                            counts.setdefault("move|vault|", 0)
                            counts["move|vault|"] += int(max(1, _alpha))
                        if not has_key:
                            counts.setdefault("move|storage|", 0)
                            counts["move|storage|"] += int(max(1, _alpha))
                            counts.setdefault("move|start|", 0)
                            counts["move|start|"] += int(max(1, _alpha))
                    else:
                        counts.setdefault(f"move|{goal_room}|", 0)
                        # Strongly bias direct move to the goal from the hub when goal != vault
                        counts[f"move|{goal_room}|"] += int(max(3, 5 * _alpha))
                # Move to hub when elsewhere
                if room not in ("hall", "vault"):
                    counts.setdefault("move|hall|", 0)
                    counts["move|hall|"] += int(max(1, _alpha))

            k = len(counts)
            denom = float(total + (_alpha * k)) if k else 1.0
            pairs: list[tuple[str, float]] = []
            for sig, c in counts.items():
                try:
                    c_int = int(c)
                except Exception:
                    c_int = 0
                p = (c_int + _alpha) / denom
                pairs.append((sig, p))
            pairs.sort(key=lambda x: x[1], reverse=True)
            # Ensure goal-directed move from the hub is considered when goal != vault
            try:
                flags = {p.split("=",1)[0]: p.split("=",1)[1] for p in fk.split(";") if "=" in p}
                room = flags.get("room", "")
            except Exception:
                room = ""
            if room == "hall" and goal_room != "vault":
                sig_goal = f"move|{goal_room}|"
                if all(s != sig_goal for s,_ in pairs):
                    c_val = counts.get(sig_goal, 0)
                    try:
                        c_int = int(c_val)
                    except Exception:
                        c_int = 0
                    p_ins = (c_int + _alpha) / (float(total + (_alpha * max(1, len(counts)))))
                    pairs.append((sig_goal, p_ins))
                    pairs.sort(key=lambda x: x[1], reverse=True)
            return pairs[:beam_local]

        def room_from_fk(fk: str) -> str:
            try:
                for part in fk.split(";"):
                    if part.startswith("room="):
                        return part.split("=", 1)[1]
            except Exception:
                pass
            return ""

        def flags_from_fk(fk: str) -> dict[str, Any]:
            out = {"room": room_from_fk(fk), "has_key": False, "unlocked": False, "open": False, "key_visible": False}
            try:
                for part in fk.split(";"):
                    if part.startswith("hk="):
                        out["has_key"] = (part.split("=",1)[1] == "1")
                    elif part.startswith("u="):
                        out["unlocked"] = (part.split("=",1)[1] == "1")
                    elif part.startswith("o="):
                        out["open"] = (part.split("=",1)[1] == "1")
                    elif part.startswith("kv="):
                        out["key_visible"] = (part.split("=",1)[1] == "1")
            except Exception:
                pass
            return out

        def heuristic_bonus(fk: str) -> float:
            f = flags_from_fk(fk)
            bonus = 0.0
            if f.get("room") == goal_room:
                bonus += 1.0
            # Only apply vault door shaping when vault is the goal
            if goal_room == "vault":
                if f.get("has_key"):
                    bonus += 0.7
                if f.get("unlocked"):
                    bonus += 0.3
                if f.get("open"):
                    bonus += 0.3
                if f.get("key_visible") and not f.get("has_key"):
                    bonus += 0.1
            return _hscale * bonus

        # Beam search over feature keys
        Node = tuple[str, float, list[str], tuple[str, ...]]  # (fk, score, path of sigs, visited fks)
        frontier: list[Node] = [(start_fk, 0.0, [], (start_fk,))]
        for depth in range(max_depth_local):
            new_frontier: list[Node] = []
            for fk, score, path, visited_fks in frontier:
                if room_from_fk(fk) == goal_room:
                    return path[0] if path else _v2_predict_action_sig(_state)
                for sig, p in best_actions_for_fk(fk):
                    trans_key = f"f:{fk}|{sig}"
                    tentry = transitions.get(trans_key)
                    if not isinstance(tentry, dict):
                        continue
                    nexts = tentry.get("next") if isinstance(tentry.get("next"), dict) else None
                    if not nexts:
                        continue
                    # Expand a few likely next feature keys
                    sorted_nexts = sorted(nexts.items(), key=lambda kv: int(kv[1]) if kv[1] is not None else 0, reverse=True)[:3]
                    for nfk, cnt in sorted_nexts:
                        step_cost = -math.log(max(p, 1e-6))
                        hcost = -heuristic_bonus(nfk)
                        loop_cost = _loop_penalty if nfk in visited_fks else 0.0
                        new_score = score + step_cost + hcost + loop_cost
                        new_path = path + [sig]
                        new_frontier.append((nfk, new_score, new_path, tuple(list(visited_fks) + [nfk])))
            if not new_frontier:
                break
            # Keep top beam
            new_frontier.sort(key=lambda n: n[1])
            frontier = new_frontier[:beam_local]
        # Fallback: 1-step pick
        return _v2_predict_action_sig(_state)

    # Substitution policy loader and helper
    def _load_substitution_policy() -> dict[str, Any]:
        try:
            path = controller.ws_dir / "substitution_policy.json"
            if path.exists():
                raw = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    return raw
        except Exception:
            return {}
        return {}

    _sub_policy = _load_substitution_policy()

    def _load_constraint_policy() -> dict[str, Any]:
        try:
            path = controller.ws_dir / "constraint_policy.json"
            if path.exists():
                raw = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    return raw
        except Exception:
            return {}
        return {}

    _constraint_policy = _load_constraint_policy()

    def _substitute_sig(desired_sig: str | None, fk: str | None, allowed: set[str] | None) -> str | None:
        if not desired_sig or not isinstance(_sub_policy, dict):
            return None
        if allowed is not None and not allowed:
            return None
        candidates: list[str] = []
        # 1) Constraint policy (learned) has priority: prefer recommended by success rate
        try:
            if isinstance(_constraint_policy, dict):
                # by_fk
                if fk and isinstance(_constraint_policy.get("by_fk"), dict):
                    rec = _constraint_policy["by_fk"].get(fk, {}).get(desired_sig)
                    if isinstance(rec, list):
                        for s in rec:
                            candidates.append(str(s))
                # global fallback
                if isinstance(_constraint_policy.get("global"), dict):
                    recg = _constraint_policy["global"].get(desired_sig)
                    if isinstance(recg, list):
                        for s in recg:
                            candidates.append(str(s))
        except Exception:
            pass
        try:
            if fk and isinstance(_sub_policy.get("by_fk"), dict):
                lst = _sub_policy["by_fk"].get(fk, {}).get(desired_sig)
                if isinstance(lst, list):
                    candidates.extend([str(x) for x in lst])
            if isinstance(_sub_policy.get("global"), dict):
                lst = _sub_policy["global"].get(desired_sig)
                if isinstance(lst, list):
                    candidates.extend([str(x) for x in lst])
        except Exception:
            candidates = []
        # Deduplicate while preserving order
        seen: set[str] = set()
        ordered: list[str] = []
        for s in candidates:
            if s not in seen:
                seen.add(s)
                ordered.append(s)
        # pick first allowed
        for s in ordered:
            if allowed is None or s in allowed:
                return s
        return None

    # Episode-scoped memory: map (feature_key, desired_sig) -> last successful alternative sig
    recent_success_alts: dict[tuple[str, str], str] = {}

    # Progress-oriented ranking for alternatives: lower is better
    def _progress_rank(sig: str, st: dict[str, Any]) -> int:
        try:
            name, target, item = _parse_signature(sig)
            room = str(st.get("room"))
            inv = set(st.get("inventory") or [])
            door = st.get("door") or {}
            has_key = ("key" in inv)
            sub = _current_subgoal(st, controller._env)
            # Strongly penalize regression to start
            if name == "move" and target == "start":
                return 5
            # Avoid no-op move
            if name == "move" and target == room:
                return 4
            # If we're in hall with key and unlock/open is blocked, prefer exploring storage over start
            if room == "hall" and has_key and name == "move":
                if target == "storage":
                    return 0
                if target == "hall":
                    return 3
            # Subgoal-aligned actions get priority
            if sub == "acquire_key":
                if name in {"search", "pickup"}:
                    return 0
                if name == "move" and target == "storage":
                    return 1
            elif sub == "reach_hall":
                if name == "move" and target == "hall":
                    return 0
            elif sub == "unlock":
                if name == "unlock":
                    return 0
            elif sub == "open":
                if name == "open":
                    return 0
            elif sub == "enter_vault":
                if name == "move" and target == "vault":
                    return 0
            # Default mild cost for other safe non-choke moves
            if name == "move":
                return 2
            # search/pickup outside of acquire_key still acceptable
            if name in {"search", "pickup"}:
                return 2
        except Exception:
            return 3
        return 3

    def _progress(prev: dict[str, Any], cur: dict[str, Any]) -> bool:
        try:
            if (prev or {}).get("room") != (cur or {}).get("room"):
                return True
            pinv = set((prev or {}).get("inventory") or [])
            cinv = set((cur or {}).get("inventory") or [])
            if ("key" in cinv) and ("key" not in pinv):
                return True
            pdoor = (prev or {}).get("door") or {}
            cdoor = (cur or {}).get("door") or {}
            if bool(cdoor.get("unlocked")) and not bool(pdoor.get("unlocked")):
                return True
            if bool(cdoor.get("open")) and not bool(pdoor.get("open")):
                return True
        except Exception:
            return False
        return False

    # Persistent workaround memory (across episodes) stored under ws_dir
    _wmemory_path = controller.ws_dir / "workaround_memory.json"

    def _load_workaround_memory() -> dict[str, Any]:
        try:
            if _wmemory_path.exists():
                raw = json.loads(_wmemory_path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    return raw
        except Exception:
            return {}
        return {}

    def _save_workaround_memory(mem: dict[str, Any]) -> None:
        try:
            _wmemory_path.write_text(json.dumps(mem, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    _wmemory = _load_workaround_memory()

    _CHOKES = {"unlock|vault|", "open|vault|", "move|vault|"}

    # Episode-local history of alternative signatures to enforce diversity
    recent_alternative_sigs: list[str] = []
    # Episode-local recent rooms to help break oscillations (e.g., hall⇄storage)
    recent_rooms: list[str] = []

    def _intelligent_alternatives(st: dict[str, Any], allowed: set[str] | None, desired_sig: str | None, masked_list: list[str]) -> list[str]:
        """Progress-aware, context-sensitive alternative proposals.

        Prefers time-kill or exploration when door operations are masked to avoid
        hall<->storage oscillation; prefers moving toward key when not holding it.
        Returns a prioritized list of signatures (not filtered for diversity).
        """
        room = str(st.get("room", ""))
        inv = set(st.get("inventory") or [])
        has_key = ("key" in inv)
        alts: list[str] = []
        allowed_set = set(allowed) if allowed is not None else None

        def add_if(sig: str):
            if allowed_set is None or sig in allowed_set:
                alts.append(sig)

        # If chokes are masked in hall while holding key, avoid regressive storage loops.
        if room == "hall" and has_key and any(m in masked_list for m in _CHOKES):
            # Prefer moving to start first; only consider storage as last resort,
            # and avoid immediately repeating the most recent alternative signature.
            try:
                prev_room = recent_rooms[-2] if len(recent_rooms) >= 2 else None
            except Exception:
                prev_room = None
            # Time-kill is often not in allow-list here; include if allowed
            add_if("search||")
            # If we just came from storage, do not send back to storage again
            if prev_room == "storage":
                add_if("move|start|")
                # Defer storage to the very end
                if (allowed_set is None or "move|storage|" in allowed_set):
                    alts.append("move|storage|")
            else:
                # Default preference: try start before storage
                add_if("move|start|")
                add_if("move|storage|")
            return alts

        # If in hall without key, favor moving to storage to acquire key, then search
        if room == "hall" and not has_key:
            add_if("move|storage|")
            add_if("search||")
            add_if("move|start|")
            return alts

        # In storage: if key not held, search; if held, return to hall.
        if room == "storage":
            if not has_key:
                add_if("search||")
            add_if("move|hall|")
            return alts

        # Default: prefer benign search, then any non-regressive moves present in allowed
        add_if("search||")
        if allowed_set is not None:
            for s in sorted(allowed_set):
                if s.startswith("move|") and s != f"move|{room}|":
                    alts.append(s)
        return alts

    def _pick_from_priority(priority: list[str], allowed: set[str] | None, desired_sig: str | None, *, force_alt: bool, diversity_k: int = 3) -> str | None:
        """Pick the first viable candidate from priority list respecting allow-list,
        force_alt, and a small diversity window over recent_alternative_sigs.
        """
        allowed_set = set(allowed) if allowed is not None else None
        recent = set(recent_alternative_sigs[-diversity_k:])
        for sig in priority:
            if allowed_set is not None and sig not in allowed_set:
                continue
            if force_alt and desired_sig and sig == desired_sig:
                continue
            if sig in recent:
                continue
            return sig
        return None

    def _constraint_context(st: dict[str, Any], allowed: set[str] | None) -> str | None:
        try:
            if allowed is None:
                return None
            masked = sorted([c for c in _CHOKES if c not in allowed])
            if not masked:
                return None
            room = str(st.get("room", ""))
            inv = set(st.get("inventory") or [])
            hk = 1 if ("key" in inv) else 0
            door = st.get("door") or {}
            u = 1 if bool(door.get("unlocked")) else 0
            o = 1 if bool(door.get("open")) else 0
            return f"room={room};hk={hk};u={u};o={o};masked={','.join(masked)}"
        except Exception:
            return None

    def _constraint_plan(st: dict[str, Any], masked: list[str]) -> list[str]:
        # Heuristic constraint-aware sequences; prefer exploration without regression
        room = str(st.get("room", ""))
        inv = set(st.get("inventory") or [])
        has_key = ("key" in inv)
        door = st.get("door") or {}
        unlocked = bool(door.get("unlocked"))
        open_flag = bool(door.get("open"))
        plan: list[str] = []
        # Common helpers
        def mv(x: str) -> str:
            return f"move|{x}|"
        # Case: in hall with key, unlock masked
        if room == "hall" and has_key and "unlock|vault|" in masked:
            # Avoid regressing into hall⇄storage oscillations; try start first, then return
            plan = [mv("start"), mv("hall"), "unlock|vault|"]
            return plan
        # Case: in hall with key, open masked
        if room == "hall" and has_key and unlocked and ("open|vault|" in masked):
            # Prefer a brief detour to start to break cycles
            plan = [mv("start"), mv("hall"), "open|vault|"]
            return plan
        # Case: move to vault masked after open
        if room == "hall" and has_key and unlocked and open_flag and ("move|vault|" in masked):
            # time-kill safely without regressing
            plan = ["search||", mv("storage"), mv("hall"), mv("vault")]
            return plan
        # Default: safe exploration without regression
        if room == "hall":
            plan = ["search||", mv("start"), mv("hall")]
            return plan
        if room == "storage":
            plan = ["search||", mv("hall")]
            return plan
        # Fallback minimal
        return ["search||"]

    for step_idx in range(max_steps):
        # Per-step alternative bookkeeping (local; controller clears its flags inside act())
        was_alt_step = False
        desired_before_alt: str | None = None
        # For world-model lookup, optionally normalize mode to the runner's requested mode
        lookup_mode = state.get("mode", mode)
        try:
            if _prefer_world_model:
                lookup_mode = mode or lookup_mode
        except Exception:
            pass
        state_key = controller._state_key(lookup_mode, state)
        predicted = policy_lookup.get(state_key)
        # WM v2 candidate signature (action|target|item)
        v2_sig: str | None = None
        if _use_wm_v2 and not world_model_disabled:
            try:
                goal_room = getattr(controller._env, "goal_room", "vault")
                if _planner_on:
                    v2_sig = _v2_plan_action_sig(state, goal_room)
                    if v2_sig:
                        used_wm_planner = True
                else:
                    v2_sig = _v2_predict_action_sig(state)
            except Exception:
                v2_sig = None
            # In force mode, if no v2_sig, attempt to pick any plausible alternative
            if (not v2_sig) and _force_wm:
                ranked_any = _v2_rank_action_sigs(state)
                if ranked_any:
                    v2_sig = ranked_any[0]
        # If preferring WM and there is no exact match, try a tolerant lookup toggling key_visible
        if predicted is None:
            try:
                if _prefer_world_model:
                    # state_key: (mode, room, unlocked, open_flag, has_key, key_visible)
                    if isinstance(state_key, tuple) and len(state_key) == 6:
                        base = list(state_key)
                        # Try with key_visible=False then True to improve hit rate
                        for kv in (False, True):
                            base[5] = kv
                            alt_key = (base[0], base[1], base[2], base[3], base[4], base[5])
                            predicted = policy_lookup.get(alt_key)
                            if predicted is not None:
                                break
            except Exception:
                pass
        # If still no exact match and preferring WM, try an approximate match by room and possession
        if predicted is None:
            try:
                if _prefer_world_model and isinstance(state_key, tuple) and len(state_key) == 6:
                    _mode, _room, _unlocked, _open_flag, _has_key, _kvis = state_key
                    best = None
                    best_count = -1
                    for k, v in policy_lookup.items():
                        if not isinstance(k, tuple) or len(k) != 6:
                            continue
                        kmode, kroom, kunlock, kopen, khas, kvis = k
                        if kmode != lookup_mode:
                            continue
                        if kroom != _room:
                            continue
                        # Prefer same possession state; tolerate differences in door flags and visibility
                        if bool(khas) != bool(_has_key):
                            continue
                        cnt = int(v.get("count", 0)) if isinstance(v, dict) else 0
                        if cnt > best_count:
                            best = v
                            best_count = cnt
                    if best is not None:
                        predicted = best
            except Exception:
                pass
        candidate_action = _normalize_action_name(predicted.get("action")) if predicted else None
        candidate_target = _normalize_optional(predicted.get("target")) if predicted else None
        candidate_item = _normalize_optional(predicted.get("item")) if predicted else None
        if v2_sig and not candidate_action:
            ak = _parse_signature(v2_sig)
            candidate_action, candidate_target, candidate_item = ak[0], ak[1], ak[2]
        action_name: str | None = None
        target_val: str | None = None
        item_val: str | None = None
        source = "heuristic"
        fallback_label: str | None = None
        world_model_signature: str | None = None
        # Internal flag to skip executing WM act when we've deliberately abstained (e.g., due to mask)
        skip_wm_act = False

        # Honor optional confidence threshold if provided
        if predicted and not world_model_disabled:
            try:
                _conf = predicted.get("confidence") if isinstance(predicted, dict) else None
                if _conf is not None and _wm_conf_threshold and float(_conf) < float(_wm_conf_threshold):
                    # Treat as unavailable if below threshold
                    predicted = None
            except Exception:
                pass

        if (predicted or v2_sig) and not world_model_disabled:
            if not _disable_loop_detection:
                state_repeat_counts[state_key] += 1
                if state_repeat_counts[state_key] > max_world_model_state_repeats:
                    world_model_disabled = True
                    fallback_triggered = True
                    if fallback_reason is None:
                        fallback_reason = "loop_detected"
                    fallback_label = "loop_detected"

        if (predicted or v2_sig) and not world_model_disabled:
            world_model_signature = _action_signature(
                _action_key(candidate_action, candidate_target, candidate_item),
            ) if candidate_action else None

            if not candidate_action or not _action_has_args(candidate_action, candidate_target, candidate_item):
                # If forcing WM, try to salvage by picking an alternative with valid args
                if _force_wm:
                    salvage: str | None = None
                    try:
                        ranked_all = _v2_rank_action_sigs(state)
                        for s in ranked_all:
                            an, at, it = _parse_signature(s)
                            if _action_has_args(an, at, it):
                                salvage = s
                                break
                    except Exception:
                        salvage = None
                    if salvage:
                        ak = _parse_signature(salvage)
                        candidate_action, candidate_target, candidate_item = ak[0], ak[1], ak[2]
                        world_model_signature = salvage
                    else:
                        world_model_disabled = True
                        fallback_triggered = True
                        if fallback_reason is None:
                            fallback_reason = "invalid_args"
                        fallback_label = "invalid_args"
                else:
                    world_model_disabled = True
                    fallback_triggered = True
                    if fallback_reason is None:
                        fallback_reason = "invalid_args"
                    fallback_label = "invalid_args"
            else:
                if not _disable_loop_detection:
                    if world_model_signature == last_signature:
                        consecutive_signature_count += 1
                    else:
                        last_signature = world_model_signature
                        consecutive_signature_count = 1
                    if consecutive_signature_count > max_world_model_repeats:
                        world_model_disabled = True
                        fallback_triggered = True
                        if fallback_reason is None:
                            fallback_reason = "loop_detected"
                        fallback_label = "loop_detected"
                # Enforce adversarial mask at boundary if provided by env
                allowed = _allowed_action_sigs_from_env()
                # Optional per-step debug trace of mask evaluation (pre-plan), even if no mask provider
                try:
                    import os as _os
                    if (_os.getenv("BRAIN_MASK_DEBUG") or "").strip().lower() in {"1","true","yes","on"}:
                        controller._append_event({
                            "event": "mask_debug",
                            "mode": controller._mode,
                            "phase": "planner_boundary",
                            "feature_key": _serialize_feat_key(_feature_key_from_state(state)),
                            "desired_sig": world_model_signature,
                            "allowed": list(allowed) if allowed is not None else ["*"],
                        })
                except Exception:
                    pass
                if allowed is not None:
                    # Masking diagnostics: record allowed() probing and shape
                    try:
                        raw = controller._read_metrics_dict()
                        if not isinstance(raw, dict):
                            raw = {}
                        md = raw.setdefault("masking", {})
                        md["allowed_probed_steps"] = int(md.get("allowed_probed_steps", 0) or 0) + 1
                        if len(allowed) == 0:
                            md["allowed_mask_all_steps"] = int(md.get("allowed_mask_all_steps", 0) or 0) + 1
                        else:
                            md["allowed_nonempty_steps"] = int(md.get("allowed_nonempty_steps", 0) or 0) + 1
                            # Selective masking detection: non-empty allow-list without wildcard
                            if "*" not in allowed:
                                md["selective_blocks_applied"] = int(md.get("selective_blocks_applied", 0) or 0) + 1
                                # Choke-action specific masking: count steps where any choke sig is excluded
                                _chokes = {"unlock|vault|", "open|vault|", "move|vault|"}
                                if any((c not in allowed) for c in _chokes):
                                    md["choke_actions_masked_steps"] = int(md.get("choke_actions_masked_steps", 0) or 0) + 1
                        controller.metrics_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
                    except Exception:
                        pass
                    # Per-step debug trace already emitted above
                    # If current signature not allowed, try workaround memory/plan, WM v2 alternatives, or abstain
                    force_alt = False
                    try:
                        import os as _os
                        force_alt = ((_os.getenv("BRAIN_MASK_FORCE_ALTERNATIVE") or "").strip().lower() in {"1","true","yes","on"})
                    except Exception:
                        force_alt = False
                    # Capture desired vs allowed snapshot for later trace emission
                    _desired_allowed_snapshot = None
                    try:
                        if world_model_signature is not None:
                            _desired_allowed_snapshot = (world_model_signature in allowed)
                    except Exception:
                        _desired_allowed_snapshot = None
                    if world_model_signature is None or (world_model_signature not in allowed) or force_alt:
                        picked: str | None = None
                        # Constraint-aware planning: memory -> heuristic plan
                        try:
                            ctx = _constraint_context(state, allowed)
                        except Exception:
                            ctx = None
                        memory_hit = False
                        planned: list[str] = []
                        masked_list = []
                        try:
                            masked_list = sorted([c for c in _CHOKES if c not in (allowed or set())])
                        except Exception:
                            masked_list = []
                        if ctx:
                            # In hall with key and chokes masked, prefer fresh constraint plan before memory to avoid stale loops.
                            try:
                                _room_now = str(state.get("room", ""))
                                _has_key_now = ("key" in set(state.get("inventory") or []))
                                prefer_plan_first = (_room_now == "hall" and _has_key_now and bool(masked_list))
                            except Exception:
                                prefer_plan_first = False
                            if prefer_plan_first:
                                seq2 = _constraint_plan(state, masked_list)
                                for s in seq2:
                                    if s in set(recent_alternative_sigs[-3:]):
                                        continue
                                    # Apply anti-oscillation guard for plan suggestions
                                    try:
                                        _prev_room = recent_rooms[-2] if len(recent_rooms) >= 2 else None
                                        if (_prev_room == "storage" and s == "move|storage|"):
                                            continue
                                    except Exception:
                                        pass
                                    if (allowed is None or s in allowed) and (not force_alt or s != world_model_signature):
                                        picked = s
                                        planned = list(seq2)
                                        break
                            if picked is None:
                                seq = _wmemory.get(ctx)
                                if isinstance(seq, list) and seq:
                                    # Use persisted sequence, pick first allowed step distinct from desired when forcing alt
                                    for s in seq:
                                        # Skip very recent alternatives to improve diversity/avoid oscillation
                                        if s in set(recent_alternative_sigs[-3:]):
                                            continue
                                        # Avoid hall(hk=1) → storage when we just came from storage
                                        try:
                                            _prev_room = recent_rooms[-2] if len(recent_rooms) >= 2 else None
                                            if (_room_now == "hall" and _has_key_now and _prev_room == "storage" and s == "move|storage|"):
                                                continue
                                        except Exception:
                                            pass
                                        if (s in allowed) and (not force_alt or s != world_model_signature):
                                            picked = s
                                            planned = list(seq)
                                            memory_hit = True
                                            break
                            if picked is None and not prefer_plan_first:
                                seq2 = _constraint_plan(state, masked_list)
                                for s in seq2:
                                    if s in set(recent_alternative_sigs[-3:]):
                                        continue
                                    # Apply the same anti-oscillation guard for plan suggestions
                                    try:
                                        _prev_room = recent_rooms[-2] if len(recent_rooms) >= 2 else None
                                        if (_prev_room == "storage" and s == "move|storage|"):
                                            continue
                                    except Exception:
                                        pass
                                    if (allowed is None or s in allowed) and (not force_alt or s != world_model_signature):
                                        picked = s
                                        planned = list(seq2)
                                        break
                        # Priority: intelligent alternatives to avoid regressive loops under masks
                        if picked is None:
                            try:
                                priority = _intelligent_alternatives(state, allowed, world_model_signature, masked_list)
                            except Exception:
                                priority = []
                            if priority:
                                cand = _pick_from_priority(priority, allowed, world_model_signature, force_alt=bool(force_alt))
                                if cand:
                                    picked = cand
                        if picked and ctx:
                            # Note pending context to write memory upon progress
                            try:
                                controller._pending_workaround_context = ctx
                                controller._pending_workaround_plan = planned
                            except Exception:
                                pass
                            # Debug event for workaround plan
                            try:
                                import os as _os
                                if (_os.getenv("BRAIN_MASK_DEBUG") or "").strip().lower() in {"1","true","yes","on"}:
                                    controller._append_event({
                                        "event": "mask_debug",
                                        "mode": controller._mode,
                                        "phase": "workaround_plan",
                                        "feature_key": _serialize_feat_key(_feature_key_from_state(state)),
                                        "context": ctx,
                                        "plan": planned,
                                        "memory_hit": bool(memory_hit),
                                    })
                            except Exception:
                                pass
                        # Memory-first: reuse last successful alternative if available
                        try:
                            fk = _serialize_feat_key(_feature_key_from_state(state))
                            mem = recent_success_alts.get((fk, str(world_model_signature)))
                            if mem and (allowed is None or mem in allowed):
                                picked = mem
                        except Exception:
                            pass
                        if _use_wm_v2:
                            # Rank WM v2 candidates by progress, then sample from top-K
                            ranked = [s for s in _v2_rank_action_sigs(state) if (allowed is None or s in allowed)]
                            # Remove desired if forcing alternative
                            if force_alt and world_model_signature:
                                ranked = [s for s in ranked if s != world_model_signature]
                            # Sort by progress rank (lower is better), stable on original prob order
                            ranked.sort(key=lambda s: _progress_rank(s, state))
                            # Optional epsilon exploration among top candidates
                            try:
                                import os as _os
                                import random as _rnd
                                _eps = float((_os.getenv("BRAIN_MASK_EXPLORE_EPS") or "0.1").strip())
                            except Exception:
                                _eps = 0.1
                            topk = ranked[:3]
                            if topk:
                                picked = _rnd.choice(topk) if (len(topk) > 1 and _eps > 0) else topk[0]
                            # If we're forcing an alternative but the best equals desired, shift to next
                            if force_alt and picked == world_model_signature and len(ranked) > 1:
                                for s in ranked[1:]:
                                    if s in allowed and s != world_model_signature:
                                        picked = s
                                        break
                        if picked is None:
                            # Try learned substitution policy
                            picked = _substitute_sig(world_model_signature, _serialize_feat_key(_feature_key_from_state(state)), allowed)
                        if picked:
                            ak = _parse_signature(picked)
                            _prev_state = dict(state)
                            # Capture desired BEFORE overwriting with picked
                            desired_sig_save = world_model_signature
                            candidate_action, candidate_target, candidate_item = ak[0], ak[1], ak[2]
                            world_model_signature = picked
                            # Mark that we're executing an alternative to the desired plan
                            try:
                                controller._action_was_alternative = True
                                controller._desired_signature_for_alt = str(desired_sig_save) if desired_sig_save else None
                            except Exception:
                                pass
                            # Record in episode-local diversity memory
                            try:
                                recent_alternative_sigs.append(str(world_model_signature))
                            except Exception:
                                pass
                            # Local flags for trace
                            was_alt_step = True
                            desired_before_alt = str(desired_sig_save) if desired_sig_save else None
                            # Emit debug showing forced or masked alternative choice
                            try:
                                import os as _os
                                if (_os.getenv("BRAIN_MASK_DEBUG") or "").strip().lower() in {"1","true","yes","on"}:
                                    controller._append_event({
                                        "event": "mask_debug",
                                        "mode": controller._mode,
                                        "phase": "planner_boundary",
                                        "feature_key": _serialize_feat_key(_feature_key_from_state(state)),
                                        "desired_sig": desired_sig_save,
                                        "allowed": list(allowed) if allowed is not None else ["*"],
                                        "force_alternative": bool(force_alt),
                                        "chosen_signature": world_model_signature,
                                    })
                            except Exception:
                                pass
                            # Metric: successful workaround chosen at plan-time (world model path)
                            try:
                                raw = controller._read_metrics_dict()
                                if not isinstance(raw, dict):
                                    raw = {}
                                md = raw.setdefault("masking", {})
                                md["workarounds_success_total"] = int(md.get("workarounds_success_total", 0) or 0) + 1
                                controller.metrics_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
                            except Exception:
                                pass
                        else:
                            # Abstain due to full mask
                            source = "world_model"
                            used_world_model = True
                            fallback_label = "masked_enforced"
                            # Emit an abstain no-op result and clear any sticky mask flag on env
                            try:
                                ob = controller.observe()
                            except Exception:
                                ob = {"state": state, "observation": ""}
                            try:
                                if hasattr(controller._env, "_mask_next"):
                                    delattr(controller._env, "_mask_next")  # type: ignore[attr-defined]
                            except Exception:
                                pass
                            # Use selective abstain message when allow-list is non-empty
                            _msg = "abstain_masked"
                            try:
                                if len(allowed) == 0:
                                    _msg = "abstain_masked_all"
                            except Exception:
                                _msg = "abstain_masked"
                            # penalties configurable via env
                            try:
                                import os as _os
                                _pen_sel = float((_os.getenv("BRAIN_MASK_PENALTY_SELECTIVE") or "-0.02").strip())
                                _pen_all = float((_os.getenv("BRAIN_MASK_PENALTY_ALL") or "-0.08").strip())
                            except Exception:
                                _pen_sel, _pen_all = -0.02, -0.08
                            result = {
                                "observation": ob.get("observation", ""),
                                "state": ob.get("state", state),
                                "reward": (_pen_all if _msg == "abstain_masked_all" else _pen_sel),
                                "done": False,
                                "success": False,
                                "message": _msg,
                            }
                            # Metrics: count planning-time abstentions due to mask
                            try:
                                raw = controller._read_metrics_dict()
                                if not isinstance(raw, dict):
                                    raw = {}
                                md = raw.setdefault("masking", {})
                                md["abstain_due_to_mask_total"] = int(md.get("abstain_due_to_mask_total", 0) or 0) + 1
                                md["planning_abstains_total"] = int(md.get("planning_abstains_total", 0) or 0) + 1
                                controller.metrics_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
                            except Exception:
                                pass
                            # Also record a synthetic act event for harvesting with the desired (masked) signature
                            try:
                                des_act: str | None = None
                                des_t: str | None = None
                                des_i: str | None = None
                                if world_model_signature:
                                    ak = _parse_signature(world_model_signature)
                                    des_act, des_t, des_i = ak[0], ak[1], ak[2]
                                controller._append_event({
                                    "event": "act",
                                    "mode": controller._mode,
                                    "action": des_act,
                                    "target": des_t,
                                    "item": des_i,
                                    "state": result.get("state", state),
                                    "reward": result.get("reward", -0.01),
                                    "done": result.get("done", False),
                                    "success": result.get("success", False),
                                    "message": result.get("message", _msg),
                                })
                            except Exception:
                                pass
                            action_name = None  # prevent act below
                            skip_wm_act = True
                else:
                    # No mask provider; optionally force an alternative for instrumentation visibility
                    try:
                        import os as _os
                        force_alt = ((_os.getenv("BRAIN_MASK_FORCE_ALTERNATIVE") or "").strip().lower() in {"1","true","yes","on"})
                    except Exception:
                        force_alt = False
                    if force_alt and world_model_signature:
                        # Try to pick a different progress-oriented alternative even without an allow-list
                        picked: str | None = None
                        if _use_wm_v2:
                            ranked_all = [s for s in _v2_rank_action_sigs(state) if s != world_model_signature]
                            ranked_all.sort(key=lambda s: _progress_rank(s, state))
                            topk = ranked_all[:3]
                            if topk:
                                import random as _rnd
                                picked = _rnd.choice(topk)
                        # Fallback to substitution policy
                        if picked is None:
                            picked = _substitute_sig(world_model_signature, _serialize_feat_key(_feature_key_from_state(state)), None)
                        if picked and picked != world_model_signature:
                            desired_sig_save = world_model_signature
                            ak = _parse_signature(picked)
                            candidate_action, candidate_target, candidate_item = ak[0], ak[1], ak[2]
                            world_model_signature = picked
                            # Mark alternative for metrics and harvesting
                            try:
                                controller._action_was_alternative = True
                                controller._desired_signature_for_alt = str(desired_sig_save) if desired_sig_save else None
                            except Exception:
                                pass
                            was_alt_step = True
                            desired_before_alt = str(desired_sig_save) if desired_sig_save else None
                            # Metrics: count workaround
                            try:
                                raw = controller._read_metrics_dict()
                                if not isinstance(raw, dict):
                                    raw = {}
                                md = raw.setdefault("masking", {})
                                md["workarounds_success_total"] = int(md.get("workarounds_success_total", 0) or 0) + 1
                                controller.metrics_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
                            except Exception:
                                pass
                            # Emit a debug event showing forced alternative even without mask
                            try:
                                import os as _os
                                if (_os.getenv("BRAIN_MASK_DEBUG") or "").strip().lower() in {"1","true","yes","on"}:
                                    controller._append_event({
                                        "event": "mask_debug",
                                        "mode": controller._mode,
                                        "phase": "planner_forced_alt",
                                        "feature_key": _serialize_feat_key(_feature_key_from_state(state)),
                                        "desired_sig": desired_sig_save,
                                        "picked_sig": world_model_signature,
                                        "allowed": ["*"],
                                        "force_alternative": True,
                                        "chosen_signature": world_model_signature,
                                    })
                            except Exception:
                                pass
                if not world_model_disabled:
                    try:
                        # Optional real-time pacing before execution
                        if _sleep_ms:
                            time.sleep(_sleep_ms / 1000.0)
                        if skip_wm_act:
                            # If abstained above, skip act
                            raise RuntimeError("masked_enforced_abstain")
                        _prev_state = dict(state)
                        result = controller.act(candidate_action, target=candidate_target, item=candidate_item)
                        # If action was externally masked, optionally retry with next-best signature once.
                        # Enforce mask at planner boundary: if masked, abstain and log.
                        if isinstance(result, dict) and str(result.get("message", "")) == "masked_action":
                            # Increment masked_denials_total and notate bypass=0 for this step
                            try:
                                raw = controller._read_metrics_dict()
                                if not isinstance(raw, dict):
                                    raw = {}
                                m = raw.setdefault("masked_counters", {})
                                m["masked_denials_total"] = int(m.get("masked_denials_total", 0) or 0) + 1
                                # Also track unified abstain counter under masking diagnostics
                                md = raw.setdefault("masking", {})
                                md["abstain_due_to_mask_total"] = int(md.get("abstain_due_to_mask_total", 0) or 0) + 1
                                controller.metrics_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
                            except Exception:
                                pass
                            # Abstain: do not terminate; clear any sticky mask flag
                            # Normalize masked_action to selective abstain with configured penalty
                            try:
                                import os as _os
                                _pen_sel = float((_os.getenv("BRAIN_MASK_PENALTY_SELECTIVE") or "-0.02").strip())
                            except Exception:
                                _pen_sel = -0.02
                            result = {
                                "observation": result.get("observation", ""),
                                "state": result.get("state"),
                                "reward": _pen_sel,
                                "done": False,
                                "success": False,
                                "message": "abstain_masked",
                            }
                            try:
                                if hasattr(controller._env, "_mask_next"):
                                    delattr(controller._env, "_mask_next")  # type: ignore[attr-defined]
                            except Exception:
                                pass
                        # Record memory if we overcame a mask and made progress
                        try:
                            if world_model_signature and _progress(_prev_state, result.get("state", {})):
                                fk = _serialize_feat_key(_feature_key_from_state(_prev_state))
                                recent_success_alts[(fk, str(world_model_signature))] = world_model_signature
                                # If a workaround plan was pending, persist to memory
                                try:
                                    ctx = getattr(controller, "_pending_workaround_context", None)
                                    plan = getattr(controller, "_pending_workaround_plan", None)
                                    if isinstance(ctx, str) and isinstance(plan, list) and plan:
                                        _wmemory[ctx] = list(plan)
                                        _save_workaround_memory(_wmemory)
                                        # Metrics: memory writes
                                        raw2 = controller._read_metrics_dict()
                                        if not isinstance(raw2, dict): raw2 = {}
                                        md2 = raw2.setdefault("masking", {})
                                        md2["workaround_memory_writes"] = int(md2.get("workaround_memory_writes", 0) or 0) + 1
                                        controller.metrics_path.write_text(json.dumps(raw2, ensure_ascii=False, indent=2), encoding="utf-8")
                                except Exception:
                                    pass
                                try:
                                    if hasattr(controller, "_pending_workaround_context"):
                                        delattr(controller, "_pending_workaround_context")
                                    if hasattr(controller, "_pending_workaround_plan"):
                                        delattr(controller, "_pending_workaround_plan")
                                except Exception:
                                    pass
                        except Exception:
                            pass
                        action_name = candidate_action
                        target_val = candidate_target
                        item_val = candidate_item
                        source = "world_model"
                        used_world_model = True
                    except Exception:
                        result = None
                        world_model_disabled = True
                        fallback_triggered = True
                        if fallback_reason is None:
                            fallback_reason = "world_model_error"
                        fallback_label = "world_model_error"

        if world_model_disabled and source == "world_model":
            # If the world model was marked disabled during this step, undo the recorded action.
            source = "heuristic"
            action_name = None
            result = None

        if action_name is None:
            action_name, target_val, item_val = _heuristic_decision(state, visited, controller._env)
            # Enforce mask at boundary for heuristic as well
            allowed = _allowed_action_sigs_from_env()
            # Emit mask_debug even when no mask provider, to make signals visible
            try:
                import os as _os
                if (_os.getenv("BRAIN_MASK_DEBUG") or "").strip().lower() in {"1","true","yes","on"}:
                    controller._append_event({
                        "event": "mask_debug",
                        "mode": controller._mode,
                        "phase": "heuristic_boundary",
                        "feature_key": _serialize_feat_key(_feature_key_from_state(state)),
                        "desired_sig": _action_signature(_action_key(action_name, target_val, item_val)),
                        "allowed": list(allowed) if allowed is not None else ["*"],
                    })
            except Exception:
                pass
            if allowed is not None:
                # Masking diagnostics for heuristic path
                try:
                    raw = controller._read_metrics_dict()
                    if not isinstance(raw, dict):
                        raw = {}
                    # Snapshot relevant env flags for debugging
                    try:
                        import os as _os
                        envsnap = {
                            "BRAIN_MASK_FORCE_ALTERNATIVE": (_os.getenv("BRAIN_MASK_FORCE_ALTERNATIVE") or None),
                            "BRAIN_MASK_DEBUG": (_os.getenv("BRAIN_MASK_DEBUG") or None),
                            "ADV_MASK_MODE": (_os.getenv("ADV_MASK_MODE") or None),
                            "ADV_MASK_PROB": (_os.getenv("ADV_MASK_PROB") or None),
                        }
                    except Exception:
                        envsnap = {}
                    md = raw.setdefault("masking", {})
                    if envsnap:
                        md["env_flags_snapshot"] = envsnap
                    md["allowed_probed_steps"] = int(md.get("allowed_probed_steps", 0) or 0) + 1
                    if len(allowed) == 0:
                        md["allowed_mask_all_steps"] = int(md.get("allowed_mask_all_steps", 0) or 0) + 1
                    else:
                        md["allowed_nonempty_steps"] = int(md.get("allowed_nonempty_steps", 0) or 0) + 1
                        if "*" not in allowed:
                            md["selective_blocks_applied"] = int(md.get("selective_blocks_applied", 0) or 0) + 1
                            _chokes = {"unlock|vault|", "open|vault|", "move|vault|"}
                            if any((c not in allowed) for c in _chokes):
                                md["choke_actions_masked_steps"] = int(md.get("choke_actions_masked_steps", 0) or 0) + 1
                    controller.metrics_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
                except Exception:
                    pass
                # Optional per-step debug trace for heuristic enforcement
                try:
                    import os as _os
                    if (_os.getenv("BRAIN_MASK_DEBUG") or "").strip().lower() in {"1","true","yes","on"}:
                        controller._append_event({
                            "event": "mask_debug",
                            "mode": controller._mode,
                            "phase": "heuristic_boundary",
                            "feature_key": _serialize_feat_key(_feature_key_from_state(state)),
                            "desired_sig": _action_signature(_action_key(action_name, target_val, item_val)),
                            "allowed": list(allowed),
                        })
                except Exception:
                    pass
                hsig = _action_signature(_action_key(action_name, target_val, item_val))
                force_alt = False
                try:
                    import os as _os
                    force_alt = ((_os.getenv("BRAIN_MASK_FORCE_ALTERNATIVE") or "").strip().lower() in {"1","true","yes","on"})
                except Exception:
                    force_alt = False
                if (hsig not in allowed) or force_alt:
                    # Constraint-aware planning: persistent memory and heuristic plan
                    try:
                        ctx = _constraint_context(state, allowed)
                    except Exception:
                        ctx = None
                    memory_hit = False
                    planned: list[str] = []
                    masked_list = []
                    try:
                        masked_list = sorted([c for c in _CHOKES if c not in (allowed or set())])
                    except Exception:
                        masked_list = []
                    picked_alt: tuple[str, str | None, str | None] | None = None
                    if ctx:
                        # Prefer fresh constraint plan first in the same hall-with-key masked context
                        try:
                            _room_now = str(state.get("room", ""))
                            _has_key_now = ("key" in set(state.get("inventory") or []))
                            prefer_plan_first = (_room_now == "hall" and _has_key_now and bool(masked_list))
                        except Exception:
                            prefer_plan_first = False
                        if prefer_plan_first:
                            seq2 = _constraint_plan(state, masked_list)
                            for s in seq2:
                                if s in set(recent_alternative_sigs[-3:]):
                                    continue
                                try:
                                    _prev_room = recent_rooms[-2] if len(recent_rooms) >= 2 else None
                                    if (_prev_room == "storage" and s == "move|storage|"):
                                        continue
                                except Exception:
                                    pass
                                if (s in allowed) and (not force_alt or s != hsig):
                                    ak = _parse_signature(s)
                                    picked_alt = (ak[0], ak[1], ak[2])
                                    planned = list(seq2)
                                    break
                        if picked_alt is None:
                            seq = _wmemory.get(ctx)
                            if isinstance(seq, list) and seq:
                                for s in seq:
                                    if s in set(recent_alternative_sigs[-3:]):
                                        continue
                                    try:
                                        _prev_room = recent_rooms[-2] if len(recent_rooms) >= 2 else None
                                        if (_room_now == "hall" and _has_key_now and _prev_room == "storage" and s == "move|storage|"):
                                            continue
                                    except Exception:
                                        pass
                                    if (s in allowed) and (not force_alt or s != hsig):
                                        ak = _parse_signature(s)
                                        picked_alt = (ak[0], ak[1], ak[2])
                                        planned = list(seq)
                                        memory_hit = True
                                        break
                        if picked_alt is None and not prefer_plan_first:
                            seq2 = _constraint_plan(state, masked_list)
                            for s in seq2:
                                if s in set(recent_alternative_sigs[-3:]):
                                    continue
                                try:
                                    _prev_room = recent_rooms[-2] if len(recent_rooms) >= 2 else None
                                    if (_prev_room == "storage" and s == "move|storage|"):
                                        continue
                                except Exception:
                                    pass
                                if (s in allowed) and (not force_alt or s != hsig):
                                    ak = _parse_signature(s)
                                    picked_alt = (ak[0], ak[1], ak[2])
                                    planned = list(seq2)
                                    break
                    # Priority: intelligent alternatives before generic HTN-lite
                    if picked_alt is None:
                        try:
                            prio = _intelligent_alternatives(state, allowed, hsig, masked_list)
                        except Exception:
                            prio = []
                        if prio:
                            cand = _pick_from_priority(prio, allowed, hsig, force_alt=bool(force_alt))
                            if cand:
                                ak = _parse_signature(cand)
                                picked_alt = (ak[0], ak[1], ak[2])
                    if picked_alt is not None and ctx:
                        try:
                            controller._pending_workaround_context = ctx
                            controller._pending_workaround_plan = planned
                        except Exception:
                            pass
                        try:
                            import os as _os
                            if (_os.getenv("BRAIN_MASK_DEBUG") or "").strip().lower() in {"1","true","yes","on"}:
                                controller._append_event({
                                    "event": "mask_debug",
                                    "mode": controller._mode,
                                    "phase": "workaround_plan",
                                    "feature_key": _serialize_feat_key(_feature_key_from_state(state)),
                                    "context": ctx,
                                    "plan": planned,
                                    "memory_hit": bool(memory_hit),
                                })
                        except Exception:
                            pass
                    # HTN-lite: propose alternatives based on subgoal
                    sub = _current_subgoal(state, controller._env)
                    if sub == "acquire_key":
                        alt_order: list[tuple[str, str | None, str | None]] = [("move", "storage", None), ("move", "hall", None), ("search", None, None)]
                    elif sub == "reach_hall":
                        alt_order = [("move", "hall", None), ("move", "start", None), ("search", None, None)]
                    elif sub == "unlock":
                        alt_order = [("move", "storage", None), ("move", "hall", None), ("search", None, None)]
                    elif sub == "open":
                        alt_order = [("move", "hall", None), ("move", "start", None), ("search", None, None)]
                    else:  # enter_vault
                        alt_order = [("move", "hall", None), ("move", "start", None), ("search", None, None)]
                    # Extend with all visible non-vault exits
                    try:
                        exits = list(state.get("exits") or [])
                    except Exception:
                        exits = []
                    for e in exits:
                        if e and e != "vault":
                            cand = ("move", str(e), None)
                            if cand not in alt_order:
                                alt_order.append(cand)
                    # If not already picked from plan/memory, continue with previous strategy
                    if picked_alt is None:
                        picked_alt = None
                    # If forcing alternatives and desired is allowed, directly pick a distinct allowed signature first
                    if force_alt:
                        try:
                            for s in list(allowed):
                                if s != hsig:
                                    ak = _parse_signature(s)
                                    picked_alt = (ak[0], ak[1], ak[2])
                                    break
                        except Exception:
                            picked_alt = None
                    # Memory-first: reuse last successful alternative if available
                    try:
                        fk = _serialize_feat_key(_feature_key_from_state(state))
                        mem = recent_success_alts.get((fk, str(hsig)))
                        if mem and mem in allowed and picked_alt is None:
                            ak = _parse_signature(mem)
                            picked_alt = (ak[0], ak[1], ak[2])
                    except Exception:
                        pass
                    for an, at, it in alt_order:
                        sig = _action_signature(_action_key(an, at, it))
                        if picked_alt is None and sig in allowed:
                            picked_alt = (an, at, it)
                            break
                    # If forced alternative and the first allowed equals desired, try next
                    if force_alt and picked_alt is not None:
                        if _action_signature(_action_key(*picked_alt)) == hsig:
                            picked_alt = None
                            for an, at, it in alt_order:
                                sig = _action_signature(_action_key(an, at, it))
                                if sig in allowed and sig != hsig:
                                    picked_alt = (an, at, it)
                                    break
                    if picked_alt is None:
                        # Progress-biased top-K selection from allowed
                        try:
                            ranked = sorted([s for s in list(allowed) if s != hsig], key=lambda s: _progress_rank(s, state))
                        except Exception:
                            ranked = []
                        if ranked:
                            import random as _rnd
                            topk = ranked[:3]
                            s = _rnd.choice(topk) if len(topk) > 1 else topk[0]
                            ak = _parse_signature(s)
                            picked_alt = (ak[0], ak[1], ak[2])
                        else:
                            # Try substitution policy
                            desired = hsig
                            fk = _serialize_feat_key(_feature_key_from_state(state))
                            psig = _substitute_sig(desired, fk, allowed)
                            if psig:
                                ak = _parse_signature(psig)
                                picked_alt = (ak[0], ak[1], ak[2])
                    if picked_alt is not None:
                        _prev_state = dict(state)
                        action_name, target_val, item_val = picked_alt
                        # Mark that we're executing an alternative for the heuristic desired hsig
                        try:
                            controller._action_was_alternative = True
                            controller._desired_signature_for_alt = str(hsig)
                        except Exception:
                            pass
                        # Record chosen alt for diversity memory
                        try:
                            sig_now = _action_signature(_action_key(action_name, target_val, item_val))
                            if sig_now:
                                recent_alternative_sigs.append(str(sig_now))
                        except Exception:
                            pass
                        was_alt_step = True
                        desired_before_alt = str(hsig)
                        # Optional real-time pacing before execution
                        if _sleep_ms:
                            time.sleep(_sleep_ms / 1000.0)
                        result = controller.act(action_name, target=target_val, item=item_val)
                        # Metric: workaround executed on heuristic path
                        try:
                            raw = controller._read_metrics_dict()
                            if not isinstance(raw, dict):
                                raw = {}
                            md = raw.setdefault("masking", {})
                            md["workarounds_success_total"] = int(md.get("workarounds_success_total", 0) or 0) + 1
                            controller.metrics_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
                        except Exception:
                            pass
                        # Record memory if progress was made
                        try:
                            if hsig and _progress(_prev_state, result.get("state", {})):
                                fk = _serialize_feat_key(_feature_key_from_state(_prev_state))
                                sig_now = _action_signature(_action_key(action_name, target_val, item_val))
                                if sig_now:
                                    recent_success_alts[(fk, str(hsig))] = sig_now
                                # Persist workaround plan on progress
                                try:
                                    ctx2 = getattr(controller, "_pending_workaround_context", None)
                                    plan2 = getattr(controller, "_pending_workaround_plan", None)
                                    if isinstance(ctx2, str) and isinstance(plan2, list) and plan2:
                                        _wmemory[ctx2] = list(plan2)
                                        _save_workaround_memory(_wmemory)
                                        raw2 = controller._read_metrics_dict()
                                        if not isinstance(raw2, dict): raw2 = {}
                                        md2 = raw2.setdefault("masking", {})
                                        md2["workaround_memory_writes"] = int(md2.get("workaround_memory_writes", 0) or 0) + 1
                                        controller.metrics_path.write_text(json.dumps(raw2, ensure_ascii=False, indent=2), encoding="utf-8")
                                except Exception:
                                    pass
                                try:
                                    if hasattr(controller, "_pending_workaround_context"):
                                        delattr(controller, "_pending_workaround_context")
                                    if hasattr(controller, "_pending_workaround_plan"):
                                        delattr(controller, "_pending_workaround_plan")
                                except Exception:
                                    pass
                        except Exception:
                            pass
                        # Post-act safeguard: normalize any masked result to abstain
                        if isinstance(result, dict) and str(result.get("message", "")) == "masked_action":
                            result = {
                                "observation": result.get("observation", ""),
                                "state": result.get("state"),
                                "reward": -0.05,
                                "done": False,
                                "success": False,
                                "message": "abstain_masked",
                            }
                            try:
                                if hasattr(controller._env, "_mask_next"):
                                    delattr(controller._env, "_mask_next")  # type: ignore[attr-defined]
                            except Exception:
                                pass
                    else:
                        # Abstain due to mask when no alternative is allowed
                        try:
                            ob = controller.observe()
                        except Exception:
                            ob = {"state": state, "observation": ""}
                        # Selective vs mask-all message based on allow-list
                        _msg2 = "abstain_masked"
                        try:
                            if len(allowed) == 0:
                                _msg2 = "abstain_masked_all"
                        except Exception:
                            _msg2 = "abstain_masked"
                        try:
                            import os as _os
                            _pen_sel = float((_os.getenv("BRAIN_MASK_PENALTY_SELECTIVE") or "-0.02").strip())
                            _pen_all = float((_os.getenv("BRAIN_MASK_PENALTY_ALL") or "-0.08").strip())
                        except Exception:
                            _pen_sel, _pen_all = -0.02, -0.08
                        result = {
                            "observation": ob.get("observation", ""),
                            "state": ob.get("state", state),
                            "reward": (_pen_all if _msg2 == "abstain_masked_all" else _pen_sel),
                            "done": False,
                            "success": False,
                            "message": _msg2,
                        }
                        # Metrics: count planning-time abstentions due to mask (heuristic path)
                        try:
                            raw = controller._read_metrics_dict()
                            if not isinstance(raw, dict):
                                raw = {}
                            md = raw.setdefault("masking", {})
                            md["abstain_due_to_mask_total"] = int(md.get("abstain_due_to_mask_total", 0) or 0) + 1
                            md["planning_abstains_total"] = int(md.get("planning_abstains_total", 0) or 0) + 1
                            controller.metrics_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
                        except Exception:
                            pass
                        # Also record a synthetic act event for harvesting with the desired (masked) signature
                        try:
                            des_act: str | None = None
                            des_t: str | None = None
                            des_i: str | None = None
                            hsig = _action_signature(_action_key(action_name, target_val, item_val)) if action_name else None
                            if hsig:
                                ak = _parse_signature(hsig)
                                des_act, des_t, des_i = ak[0], ak[1], ak[2]
                            controller._append_event({
                                "event": "act",
                                "mode": controller._mode,
                                "action": des_act,
                                "target": des_t,
                                "item": des_i,
                                "state": result.get("state", state),
                                "reward": result.get("reward", -0.01),
                                "done": result.get("done", False),
                                "success": result.get("success", False),
                                "message": result.get("message", _msg2),
                            })
                        except Exception:
                            pass
                else:
                    # No mask provider; optionally force an alternative for visibility
                    try:
                        import os as _os
                        force_alt = ((_os.getenv("BRAIN_MASK_FORCE_ALTERNATIVE") or "").strip().lower() in {"1","true","yes","on"})
                    except Exception:
                        force_alt = False
                    if force_alt and action_name is not None:
                        # Propose a simple alternative distinct from desired
                        hsig = _action_signature(_action_key(action_name, target_val, item_val))
                        picked_alt: tuple[str, str | None, str | None] | None = None
                        # Use WM v2 ranking with progress bias, exclude desired
                        if _use_wm_v2:
                            ranked = [s for s in _v2_rank_action_sigs(state) if s != hsig]
                            ranked.sort(key=lambda s: _progress_rank(s, state))
                            if ranked:
                                import random as _rnd
                                s = _rnd.choice(ranked[:3]) if len(ranked) > 1 else ranked[0]
                                ak = _parse_signature(s)
                                picked_alt = (ak[0], ak[1], ak[2])
                        if picked_alt is None:
                            fk = _serialize_feat_key(_feature_key_from_state(state))
                            psig = _substitute_sig(hsig, fk, None)
                            if psig:
                                ak = _parse_signature(psig)
                                picked_alt = (ak[0], ak[1], ak[2])
                        if picked_alt is not None:
                            # Mark alternative and execute
                            try:
                                controller._action_was_alternative = True
                                controller._desired_signature_for_alt = str(hsig)
                            except Exception:
                                pass
                            was_alt_step = True
                            desired_before_alt = str(hsig)
                            if _sleep_ms:
                                time.sleep(_sleep_ms / 1000.0)
                            action_name, target_val, item_val = picked_alt
                            result = controller.act(action_name, target=target_val, item=item_val)
                            # Metric: workaround executed
                            try:
                                raw = controller._read_metrics_dict()
                                if not isinstance(raw, dict):
                                    raw = {}
                                md = raw.setdefault("masking", {})
                                md["workarounds_success_total"] = int(md.get("workarounds_success_total", 0) or 0) + 1
                                controller.metrics_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
                            except Exception:
                                pass
                        else:
                            # Fall back to desired action when no alternative found
                            if _sleep_ms:
                                time.sleep(_sleep_ms / 1000.0)
                            result = controller.act(action_name, target=target_val, item=item_val)
                    else:
                        # Optional real-time pacing before execution
                        if _sleep_ms:
                            time.sleep(_sleep_ms / 1000.0)
                        result = controller.act(action_name, target=target_val, item=item_val)
            else:
                # Optional real-time pacing before execution
                if _sleep_ms:
                    time.sleep(_sleep_ms / 1000.0)
                result = controller.act(action_name, target=target_val, item=item_val)
            source = "heuristic"
            if world_model_disabled and not world_model_forced_off and fallback_label is None:
                fallback_label = "world_model_disabled"
            if world_model_forced_off:
                fallback_label = None
            # Reset loop detection since heuristic took over.
            last_signature = None
            consecutive_signature_count = 0
        else:
            # Successful world-model step; keep loop detection state.
            pass

        if action_name == "move" and target_val:
            visited.add(target_val)

        _trace_obj = {
            "step": step_idx,
            "source": source,
            "action": action_name,
            "target": target_val,
            "item": item_val,
            "room": state.get("room"),
            "has_key": "key" in set(state.get("inventory") or []),
            "door_unlocked": bool((state.get("door") or {}).get("unlocked")),
            "door_open": bool((state.get("door") or {}).get("open")),
            "key_visible": ("key" in list(state.get("visible_items") or [])) if isinstance(state.get("visible_items"), (list, tuple, set)) else False,
            "exits_count": len(list(state.get("exits") or [])) if isinstance(state.get("exits"), (list, tuple, set)) else 0,
            "goal_room": getattr(controller._env, "goal_room", None),
            "feature_key": _serialize_feat_key(_feature_key_from_state(state)),
            "fallback": fallback_label,
            "step_message": (result.get("message") if isinstance(result, dict) else None),
        }
        # Include whether an alternative was executed this step and what the desired signature was before substitution
        try:
            _trace_obj["was_alternative"] = bool(was_alt_step)
            _trace_obj["desired_before_alt"] = desired_before_alt
        except Exception:
            pass
        # Optional WM debug: include candidate/desired signatures and planner flags when enabled
        try:
            import os as _os
            if (_os.getenv("BRAIN_WM_DEBUG") or "").strip().lower() in {"1","true","yes","on"}:
                _trace_obj.update({
                    "wm_v2_sig": v2_sig,
                    "wm_desired_sig": world_model_signature,
                    "wm_used": (source == "world_model"),
                    "wm_v2_on": bool(_use_wm_v2),
                    "wm_planner_on": bool(_planner_on),
                    "wm_forced": bool(_force_wm),
                })
                # Mask debug: include allowed snapshot and whether desired was allowed and if force_alt applied
                try:
                    if allowed is not None:
                        _trace_obj.update({
                            "mask_allowed_count": int(len(allowed)),
                            "mask_desired_allowed": bool(_desired_allowed_snapshot) if _desired_allowed_snapshot is not None else None,
                            "mask_force_alt": bool(force_alt),
                        })
                except Exception:
                    pass
        except Exception:
            pass
        policy_trace.append(_trace_obj)

        # Clear alternative flags so they don't leak into the next step
        try:
            if hasattr(controller, "_action_was_alternative"):
                delattr(controller, "_action_was_alternative")
            if hasattr(controller, "_desired_signature_for_alt"):
                delattr(controller, "_desired_signature_for_alt")
        except Exception:
            pass

        state = result["state"]
        # Track recent rooms for anti-oscillation heuristics
        try:
            r_now = str(state.get("room", ""))
            if not recent_rooms or recent_rooms[-1] != r_now:
                recent_rooms.append(r_now)
        except Exception:
            pass
        if result.get("done"):
            _metrics_counted = True
            break

    if result is None:
        result = {
            "observation": obs["observation"],
            "state": state,
            "reward": 0.0,
            "done": True,
            "success": False,
            "message": "no_actions",
        }

    # Final trace normalization: convert any lingering masked_action messages to abstain_masked
    try:
        for st in policy_trace:
            if isinstance(st, dict) and st.get("step_message") == "masked_action":
                st["step_message"] = "abstain_masked"
    except Exception:
        pass

    result_out = dict(result)
    result_out["policy_trace"] = policy_trace
    result_out["used_world_model"] = used_world_model
    result_out["planner_source"] = "world_model" if used_world_model else "heuristic"
    result_out["wm_planner_used"] = used_wm_planner
    world_model_steps = sum(1 for step in policy_trace if step.get("source") == "world_model")
    heuristic_steps = sum(1 for step in policy_trace if step.get("source") == "heuristic")
    controller.record_planner_usage(
        mode=mode,
        planner_source=result_out["planner_source"],
        used_world_model=used_world_model,
        total_steps=len(policy_trace),
        world_model_steps=world_model_steps,
        heuristic_steps=heuristic_steps,
        fallback_triggered=fallback_triggered and not world_model_forced_off,
        fallback_reason=fallback_reason if not world_model_forced_off else None,
    )
    # Ensure episodes counter in metrics.json is at least the reconstructed total.
    # If the environment did not mark done (or metrics wasn't updated via act), update here.
    if not _metrics_counted:
        try:
            controller._update_metrics(success=bool(result_out.get("success")), reward=float(result_out.get("reward", 0.0)))
        except Exception:
            pass
    # Best-effort sync: bump metrics['episodes'] to match reconstructed episodes across ALL workspaces (>= 3 steps)
    try:
        raw = controller._read_metrics_dict()
        if not isinstance(raw, dict):
            raw = {}
        # Reconstruct globally from episodes.jsonl
        total_ge3 = 0
        steps_in_current = 0
        have_episode = False
        if controller.episodes_path.exists():
            for ln in controller.episodes_path.read_text(encoding="utf-8").splitlines():
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    obj = json.loads(ln)
                except Exception:
                    continue
                evt = obj.get("event")
                if evt == "reset":
                    if have_episode and steps_in_current >= 3:
                        total_ge3 += 1
                    steps_in_current = 0
                    have_episode = True
                    continue
                if evt == "act":
                    if not have_episode:
                        have_episode = True
                    steps_in_current += 1
                    if bool(obj.get("done")):
                        if steps_in_current >= 3:
                            total_ge3 += 1
                        steps_in_current = 0
                        have_episode = False
        # Flush final
        if have_episode and steps_in_current >= 3:
            total_ge3 += 1
        cur = int(raw.get("episodes", 0) or 0)
        if total_ge3 > cur:
            raw["episodes"] = int(total_ge3)
            controller.metrics_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass
    return result_out
