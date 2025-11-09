from __future__ import annotations

from dataclasses import dataclass
import random

# Base map
_ADJACENCY: dict[str, list[str]] = {
    "start": ["hall"],
    "hall": ["start", "storage", "vault"],
    "storage": ["hall"],
    "vault": ["hall"],
}

# Rooms+ variant: two extra rooms connected to the hub (hall)
_ADJACENCY_PLUS: dict[str, list[str]] = {
    "start": ["hall"],
    "hall": ["start", "storage", "vault", "office", "lab"],
    "storage": ["hall"],
    "vault": ["hall"],
    "office": ["hall"],
    "lab": ["hall"],
}


@dataclass
class DoorView:
    locked: bool
    unlocked: bool
    open: bool

    def as_dict(self) -> dict[str, bool]:
        return {"locked": self.locked, "unlocked": self.unlocked, "open": self.open}


class KeysDoorsEnv:
    """Simple keys-and-doors curriculum used for embodied loop tests."""

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random()
        self._seed_value = 0
        self.mode = "curriculum"
        self.key_room = "storage"
        self.goal_room = "vault"
        self.room = "start"
        self.inventory: set[str] = set()
        self.door_unlocked = False
        self.door_open = False
        self.done = False
        self.steps = 0
        self.hidden_key_mode = False
        self.key_revealed = True
        if seed is not None:
            self._seed_value = int(seed)
        self.reset(mode=self.mode, seed=self._seed_value)

    def reset(self, mode: str = "curriculum", seed: int | None = None) -> dict[str, dict]:
        if seed is not None:
            self._seed_value = int(seed)
        self.mode = mode or "curriculum"
        self._configure_key_room()
        self.hidden_key_mode = self.mode == "generalization_plus"
        self.key_revealed = not self.hidden_key_mode
        self.room = "start"
        self.inventory.clear()
        self.door_unlocked = False
        self.door_open = False
        self.done = False
        self.steps = 0
        return self.observe()

    def observe(self) -> dict[str, dict]:
        visible_items: list[str] = []
        if (
            not self.done
            and self.room == self.key_room
            and "key" not in self.inventory
            and (self.key_revealed or not self.hidden_key_mode)
        ):
            visible_items.append("key")
        door = DoorView(locked=not self.door_unlocked, unlocked=self.door_unlocked, open=self.door_open)
        state = {
            "room": self.room,
            "inventory": sorted(self.inventory),
            "visible_items": visible_items,
            "exits": list(_ADJACENCY.get(self.room, [])),
            "door": door.as_dict(),
            "mode": self.mode,
            "steps": self.steps,
            "done": self.done,
        }
        parts: list[str] = [f"You are in the {self.room}."]
        if visible_items:
            parts.append(f"You see {', '.join(visible_items)} here.")
        if self.room == "hall":
            if self.door_open:
                parts.append("The vault door is open.")
            elif self.door_unlocked:
                parts.append("The vault door is unlocked but closed.")
            else:
                parts.append("The vault door is locked.")
        observation = " ".join(parts)
        return {"observation": observation, "state": state}

    def step(self, action: str, *, target: str | None = None, item: str | None = None) -> dict[str, object]:
        if self.done:
            obs = self.observe()
            return {
                "observation": obs["observation"],
                "state": obs["state"],
                "reward": 0.0,
                "done": True,
                "success": True,
                "message": "episode_already_complete",
            }
        act = (action or "").strip().lower()
        reward = 0.0
        message = ""
        if act == "move":
            reward -= 0.01
            message = self._move(target)
        elif act == "pickup":
            message = self._pickup(item)
        elif act == "unlock":
            message = self._unlock(target)
        elif act == "open":
            message = self._open(target)
        elif act == "search":
            reward -= 0.005
            message = self._search()
        elif act == "look":
            message = "look"
        else:
            raise ValueError("unknown_action")
        self.steps += 1
        if self.room == self.goal_room:
            self.done = True
            self.door_open = True
            reward += 1.0
            message = "success"
        obs = self.observe()
        return {
            "observation": obs["observation"],
            "state": obs["state"],
            "reward": reward,
            "done": self.done,
            "success": self.done and self.room == self.goal_room,
            "message": message,
        }
    def serialize(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "seed": self._seed_value,
            "room": self.room,
            "inventory": sorted(self.inventory),
            "key_room": self.key_room,
            "door_unlocked": self.door_unlocked,
            "door_open": self.door_open,
            "hidden_key_mode": self.hidden_key_mode,
            "key_revealed": self.key_revealed,
            "done": self.done,
            "steps": self.steps,
        }
    def load_state(self, data: dict[str, object]) -> None:
        self.mode = str(data.get("mode", "curriculum"))
        self._seed_value = int(data.get("seed", 0))
        self.key_room = str(data.get("key_room", "storage"))
        self.room = str(data.get("room", "start"))
        self.inventory = set(str(x) for x in data.get("inventory", []))
        self.door_unlocked = bool(data.get("door_unlocked", False))
        self.door_open = bool(data.get("door_open", False))
        self.done = bool(data.get("done", False))
        self.steps = int(data.get("steps", 0))
        self.hidden_key_mode = bool(data.get("hidden_key_mode", False))
        self.key_revealed = bool(data.get("key_revealed", True))

    def _configure_key_room(self) -> None:
        if self.mode == "curriculum":
            self.key_room = "storage"
            return
        if self.mode == "generalization_plus":
            self.key_room = "storage"
            return
        choices = ["hall", "storage"]
        if self.mode == "generalization":
            choices = ["hall", "storage", "start"]
        idx = 0
        if choices:
            self._rng.seed(self._seed_value)
            idx = self._seed_value % len(choices)
        self.key_room = choices[idx]

    def _move(self, target: str | None) -> str:
        dest = (target or "").strip().lower()
        if dest not in _ADJACENCY.get(self.room, []):
            raise ValueError("invalid_move")
        if dest == "vault" and not self.door_open:
            raise ValueError("door_closed")
        self.room = dest
        return f"move_{dest}"

    def _pickup(self, item: str | None) -> str:
        itm = (item or "").strip().lower()
        if itm != "key":
            raise ValueError("unknown_item")
        if self.room != self.key_room or "key" in self.inventory:
            raise ValueError("key_not_available")
        if self.hidden_key_mode and not self.key_revealed:
            raise ValueError("key_not_available")
        self.inventory.add("key")
        return "pickup_key"

    def _unlock(self, target: str | None) -> str:
        dest = (target or "").strip().lower()
        if dest != "vault":
            raise ValueError("unknown_unlock_target")
        if self.room != "hall" or "key" not in self.inventory:
            raise ValueError("cannot_unlock")
        self.door_unlocked = True
        return "unlock_vault"

    def _open(self, target: str | None) -> str:
        dest = (target or "").strip().lower()
        if dest != "vault":
            raise ValueError("unknown_open_target")
        if self.room != "hall" or not self.door_unlocked:
            raise ValueError("cannot_open")
        self.door_open = True
        return "open_vault"

    def _search(self) -> str:
        if not self.hidden_key_mode:
            raise ValueError("nothing_hidden")
        if self.room != self.key_room:
            raise ValueError("nothing_to_search")
        if self.key_revealed:
            return "search_nothing"
        self.key_revealed = True
        return "search_reveal_key"


def scripted_policy(env: KeysDoorsEnv, max_steps: int = 16) -> dict[str, object]:
    """Run a deterministic policy that solves the curriculum across seeds."""
    obs = env.observe()
    state = obs["state"]
    visited: set[str] = set()
    for _ in range(max_steps):
        room = state["room"]
        inventory = set(state["inventory"])
        door = state["door"]
        if "key" in inventory:
            if room != "hall":
                result = env.step("move", target="hall")
            elif not door["unlocked"]:
                result = env.step("unlock", target="vault")
            elif not door["open"]:
                result = env.step("open", target="vault")
            else:
                result = env.step("move", target="vault")
        else:
            if getattr(env, "hidden_key_mode", False) and not getattr(env, "key_revealed", True):
                if room == env.key_room:
                    result = env.step("search")
                    state = result["state"]
                    if result["done"]:
                        return result
                    continue
            if "key" in state["visible_items"]:
                result = env.step("pickup", item="key")
            else:
                target = "hall"
                for candidate in state["exits"]:
                    if candidate == "vault":
                        continue
                    if candidate not in visited and candidate != room:
                        target = candidate
                        break
                visited.add(target)
                result = env.step("move", target=target)
        state = result["state"]
        if result["done"]:
            return result
    return result


class RoomsPlusEnv(KeysDoorsEnv):
    """Environment variant with extra rooms (office, lab) connected to the hub.

    Door semantics remain for the vault only. Inherits all behavior from KeysDoorsEnv
    but overrides adjacency usage and key placement choices.
    """

    def observe(self) -> dict[str, dict]:
        visible_items: list[str] = []
        if (
            not self.done
            and self.room == self.key_room
            and "key" not in self.inventory
            and (self.key_revealed or not self.hidden_key_mode)
        ):
            visible_items.append("key")
        door = DoorView(locked=not self.door_unlocked, unlocked=self.door_unlocked, open=self.door_open)
        state = {
            "room": self.room,
            "inventory": sorted(self.inventory),
            "visible_items": visible_items,
            "exits": list(_ADJACENCY_PLUS.get(self.room, [])),
            "door": door.as_dict(),
            "mode": self.mode,
            "steps": self.steps,
            "done": self.done,
        }
        parts: list[str] = [f"You are in the {self.room}."]
        if visible_items:
            parts.append(f"You see {', '.join(visible_items)} here.")
        if self.room == "hall":
            if self.door_open:
                parts.append("The vault door is open.")
            elif self.door_unlocked:
                parts.append("The vault door is unlocked but closed.")
            else:
                parts.append("The vault door is locked.")
        observation = " ".join(parts)
        return {"observation": observation, "state": state}

    def _move(self, target: str | None) -> str:
        dest = (target or "").strip().lower()
        if dest not in _ADJACENCY_PLUS.get(self.room, []):
            raise ValueError("invalid_move")
        if dest == "vault" and not self.door_open:
            raise ValueError("door_closed")
        self.room = dest
        return f"move_{dest}"

    def _configure_key_room(self) -> None:
        # Keep curriculum simple
        if self.mode == "curriculum":
            self.key_room = "storage"
            return
        if self.mode == "generalization_plus":
            # Default to storage; other modes can randomize
            self.key_room = "storage"
            return
        # Generalization can place key in any non-vault room including new ones
        choices = ["hall", "storage", "start", "office", "lab"]
        idx = 0
        if choices:
            self._rng.seed(self._seed_value)
            idx = self._seed_value % len(choices)
        self.key_room = choices[idx]
