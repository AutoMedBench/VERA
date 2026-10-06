"""UUID-only runtime identity factories."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from uuid import NAMESPACE_URL, uuid4, uuid5


class RandomUUIDFactory:
    """Production identity source; purposes are labels, never ID prefixes."""

    def new(self, purpose: str) -> str:
        if not purpose:
            raise ValueError("identity purpose is required")
        return str(uuid4())


@dataclass
class DeterministicUUIDFactory:
    """Repeatable UUID source for fixtures and replay tests."""

    seed: str
    _counter: int = 0
    _lock: Lock = field(default_factory=Lock, repr=False)

    def new(self, purpose: str) -> str:
        if not purpose:
            raise ValueError("identity purpose is required")
        with self._lock:
            ordinal = self._counter
            self._counter += 1
        namespace = uuid5(NAMESPACE_URL, self.seed)
        return str(uuid5(namespace, f"{ordinal}:{purpose}"))


__all__ = ["DeterministicUUIDFactory", "RandomUUIDFactory"]
