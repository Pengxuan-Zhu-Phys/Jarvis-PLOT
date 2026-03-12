#!/usr/bin/env python3 

#!/usr/bin/env python3
from __future__ import annotations
from typing import Any, Callable, Dict, Optional
import threading

class SharedContent:
    """
    Session-level variable shared storage for all Figures (supports lazy evaluation and updates).
    - register(name, compute_fn): Register a lazy evaluation function.
    - get(name): Return cached value if available; otherwise, compute using compute_fn and cache the result.
    - update(name, value): Explicitly write or overwrite a value.
    - invalidate(name=None): Invalidate a specific entry or all entries.
    - stats(): Diagnostic information.
    """
    def __init__(self, seed: Optional[int] = None, logger: Any = None):
        self._logger = logger
        self._seed = seed
        self._store: Dict[str, Any] = {}
        self._registry: Dict[str, Callable[[SharedContent], Any]] = {}
        self._release_registry: Dict[str, Callable[[], None]] = {}
        self._remaining_uses: Dict[str, int] = {}
        self._lock = threading.RLock()

    # ---- 懒计算接口 ----
    def register(
        self,
        name: str,
        compute_fn: Callable[[SharedContent], Any],
        release_fn: Optional[Callable[[], None]] = None,
    ) -> None:
        with self._lock:
            self._registry[name] = compute_fn
            if release_fn is not None:
                self._release_registry[name] = release_fn
            if self._logger:
                self._logger.debug(f"SharedContent: register -> {name}")

    def get(self, name: str) -> Any:
        with self._lock:
            if name in self._store:
                return self._store[name]
            if name in self._registry:
                if self._logger:
                    self._logger.debug(f"SharedContent: MISS -> {name}; computing...")
                val = self._registry[name](self)
                self._store[name] = val
                return val
            if self._logger:
                self._logger.debug(f"SharedContent: MISS (no registry) -> {name}; returning None")
            return None

    def update(self, name: str, value: Any, release_fn: Optional[Callable[[], None]] = None) -> None:
        with self._lock:
            self._store[name] = value
            if release_fn is not None:
                self._release_registry[name] = release_fn
            if self._logger:
                self._logger.debug(f"SharedContent: update -> {name}")

    def invalidate(self, name: Optional[str] = None) -> None:
        with self._lock:
            if name is None:
                for key, release_fn in list(self._release_registry.items()):
                    try:
                        release_fn()
                    except Exception:
                        pass
                self._store.clear()
                if self._logger:
                    self._logger.debug("SharedContent: invalidate ALL")
            else:
                self._store.pop(name, None)
                release_fn = self._release_registry.get(name)
                if release_fn is not None:
                    try:
                        release_fn()
                    except Exception:
                        pass
                if self._logger:
                    self._logger.debug(f"SharedContent: invalidate -> {name}")

    def set_usage_plan(self, counts: Dict[str, int]) -> None:
        with self._lock:
            self._remaining_uses = {}
            for name, count in counts.items():
                try:
                    ival = int(count)
                except Exception:
                    continue
                if ival >= 0:
                    self._remaining_uses[str(name)] = ival
            if self._logger:
                self._logger.debug(
                    "SharedContent: usage plan loaded -> {}".format(
                        ", ".join(f"{k}:{v}" for k, v in sorted(self._remaining_uses.items()))
                    )
                )

    def remaining_uses(self, name: str) -> int:
        with self._lock:
            return int(self._remaining_uses.get(str(name), 0))

    def consume(self, name: str, amount: int = 1) -> int:
        key = str(name)
        with self._lock:
            if key not in self._remaining_uses:
                return 0
            try:
                dec = max(int(amount), 0)
            except Exception:
                dec = 0
            remain = max(int(self._remaining_uses.get(key, 0)) - dec, 0)
            self._remaining_uses[key] = remain
            if self._logger:
                self._logger.debug(f"SharedContent: consume -> {key}, remain={remain}")
        if remain == 0:
            self.invalidate(key)
        return remain

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {
                "cached": len(self._store),
                "registered": len(self._registry),
                "tracked": len(self._remaining_uses),
            }

class DataContext:
    """
    Inject it into the facade of each Figure to isolate the Figure from the core implementation.
    Figures use it to get, update, register, and invalidate shared content.
    """
    def __init__(self, shared: SharedContent):
        self._shared = shared

    def get(self, name: str) -> Any:
        return self._shared.get(name)

    def update(self, name: str, value: Any, release_fn: Optional[Callable[[], None]] = None) -> None:
        self._shared.update(name, value, release_fn=release_fn)

    def register(
        self,
        name: str,
        compute_fn: Callable[[SharedContent], Any],
        release_fn: Optional[Callable[[], None]] = None,
    ) -> None:
        self._shared.register(name, compute_fn, release_fn=release_fn)

    def invalidate(self, name: Optional[str] = None) -> None:
        self._shared.invalidate(name)

    def set_usage_plan(self, counts: Dict[str, int]) -> None:
        self._shared.set_usage_plan(counts)

    def remaining_uses(self, name: str) -> int:
        return self._shared.remaining_uses(name)

    def consume(self, name: str, amount: int = 1) -> int:
        return self._shared.consume(name, amount=amount)

    def stats(self) -> Dict[str, int]:
        return self._shared.stats()
