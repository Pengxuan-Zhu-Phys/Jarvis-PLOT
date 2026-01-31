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
        self._lock = threading.RLock()

    # ---- 懒计算接口 ----
    def register(self, name: str, compute_fn: Callable[[SharedContent], Any]) -> None:
        with self._lock:
            self._registry[name] = compute_fn
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

    def update(self, name: str, value: Any) -> None:
        with self._lock:
            self._store[name] = value
            if self._logger:
                self._logger.debug(f"SharedContent: update -> {name}")

    def invalidate(self, name: Optional[str] = None) -> None:
        with self._lock:
            if name is None:
                self._store.clear()
                if self._logger:
                    self._logger.debug("SharedContent: invalidate ALL")
            else:
                self._store.pop(name, None)
                if self._logger:
                    self._logger.debug(f"SharedContent: invalidate -> {name}")

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {"cached": len(self._store), "registered": len(self._registry)}

class DataContext:
    """
    Inject it into the facade of each Figure to isolate the Figure from the core implementation.
    Figures use it to get, update, register, and invalidate shared content.
    """
    def __init__(self, shared: SharedContent):
        self._shared = shared

    def get(self, name: str) -> Any:
        return self._shared.get(name)

    def update(self, name: str, value: Any) -> None:
        self._shared.update(name, value)

    def register(self, name: str, compute_fn: Callable[[SharedContent], Any]) -> None:
        self._shared.register(name, compute_fn)

    def invalidate(self, name: Optional[str] = None) -> None:
        self._shared.invalidate(name)

    def stats(self) -> Dict[str, int]:
        return self._shared.stats()