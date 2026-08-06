#!/usr/bin/env python3
"""Session-level shared-value support for the YAML figure pipeline.

This module is intentionally a **support layer**, not a runtime owner of
transforms, layout, or rendering.

Lifecycle contract
------------------
``SharedContent`` / ``DataContext`` provide:

1. **Lazy registry** — ``register(name, compute_fn, release_fn=None)`` binds a
   name to a factory.  Dataset sources and named ``share_data`` values use this.
2. **Cached store** — ``get(name)`` returns a cached value or computes once via
   the registry and stores the result.  Missing names return ``None``.
3. **Usage plan** — ``set_usage_plan({name: count})`` records how many times a
   tracked source is expected to be consumed across figures/layers.  Planning
   lives in ``core_runtime.prepare_usage_plan``; this module only stores counts.
4. **Consume** — ``consume(name)`` decrements a tracked count.  When the count
   reaches zero the **cached value** is invalidated (and ``release_fn`` runs if
   present).  The **registry entry remains**, so a later accidental ``get`` can
   recompute rather than hard-fail.  Untracked names are no-ops for ``consume``.
5. **Invalidate** — drops the cached value and calls ``release_fn`` when a
   value was present.  Does not remove registry factories.

Boundary rules
--------------
- Do not put transform execution, column planning, or render dispatch here.
- ``share_data`` persistence and cache identity stay in ``preprocessor`` /
  ``layer_runtime``; this module only holds the in-session values and counters.
- Figures should talk to ``DataContext``, not reach into ``SharedContent``
  internals.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional
import threading


class SharedContent:
    """Session-level shared storage (lazy compute, usage counts, invalidation).

    Support API only — see module docstring for the lifecycle contract.
    """

    def __init__(self, seed: Optional[int] = None, logger: Any = None):
        self._logger = logger
        self._seed = seed
        self._store: Dict[str, Any] = {}
        self._registry: Dict[str, Callable[[SharedContent], Any]] = {}
        self._release_registry: Dict[str, Callable[[], None]] = {}
        self._remaining_uses: Dict[str, int] = {}
        self._lock = threading.RLock()

    def register(
        self,
        name: str,
        compute_fn: Callable[[SharedContent], Any],
        release_fn: Optional[Callable[[], None]] = None,
    ) -> None:
        """Register a lazy factory.  Does not compute immediately."""
        with self._lock:
            self._registry[name] = compute_fn
            if release_fn is not None:
                self._release_registry[name] = release_fn
            if self._logger:
                self._logger.debug(f"SharedContent: register -> {name}")

    def get(self, name: str) -> Any:
        """Return cached value, or compute via registry once, or ``None``."""
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
        """Write or overwrite a cached value without requiring a registry entry."""
        with self._lock:
            self._store[name] = value
            if release_fn is not None:
                self._release_registry[name] = release_fn
            if self._logger:
                self._logger.debug(f"SharedContent: update -> {name}")

    def has_cached(self, name: str) -> bool:
        """Return True when a value is currently resident in the store."""
        with self._lock:
            return str(name) in self._store

    def is_registered(self, name: str) -> bool:
        """Return True when a lazy factory is registered for ``name``."""
        with self._lock:
            return str(name) in self._registry

    def is_tracked(self, name: str) -> bool:
        """Return True when ``name`` appears in the usage plan."""
        with self._lock:
            return str(name) in self._remaining_uses

    def invalidate(self, name: Optional[str] = None) -> None:
        """Drop cached value(s) and run release hooks for values that were present.

        Registry factories are kept so a later ``get`` can recompute.  Usage-plan
        counters are not modified here.
        """
        with self._lock:
            if name is None:
                for key in list(self._store.keys()):
                    self._release_cached_locked(key)
                self._store.clear()
                if self._logger:
                    self._logger.debug("SharedContent: invalidate ALL")
            else:
                key = str(name)
                if key in self._store:
                    self._release_cached_locked(key)
                    self._store.pop(key, None)
                if self._logger:
                    self._logger.debug(f"SharedContent: invalidate -> {key}")

    def _release_cached_locked(self, key: str) -> None:
        release_fn = self._release_registry.get(key)
        if release_fn is None:
            return
        try:
            release_fn()
        except Exception:
            pass

    def set_usage_plan(self, counts: Mapping[str, int]) -> None:
        """Replace the session usage plan.  Counts must be non-negative integers."""
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
        """Return remaining planned uses, or 0 when untracked."""
        with self._lock:
            return int(self._remaining_uses.get(str(name), 0))

    def consume(self, name: str, amount: int = 1) -> int:
        """Decrement a tracked usage count; invalidate cache at zero remaining.

        Untracked names return 0 and do not touch the store.  When remaining hits
        zero, only the **cached value** is released — the registry factory stays.
        """
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
            should_release = remain == 0 and key in self._store
        if should_release:
            self.invalidate(key)
        return remain

    def stats(self) -> Dict[str, int]:
        """Diagnostic counters for logs and tests."""
        with self._lock:
            return {
                "cached": len(self._store),
                "registered": len(self._registry),
                "tracked": len(self._remaining_uses),
            }


class DataContext:
    """Figure-facing facade over :class:`SharedContent`.

    Isolates figure/layer code from the session store implementation.  Does not
    own planning, transforms, or rendering.
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

    def has_cached(self, name: str) -> bool:
        return self._shared.has_cached(name)

    def is_registered(self, name: str) -> bool:
        return self._shared.is_registered(name)

    def is_tracked(self, name: str) -> bool:
        return self._shared.is_tracked(name)

    def set_usage_plan(self, counts: Mapping[str, int]) -> None:
        self._shared.set_usage_plan(counts)

    def remaining_uses(self, name: str) -> int:
        return self._shared.remaining_uses(name)

    def consume(self, name: str, amount: int = 1) -> int:
        return self._shared.consume(name, amount=amount)

    def stats(self) -> Dict[str, int]:
        return self._shared.stats()
