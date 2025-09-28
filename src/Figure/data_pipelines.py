#!/usr/bin/env python3 

#!/usr/bin/env python3
from __future__ import annotations
from typing import Any, Callable, Dict, Optional
import threading

class SharedContent:
    """
    会话级可变共享存储，供所有 Figure 共享使用（懒计算 + 可更新）。
    - register(name, compute_fn): 注册懒计算函数
    - get(name): 命中则返缓存；否则调用 compute_fn 计算并缓存
    - update(name, value): 显式写入/覆盖
    - invalidate(name=None): 失效一个或全部
    - stats(): 诊断信息
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
    注入到每个 Figure 的门面，隔离 Figure 与核心实现。
    Figure 用它来 get/update/register/invalidate 共享内容。
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