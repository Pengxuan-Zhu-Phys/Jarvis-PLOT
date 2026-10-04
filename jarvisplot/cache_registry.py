"""Global registry and maintenance of workdir-local plot caches (stdlib only)."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Iterator


REGISTRY_SCHEMA = "jarvisplot.cache-registry/v1"
CACHE_SCHEMA = "jarvisplot.cache/v1"
MARKER = ".jarvisplot.json"
PIPELINE_COMPONENTS = ("data", "named", "summary", "materialized")
COMPONENTS = frozenset((*PIPELINE_COMPONENTS, "agent_twins"))


class CacheUsageError(ValueError):
    """A maintenance command selected an invalid target or interval."""


def registry_path() -> Path:
    home = Path(os.environ.get("JARVIS_HOME") or Path.home() / ".jarvis")
    return home.expanduser().resolve() / "plot.json"


def _read_json(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return data


def _write_json(path: Path, data: dict) -> None:
    """Replace a complete JSON file; readers never see a partial write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


class FileLock:
    """Process-held locks; closing or crashing automatically releases them."""

    def __init__(self, path: Path, *, shared: bool = False, blocking: bool = True):
        self.handle = None
        path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = path.open("a+b")
        try:
            if os.name == "nt":
                self._lock_windows(shared, blocking)
            else:
                import fcntl

                operation = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
                fcntl.flock(self.handle.fileno(), operation | (0 if blocking else fcntl.LOCK_NB))
        except BaseException:
            self.handle.close()
            raise

    def _lock_windows(self, shared: bool, blocking: bool) -> None:
        import ctypes
        from ctypes import wintypes
        import msvcrt

        class Overlapped(ctypes.Structure):
            _fields_ = [("internal", ctypes.c_size_t), ("internal_high", ctypes.c_size_t),
                        ("offset", wintypes.DWORD), ("offset_high", wintypes.DWORD), ("event", wintypes.HANDLE)]

        self._overlapped = Overlapped()
        self._kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self._kernel.LockFileEx.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD,
                                          wintypes.DWORD, wintypes.DWORD, ctypes.POINTER(Overlapped)]
        self._kernel.UnlockFileEx.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD,
                                            wintypes.DWORD, ctypes.POINTER(Overlapped)]
        self._native_handle = msvcrt.get_osfhandle(self.handle.fileno())
        flags = (0 if shared else 2) | (0 if blocking else 1)
        if not self._kernel.LockFileEx(self._native_handle, flags, 0, 1, 0, ctypes.byref(self._overlapped)):
            code = ctypes.get_last_error()
            if code == 33:
                raise BlockingIOError("Cache is in use")
            raise ctypes.WinError(code)

    def close(self) -> None:
        if self.handle is None or self.handle.closed:
            return
        try:
            if os.name == "nt":
                import ctypes

                self._kernel.UnlockFileEx(self._native_handle, 0, 1, 0, ctypes.byref(self._overlapped))
        finally:
            self.handle.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        self.close()


def _date(text: str) -> datetime:
    value = datetime.fromisoformat(text)
    if value.tzinfo is None:
        raise ValueError("Cache timestamps must include a timezone")
    return value


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _marker(root: Path) -> dict:
    if not root.is_absolute() or root.name != ".cache" or root.is_symlink() or root.parent != root.parent.resolve():
        raise ValueError(f"Unsafe cache location: {root}")
    if (root / MARKER).is_symlink():
        raise ValueError(f"Unsafe cache marker: {root / MARKER}")
    data = _read_json(root / MARKER)
    components = data.get("components")
    configs = data.get("configs", [])
    if (data.get("schema") != CACHE_SCHEMA or data.get("workdir") != str(root.parent)
            or not isinstance(components, list) or not components
            or any(not isinstance(name, str) or name not in COMPONENTS for name in components)
            or not isinstance(configs, list) or any(not isinstance(path, str) for path in configs)
            or not isinstance(data.get("last_used"), str)):
        raise ValueError(f"Invalid Jarvis-PLOT cache marker: {root / MARKER}")
    _date(data["last_used"])
    return data


def _stats(root: Path, components: list[str]) -> tuple[int, float]:
    size, newest = 0, 0.0
    for name in [*components, "manifest.json", MARKER]:
        path = root / name
        if path.is_symlink():
            continue
        if path.is_dir():
            paths = (Path(directory) / filename for directory, _, files in os.walk(path, followlinks=False) for filename in files)
        else:
            paths = [path]
        for file in paths:
            if file.is_symlink():
                continue
            try:
                stat = file.stat()
            except FileNotFoundError:
                continue
            size += stat.st_size
            newest = max(newest, stat.st_mtime)
    return size, newest


def _entry_root(key: str, entry: dict) -> Path:
    root = Path(key)
    if not root.is_absolute() or entry["cache_dir"] != key or root != Path(entry["workdir"]) / ".cache":
        raise ValueError("Registry cache path does not match its workdir")
    return root


class CacheRegistry:
    def __init__(self, path: Path | None = None):
        self.path = path or registry_path()

    def _load(self) -> dict:
        if not self.path.exists():
            return {"schema": REGISTRY_SCHEMA, "caches": {}}
        data = _read_json(self.path)
        if data.get("schema") != REGISTRY_SCHEMA or not isinstance(data.get("caches"), dict):
            raise ValueError(f"Invalid plot cache registry: {self.path}")
        return data

    @contextmanager
    def _locked(self) -> Iterator[dict]:
        with FileLock(self.path.with_suffix(".lock")):
            yield self._load()

    def lease(self, workdir: str | Path, *, shared: bool = True, blocking: bool = True) -> FileLock:
        root = str(Path(workdir).expanduser().resolve() / ".cache")
        digest = hashlib.sha256(root.encode()).hexdigest()
        return FileLock(self.path.parent / "plot-locks" / f"{digest}.lock", shared=shared, blocking=blocking)

    def register(self, workdir: str | Path, *, components=PIPELINE_COMPONENTS,
                 config_path: str | None = None, last_used: str | None = None) -> dict:
        """Call while holding a lease, before using or creating payloads."""
        root = Path(workdir).expanduser().resolve() / ".cache"
        if root.is_symlink() or (root / MARKER).is_symlink():
            raise ValueError(f"Cannot register a symlink cache or marker: {root}")
        if not components or any(name not in COMPONENTS for name in components):
            raise ValueError("Unknown plot cache components")
        if last_used is not None:
            _date(last_used)
        with self._locked() as registry:
            previous = _marker(root) if (root / MARKER).exists() else {}
            configs = set(previous.get("configs", []))
            if config_path:
                configs.add(str(Path(config_path).expanduser().resolve()))
            marker = {"schema": CACHE_SCHEMA, "workdir": str(root.parent),
                      "components": sorted(set(previous.get("components", [])) | set(components)),
                      "configs": sorted(configs), "last_used": last_used or _now()}
            _write_json(root / MARKER, marker)
            entry = {"workdir": str(root.parent), "cache_dir": str(root),
                     "configs": marker["configs"], "last_used": marker["last_used"]}
            registry["caches"][str(root)] = entry
            _write_json(self.path, registry)
        return entry

    def list(self) -> list[dict]:
        with self._locked() as registry:
            entries = list(registry["caches"].items())
        rows = []
        for key, entry in sorted(entries):
            row = {"cache_dir": key, "bytes": 0, "status": "invalid"}
            try:
                root = _entry_root(key, entry)
                row.update(entry)
                if not root.exists() and not root.is_symlink():
                    row["status"] = "missing"
                else:
                    marker = _marker(root)
                    row.update(last_used=marker["last_used"], configs=marker.get("configs", []), status="ready")
                    try:
                        with self.lease(root.parent, shared=False, blocking=False):
                            pass
                    except BlockingIOError:
                        row["status"] = "active"
                    row["bytes"], _ = _stats(root, marker["components"])
            except (OSError, ValueError, KeyError, TypeError) as exc:
                row.update(status="invalid", error=str(exc))
            rows.append(row)
        return rows

    def scan(self, directories: list[str]) -> dict:
        result = {"registered": [], "skipped": [], "errors": []}
        seen = set()
        for directory in directories:
            start = Path(directory).expanduser().resolve()
            if not start.is_dir():
                result["errors"].append({"path": str(start), "error": "Directory does not exist"})
                continue
            candidates = [start] if start.name == ".cache" else self._scan_roots(start)
            for root in candidates:
                if root in seen or root.is_symlink():
                    continue
                seen.add(root)
                try:
                    with self.lease(root.parent, shared=False, blocking=False):
                        if (root / MARKER).exists():
                            marker = _marker(root)
                            components, last_used = marker["components"], marker["last_used"]
                        elif (root / "manifest.json").is_file() and all(
                            (root / name).is_dir() and not (root / name).is_symlink() for name in PIPELINE_COMPONENTS
                        ):
                            # Legacy ProjectCache layout plus its manifest is
                            # required before adopting an unmarked .cache.
                            manifest = _read_json(root / "manifest.json")
                            if manifest.get("schema") != 1 or not isinstance(manifest.get("files"), dict) or not isinstance(manifest.get("named"), dict):
                                continue
                            components = list(PIPELINE_COMPONENTS)
                            _, newest = _stats(root, components)
                            last_used = datetime.fromtimestamp(newest or root.stat().st_mtime, timezone.utc).isoformat()
                        else:
                            continue
                        result["registered"].append(self.register(root.parent, components=components, last_used=last_used))
                except BlockingIOError:
                    result["skipped"].append({"path": str(root), "reason": "active"})
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    result["errors"].append({"path": str(root), "error": str(exc)})
        return result

    @staticmethod
    def _scan_roots(start: Path) -> Iterator[Path]:
        for directory, dirs, _ in os.walk(start, followlinks=False):
            if ".cache" in dirs:
                yield Path(directory) / ".cache"
            dirs[:] = [name for name in dirs if name not in {".cache", ".git", ".venv", "node_modules", "__pycache__"}]

    def clean(self, *, workdirs: list[str] | None = None, older_than: float | None = None, dry_run: bool = False) -> dict:
        """Clean registered caches in each selected directory tree, including its root."""
        if older_than is not None and (not math.isfinite(older_than) or older_than < 0):
            raise CacheUsageError("Days must be a finite nonnegative number")
        directories = {Path(path).expanduser().resolve() for path in workdirs} if workdirs is not None else None
        result = {"removed": [], "missing": [], "skipped": [], "errors": [], "bytes": 0, "dry_run": dry_run}
        with self._locked() as registry:
            entries = registry["caches"]
            selected = None
            if directories is not None:
                selected = set()
                unmatched = []
                for directory in directories:
                    matches = {key for key in entries
                               if Path(key).parent == directory or directory in Path(key).parent.parents}
                    selected.update(matches)
                    if not matches:
                        unmatched.append(str(directory))
                if unmatched:
                    raise CacheUsageError("No registered plot caches under: " + ", ".join(sorted(unmatched)) +
                                          "; run cache scan first")
            for key, entry in list(entries.items()):
                if selected is not None and key not in selected:
                    continue
                try:
                    root = _entry_root(key, entry)
                    with self.lease(root.parent, shared=False, blocking=False):
                        if not root.exists() and not root.is_symlink():
                            result["missing"].append(key)
                            if not dry_run:
                                del entries[key]
                            continue
                        marker = _marker(root)
                        if older_than is not None and (datetime.now(timezone.utc) - _date(marker["last_used"])).total_seconds() < older_than * 86400:
                            continue
                        size, _ = _stats(root, marker["components"])
                        if not dry_run:
                            for name in [*marker["components"], "manifest.json"]:
                                path = root / name
                                if path.is_dir() and not path.is_symlink():
                                    shutil.rmtree(path)
                                else:
                                    path.unlink(missing_ok=True)
                            (root / MARKER).unlink()
                            if not any(root.iterdir()):
                                root.rmdir()
                            del entries[key]
                        result["removed"].append(key)
                        result["bytes"] += size
                except BlockingIOError:
                    result["skipped"].append({"path": key, "reason": "active"})
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    result["errors"].append({"path": key, "error": str(exc)})
            if not dry_run:
                _write_json(self.path, registry)
        return result
