#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
import hashlib
import json
import os
import re
import time
import weakref
import tempfile
import threading
from functools import wraps
from contextlib import contextmanager

import pandas as pd
from .memtrace import memtrace_checkpoint, memtrace_file_checkpoint, memtrace_object_inventory
from .cache_registry import CacheRegistry, CacheUsageError, TEMP_PREFIX, _write_json


def _serialized(method):
    @wraps(method)
    def call(self, *args, **kwargs):
        with self._writing():
            return method(self, *args, **kwargs)
    return call


class ProjectCache:
    """Workdir-local cache store: <workdir>/.cache."""

    def __init__(self, workdir: str, logger=None, rebuild: bool = False, *, config_path: str | None = None):
        self.logger = logger
        self.workdir = Path(workdir).expanduser().resolve()
        self.root = self.workdir / ".cache"
        self.data_dir = self.root / "data"
        self.summary_dir = self.root / "summary"
        self.named_dir = self.root / "named"
        self.materialized_dir = self.root / "materialized"
        self.manifest_path = self.root / "manifest.json"
        self.rebuild = bool(rebuild)
        self._lease_finalizer = None
        self._closed = False
        self._thread_lock = threading.RLock()
        self._write_depth = 0
        self.registry = CacheRegistry()
        # Invalid policy is an input error; maintenance I/O errors are warnings.
        self.registry.policy()
        self._maintain()
        try:
            lease = self.registry.lease(self.workdir, shared=not self.rebuild,
                                        blocking=False)
        except BlockingIOError as exc:
            action = "rebuild active" if self.rebuild else "open exclusively held"
            raise CacheUsageError(f"Cannot {action} plot cache: {self.root}; retry after other tasks finish") from exc
        except OSError as exc:
            raise CacheUsageError(f"Cannot lock plot cache {self.root}: {exc}; restore access to {self.registry.path.parent}") from exc
        self._lease_finalizer = weakref.finalize(self, lease.close)
        try:
            if self.root.is_symlink():
                raise CacheUsageError(f"Cannot use a symlink plot cache: {self.root}")
            self.root.mkdir(parents=True, exist_ok=True)
            if self.rebuild:
                # The exclusive lease is retained throughout this session, with
                # no unlock/relock window for another reader to see a rebuild.
                self.registry.register(self.workdir, config_path=config_path)
                from .cache_registry import _marker
                self.registry._remove_owned(self.root, _marker(self.root))
            for directory in (self.root, self.data_dir, self.summary_dir,
                              self.named_dir, self.materialized_dir):
                directory.mkdir(parents=True, exist_ok=True)
            self.manifest = self._load_json(
                self.manifest_path, default={"schema": 1, "files": {}, "named": {}})
            self._written_dataframe_keys: set[str] = set()
            self._written_materialized_keys: set[str] = set()
            self._written_named_keys: set[str] = set()
            self._written_summary_keys: set[str] = set()
            self._used_paths: set[Path] = set()
            self._used_named_names: set[str] = set()
            self._materialized_builds: dict[str, Path] = {}
            try:
                self.registry.register(self.workdir, config_path=config_path)
            except (OSError, ValueError) as exc:
                self._warn(f"Failed registering plot cache: {exc}")
        except BaseException:
            self._lease_finalizer()
            raise

    def _maintain(self) -> dict:
        try:
            result = self.registry.maintain()
            for error in result["errors"]:
                self._warn(f"Cache maintenance: {error['path']}: {error['error']}")
            if result["removed"]:
                self._info(f"Cache maintenance removed {len(result['removed'])} idle cache(s), {result['bytes']} bytes")
            return result
        except (OSError, ValueError) as exc:
            self._warn(f"Cache maintenance failed: {exc}")
            return {"cleanup": {}}

    def close(self) -> dict:
        if self._closed:
            return {"cleanup": {}}
        self._closed = True
        try:
            self.registry.register(self.workdir)
        except (OSError, ValueError) as exc:
            self._warn(f"Failed updating cache last use: {exc}")
        finally:
            if self._lease_finalizer is not None:
                self._lease_finalizer()
        return self._maintain()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def _track_path(self, path: Path) -> Path:
        if self._closed:
            raise RuntimeError("Plot cache session is closed")
        # Named paths come from the manifest; never clean outside this cache.
        resolved = path.resolve()
        if resolved != self.root and resolved.is_relative_to(self.root):
            self._used_paths.add(path)
        return path

    def clear_used(self) -> dict[str, int]:
        """Finish this session; defer final-render cleanup while readers remain."""
        if self._closed:
            return {"removed": 0, "failed": 0, "deferred": 0}
        try:
            self.registry.queue_cleanup(self.workdir, self._used_paths)
        except (OSError, ValueError) as exc:
            self._warn(f"Failed scheduling cache cleanup: {exc}")
            return {"removed": 0, "failed": len(self._used_paths), "deferred": 0}
        result = self.close()
        return result.get("cleanup", {}).get(str(self.root),
                    {"removed": 0, "failed": 0, "deferred": len(self._used_paths)})

    @contextmanager
    def _writing(self):
        with self._thread_lock:
            if self._closed:
                raise RuntimeError("Plot cache session is closed")
            if self._write_depth:
                yield
                return
            with self.registry.writer(self.workdir):
                self._write_depth += 1
                try:
                    yield
                finally:
                    self._write_depth -= 1

    @contextmanager
    def _atomic_path(self, path: Path):
        """Write a sibling temporary file and publish only a complete payload."""
        if self._closed:
            raise RuntimeError("Plot cache session is closed")
        fd, temporary = tempfile.mkstemp(prefix=TEMP_PREFIX, dir=path.parent)
        os.close(fd)
        temporary = Path(temporary)
        try:
            yield temporary
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def _debug(self, msg: str) -> None:
        if self.logger:
            try:
                self.logger.debug(msg)
            except Exception:
                pass

    def _warn(self, msg: str) -> None:
        if self.logger:
            try:
                self.logger.warning(msg)
            except Exception:
                pass

    def _info(self, msg: str) -> None:
        if self.logger:
            try:
                self.logger.info(msg)
            except Exception:
                pass

    @staticmethod
    def _load_json(path: Path, default: Dict[str, Any]) -> Dict[str, Any]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
        except Exception:
            pass
        return dict(default)

    def _set_manifest_item(self, section: str, key: str, value: dict) -> None:
        # Merge only the changed entry with the current on-disk manifest.
        with self._writing():
            manifest = self._load_json(self.manifest_path,
                                      default={"schema": 1, "files": {}, "named": {}})
            manifest.setdefault(section, {})[key] = value
            _write_json(self.manifest_path, manifest)
            self.manifest = manifest

    @staticmethod
    def _canonical_json(payload: Any) -> str:
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)

    @classmethod
    def cache_key(cls, payload: Any) -> str:
        raw = cls._canonical_json(payload).encode("utf-8")
        return hashlib.sha1(raw).hexdigest()

    @staticmethod
    def _md5_file(path: str, chunk_size: int = 8 * 1024 * 1024) -> str:
        md5 = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                md5.update(chunk)
        return md5.hexdigest()

    @staticmethod
    def _sha1_file(path: str, chunk_size: int = 8 * 1024 * 1024) -> str:
        sha1 = hashlib.sha1()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                sha1.update(chunk)
        return sha1.hexdigest()

    @_serialized
    def source_fingerprint(self, path: str, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        abs_path = str(Path(path).expanduser().resolve())
        try:
            st = os.stat(abs_path)
        except Exception as e:
            self._warn(f"Cannot stat source file '{abs_path}': {e}")
            fp = {"path": abs_path, "size": None, "mtime_ns": None, "md5": None}
            if extra:
                fp.update(extra)
            return fp

        files = self.manifest.setdefault("files", {})
        old = files.get(abs_path, {})

        size = int(st.st_size)
        mtime_ns = int(st.st_mtime_ns)
        md5 = old.get("md5")

        if not (old.get("size") == size and old.get("mtime_ns") == mtime_ns and md5):
            t0 = time.perf_counter()
            md5 = self._md5_file(abs_path)
            dt = time.perf_counter() - t0
            self._debug(f"Computed md5 for '{abs_path}' in {dt:.2f}s")

        files[abs_path] = {"size": size, "mtime_ns": mtime_ns, "md5": md5}
        self._set_manifest_item("files", abs_path, files[abs_path])

        fp = {"path": abs_path, "size": size, "mtime_ns": mtime_ns, "md5": md5}
        if extra:
            fp.update(extra)
        return fp

    @_serialized
    def get_dataframe(self, key: str):
        if self.rebuild and str(key) not in self._written_dataframe_keys:
            return None
        p = self.dataframe_cache_path(key)
        if not p.exists():
            return None
        try:
            return pd.read_pickle(p)
        except Exception as e:
            self._warn(f"Failed loading dataframe cache '{p}': {e}")
            return None

    def dataframe_cache_path(self, key: str) -> Path:
        self._track_path(self.data_dir / f"{key}.json")
        return self._track_path(self.data_dir / f"{key}.pkl")

    @_serialized
    def is_dataframe_meta_consistent(self, key: str, meta: Optional[Dict[str, Any]]) -> bool:
        if self.rebuild and str(key) not in self._written_dataframe_keys:
            return False
        if not isinstance(meta, dict):
            return False
        p = self.dataframe_cache_path(key)
        if not p.exists():
            return False
        try:
            st = p.stat()
        except Exception:
            return False

        expect_size = meta.get("cache_file_size")
        if expect_size is not None:
            try:
                if int(expect_size) != int(st.st_size):
                    return False
            except Exception:
                return False

        expect_sha1 = meta.get("cache_file_sha1")
        if expect_sha1:
            try:
                real_sha1 = self._sha1_file(str(p))
            except Exception:
                return False
            if str(real_sha1) != str(expect_sha1):
                return False
        return True

    @_serialized
    def get_dataframe_meta(self, key: str) -> Optional[Dict[str, Any]]:
        if self.rebuild and str(key) not in self._written_dataframe_keys:
            return None
        p = self._track_path(self.data_dir / f"{key}.json")
        if not p.exists():
            return None
        try:
            with open(p, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, dict):
                return obj
            return None
        except Exception as e:
            self._warn(f"Failed loading dataframe cache meta '{p}': {e}")
            return None

    @_serialized
    def put_dataframe(self, key: str, df, meta: Optional[Dict[str, Any]] = None) -> None:
        p = self.dataframe_cache_path(key)
        try:
            memtrace_checkpoint(
                self.logger,
                "cache.put_dataframe.before",
                df,
                extra={"cache_key": str(key)[:16], "cache_file": str(p)},
            )
            memtrace_object_inventory(
                self.logger,
                "cache.put_dataframe.inventory",
                {"df": df},
                roles={"df": "cache-ready dataframe"},
                min_bytes=64 * 1024 * 1024,
            )
            with self._atomic_path(p) as temporary:
                df.to_pickle(temporary)
            payload: Dict[str, Any] = {}
            if isinstance(meta, dict):
                payload.update(meta)
            payload.setdefault("cache_schema", "jp-cache-v2")
            payload["cache_key"] = str(key)
            payload["cache_file"] = str(p.resolve())
            try:
                st = p.stat()
                payload["cache_file_size"] = int(st.st_size)
                payload["cache_file_mtime_ns"] = int(st.st_mtime_ns)
            except Exception:
                pass
            try:
                payload["cache_file_sha1"] = self._sha1_file(str(p))
            except Exception:
                pass

            m = self.data_dir / f"{key}.json"
            _write_json(m, payload)
            self._written_dataframe_keys.add(str(key))
            memtrace_file_checkpoint(
                self.logger,
                "cache.put_dataframe.after",
                p,
                extra={"cache_key": str(key)[:16]},
            )
        except Exception as e:
            self._warn(f"Failed writing dataframe cache '{p}': {e}")

    def _summary_path(self, source_fp: Dict[str, Any]) -> Path:
        key = self.cache_key({"kind": "summary", "source": source_fp})
        return self._track_path(self.summary_dir / f"{key}.txt")

    def materialized_slot(self, key: str) -> Path:
        return self._track_path(self.materialized_dir / str(key))

    def get_materialized_manifest(self, key: str) -> Optional[Dict[str, Any]]:
        if self.rebuild and str(key) not in self._written_materialized_keys:
            return None
        path = self.materialized_slot(key) / "manifest.json"
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else None
        except Exception as e:
            self._warn(f"Failed loading materialized manifest '{path}': {e}")
            return None

    def put_materialized_manifest(self, key: str, manifest: Dict[str, Any]) -> None:
        slot = self.materialized_slot(key)
        slot.mkdir(parents=True, exist_ok=True)
        build = self._materialized_builds.pop(str(key), None)
        if build is not None:
            generation = slot / ("gen-" + build.name.removeprefix(TEMP_PREFIX))
            os.replace(build, generation)
            manifest["part_files"] = [str(Path(generation.name) / part)
                                      for part in manifest.get("part_files", [])]
        # Every published generation is immutable until all reader leases end.
        _write_json(slot / "manifest.json", manifest)
        self._written_materialized_keys.add(str(key))

    def clear_materialized_slot(self, key: str) -> Path:
        # Build privately; never remove an existing lazy reader's parquet parts.
        slot = self.materialized_slot(key)
        slot.mkdir(parents=True, exist_ok=True)
        build = Path(tempfile.mkdtemp(prefix=TEMP_PREFIX, dir=slot))
        self._materialized_builds[str(key)] = build
        return build

    def get_summary(self, source_fp: Dict[str, Any]) -> Optional[str]:
        key = self.cache_key({"kind": "summary", "source": source_fp})
        if self.rebuild and str(key) not in self._written_summary_keys:
            return None
        p = self._summary_path(source_fp)
        if not p.exists():
            return None
        try:
            return p.read_text(encoding="utf-8")
        except Exception:
            return None

    def put_summary(self, source_fp: Dict[str, Any], text: str) -> None:
        p = self._summary_path(source_fp)
        try:
            with self._atomic_path(p) as temporary:
                temporary.write_text(str(text), encoding="utf-8")
            key = self.cache_key({"kind": "summary", "source": source_fp})
            self._written_summary_keys.add(str(key))
        except Exception as e:
            self._warn(f"Failed writing summary cache '{p}': {e}")

    @staticmethod
    def _safe_name(name: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name))

    @_serialized
    def put_named(self, name: str, signature: str, df) -> None:
        safe = self._safe_name(name)
        slot = self.cache_key({"name": name, "signature": signature})[:12]
        p = self.named_dir / f"{safe}__{slot}.pkl"
        self._track_path(p)
        self._used_named_names.add(str(name))
        try:
            memtrace_checkpoint(
                self.logger,
                "cache.put_named.before",
                df,
                extra={"name": str(name), "signature": str(signature)[:16]},
            )
            with self._atomic_path(p) as temporary:
                df.to_pickle(temporary)
            named = self.manifest.setdefault("named", {})
            sha1 = None
            size = None
            mtime_ns = None
            try:
                st = p.stat()
                size = int(st.st_size)
                mtime_ns = int(st.st_mtime_ns)
            except Exception:
                pass
            try:
                sha1 = self._sha1_file(str(p))
            except Exception:
                sha1 = None
            named[str(name)] = {
                "kind": "dataframe",
                "signature": str(signature),
                "path": str(p.relative_to(self.root)),
                "sha1": sha1,
                "size": size,
                "mtime_ns": mtime_ns,
            }
            self._written_named_keys.add(str(name))
            self._set_manifest_item("named", str(name), named[str(name)])
            memtrace_file_checkpoint(
                self.logger,
                "cache.put_named.after",
                p,
                extra={"name": str(name), "signature": str(signature)[:16]},
            )
        except Exception as e:
            self._warn(f"Failed writing named cache '{name}': {e}")

    @_serialized
    def put_named_reference(self, name: str, signature: str, data_key: str) -> None:
        key = str(data_key or "").strip()
        if not key:
            return
        p = self.dataframe_cache_path(key)
        if not p.exists():
            return
        try:
            st = p.stat()
        except Exception:
            return
        named = self.manifest.setdefault("named", {})
        named[str(name)] = {
            "kind": "dataframe-ref",
            "signature": str(signature),
            "ref_key": key,
            "size": int(st.st_size),
        }
        self._written_named_keys.add(str(name))
        self._used_named_names.add(str(name))
        self._set_manifest_item("named", str(name), named[str(name)])

    @_serialized
    def get_named(self, name: str, signature: str):
        if self.rebuild and str(name) not in self._written_named_keys:
            return None
        self.manifest = self._load_json(self.manifest_path, default=self.manifest)
        named = self.manifest.get("named", {})
        item = named.get(str(name))
        if not item:
            return None
        if str(item.get("signature")) != str(signature):
            return None
        self._used_named_names.add(str(name))
        try:
            kind = str(item.get("kind", "dataframe")).strip().lower()
            if kind == "dataframe-ref":
                ref_key = str(item.get("ref_key", "")).strip()
                if not ref_key:
                    return None
                try:
                    return self.get_dataframe(ref_key)
                except Exception:
                    return None
            p = self._track_path(self.root / item["path"])
            if not p.exists():
                return None
            try:
                st = p.stat()
                expect_size = item.get("size")
                if expect_size is not None and int(expect_size) != int(st.st_size):
                    return None
            except Exception:
                return None
            expect_sha1 = item.get("sha1")
            if expect_sha1:
                try:
                    real_sha1 = self._sha1_file(str(p))
                except Exception:
                    return None
                if str(real_sha1) != str(expect_sha1):
                    return None
            return pd.read_pickle(p)
        except Exception:
            return None
