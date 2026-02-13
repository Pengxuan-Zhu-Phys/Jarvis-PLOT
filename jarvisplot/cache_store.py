#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
import hashlib
import json
import os
import re
import shutil
import time

import pandas as pd


class ProjectCache:
    """Workdir-local cache store: <workdir>/.cache."""

    def __init__(self, workdir: str, logger=None, rebuild: bool = False):
        self.logger = logger
        self.workdir = Path(workdir).expanduser().resolve()
        self.root = self.workdir / ".cache"
        self.data_dir = self.root / "data"
        self.summary_dir = self.root / "summary"
        self.named_dir = self.root / "named"
        self.manifest_path = self.root / "manifest.json"
        self.rebuild = bool(rebuild)

        if self.rebuild and self.root.exists():
            shutil.rmtree(self.root, ignore_errors=True)

        self.root.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.summary_dir.mkdir(parents=True, exist_ok=True)
        self.named_dir.mkdir(parents=True, exist_ok=True)

        self.manifest: Dict[str, Any] = self._load_json(
            self.manifest_path,
            default={"schema": 1, "files": {}, "named": {}},
        )

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

    def _save_manifest(self) -> None:
        try:
            with open(self.manifest_path, "w", encoding="utf-8") as f:
                json.dump(self.manifest, f, ensure_ascii=True, indent=2, sort_keys=True)
        except Exception as e:
            self._warn(f"Failed writing cache manifest: {e}")

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
        self._save_manifest()

        fp = {"path": abs_path, "size": size, "mtime_ns": mtime_ns, "md5": md5}
        if extra:
            fp.update(extra)
        return fp

    def get_dataframe(self, key: str):
        if self.rebuild:
            return None
        p = self.data_dir / f"{key}.pkl"
        if not p.exists():
            return None
        try:
            return pd.read_pickle(p)
        except Exception as e:
            self._warn(f"Failed loading dataframe cache '{p}': {e}")
            return None

    def put_dataframe(self, key: str, df, meta: Optional[Dict[str, Any]] = None) -> None:
        p = self.data_dir / f"{key}.pkl"
        try:
            df.to_pickle(p)
            if meta is not None:
                m = self.data_dir / f"{key}.json"
                with open(m, "w", encoding="utf-8") as f:
                    json.dump(meta, f, ensure_ascii=True, indent=2, sort_keys=True, default=str)
        except Exception as e:
            self._warn(f"Failed writing dataframe cache '{p}': {e}")

    def _summary_path(self, source_fp: Dict[str, Any]) -> Path:
        key = self.cache_key({"kind": "summary", "source": source_fp})
        return self.summary_dir / f"{key}.txt"

    def get_summary(self, source_fp: Dict[str, Any]) -> Optional[str]:
        if self.rebuild:
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
            p.write_text(str(text), encoding="utf-8")
        except Exception as e:
            self._warn(f"Failed writing summary cache '{p}': {e}")

    @staticmethod
    def _safe_name(name: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name))

    def put_named(self, name: str, signature: str, df) -> None:
        safe = self._safe_name(name)
        slot = self.cache_key({"name": name, "signature": signature})[:12]
        p = self.named_dir / f"{safe}__{slot}.pkl"
        try:
            df.to_pickle(p)
            named = self.manifest.setdefault("named", {})
            named[str(name)] = {
                "signature": str(signature),
                "path": str(p.relative_to(self.root)),
            }
            self._save_manifest()
        except Exception as e:
            self._warn(f"Failed writing named cache '{name}': {e}")

    def get_named(self, name: str, signature: str):
        if self.rebuild:
            return None
        named = self.manifest.get("named", {})
        item = named.get(str(name))
        if not item:
            return None
        if str(item.get("signature")) != str(signature):
            return None
        try:
            p = self.root / item["path"]
            if not p.exists():
                return None
            return pd.read_pickle(p)
        except Exception:
            return None
