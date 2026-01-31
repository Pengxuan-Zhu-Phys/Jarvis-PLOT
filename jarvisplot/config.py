#!/usr/bin/env python3 

from __future__ import annotations
from pathlib import Path
from typing import Optional, Any, Dict, List
import yaml
import os 

class ConfigLoader:
    """Lightweight config holder with auto-updating path fields.

    Setting `self.file` updates:
      - `self.path`  : absolute file path (str)
      - `self.dir`   : directory path (str)
      - `self.stem`  : filename without suffix (str)
      - `self.ext`   : file suffix like ".yaml" (str)
    """
    def __init__(self) -> None:
        self._file:     Optional[str]   = None
        self.logger:    Any             = None
        self.path:      Optional[str]   = None
        self.dir:       Optional[str]   = None
        self.stem:      Optional[str]   = None
        self.ext:       Optional[str]   = None
        self.config: Optional[Dict[str, Any]] = None

    # ---- property: file ----
    @property
    def file(self) -> Optional[str]:
        return self._file

    @file.setter
    def file(self, value: Optional[str]) -> None:
        if value is None:
            # reset all derived fields
            self._file = None
            self.path = None
            self.dir = None
            self.stem = None
            self.ext = None
            return
        p = Path(value).expanduser().resolve()
        self._file = str(p)
        self.path = os.path.abspath(p)
        self.dir = os.path.dirname(p)
        self.stem = p.stem
        self.ext = p.suffix

    # ---- helpers ----
    def load(self) -> Dict[str, Any]:
        """Load YAML from `self.file` into `self.config` and return it."""
        if not self._file:
            raise ValueError("ConfigLoader.file is not set")
        with open(self._file, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        return self.config or {}

    def __repr__(self) -> str:  # pragma: no cover
        return f"ConfigLoader(file={self._file!r}, dir={self.dir!r})"
        
    def update_dataset(self, target_name, updates: dict):
        for d in self.config["DataSet"]:
            if d.get("name") == target_name:
                print(d.keys())
                d.update(updates)
                print(d.keys())
                return True   # 成功更新
        return False          # 没找到

