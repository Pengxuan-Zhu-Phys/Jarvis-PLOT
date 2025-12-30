# src/utils/interpolator.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Literal


BoundsMode = Literal["clamp", "extrapolate", "nan", "error"]


@dataclass(frozen=True)
class CSVSourceSpec:
    """Where to read x/y pairs from."""
    path: str           # relative to YAML dir or absolute
    x: str              # column name (or index key later)
    y: str              # column name
    sort_by: Optional[str] = None
    drop_duplicates: Optional[str] = None


@dataclass(frozen=True)
class Interp1DSpec:
    """Interpolation method + options."""
    kind: str = "linear"
    bounds: BoundsMode = "clamp"
    fill_value: Optional[float] = None
    assume_sorted: bool = False


@dataclass(frozen=True)
class InterpolatorSpec:
    """Single named interpolator spec parsed from YAML."""
    name: str
    source: CSVSourceSpec
    method: str = "interp1d"          # keep extensible: "interp1d", "pchip", ...
    options: Interp1DSpec = Interp1DSpec()


class LazyCallable:
    """
    A callable wrapper that builds the real function on first call.

    IMPORTANT: This is what you inject into eval globals so Figure/layer code
    doesn't need to know about lazy loading.
    """
    def __init__(self, name: str, builder: Callable[[], Callable], logger=None):
        self._name = name
        self._builder = builder
        self._logger = logger
        self._fn: Optional[Callable] = None

    def _ensure(self) -> Callable:
        if self._fn is None:
            if self._logger:
                self._logger.debug(f"LazyLoad: building interpolator '{self._name}'")
            self._fn = self._builder()
        return self._fn

    def __call__(self, *args, **kwargs):
        fn = self._ensure()
        return fn(*args, **kwargs)


class InterpolatorManager:
    """
    Parses YAML 'Functions' and provides lazy-callable interpolators.

    - Parsing stage: stores specs only (no IO, no scipy work)
    - Runtime stage: first call triggers CSV read + interpolator build
    - Cache: built callables are cached for reuse in the same run

    Suggested usage:
      mgr = InterpolatorManager.from_yaml(cfg, yaml_dir, shared, logger)
      funcs_dict = mgr.as_eval_funcs()     # inject into expression globals
      # expression can call f_cut(x) directly
    """

    def __init__(self, yaml_dir: Path, shared=None, logger=None):
        self._yaml_dir = Path(yaml_dir)
        self._shared = shared
        self._logger = logger

        self._specs: Dict[str, InterpolatorSpec] = {}
        self._lazy: Dict[str, LazyCallable] = {}
        self._built: Dict[str, Callable] = {}

    # -------------------------
    # YAML parsing (no IO)
    # -------------------------
    @classmethod
    def from_yaml(
        cls,
        config: Any,
        yaml_dir: str | Path,
        shared=None,
        logger=None,
        key: str = "Functions",
    ) -> "InterpolatorManager":
        """
        Parse config[key] list into specs and register lazy wrappers.

        Accepts either the full YAML config dict (with top-level key 'Functions'), or
        directly a list (cfg['Functions']).
        """
        mgr = cls(yaml_dir=Path(yaml_dir), shared=shared, logger=logger)

        # Accept either the full YAML config dict (preferred) or directly a list (cfg['Functions']).
        if isinstance(config, list):
            items = config
        elif isinstance(config, dict):
            items = config.get(key, []) or []
        else:
            items = []

        for item in items:
            spec = mgr._parse_one(item)
            mgr.register(spec)
        return mgr

    def _parse_one(self, item: Dict[str, Any]) -> InterpolatorSpec:
        """
        Convert a single YAML dict -> InterpolatorSpec.

        Expected YAML shape (example):
          - name: f_cut
            source: {type: csv, path: ./a.csv, x: x, y: y, sort_by: x, drop_duplicates: x}
            method: interp1d
            options: {kind: linear, bounds: clamp, fill_value: null, assume_sorted: false}

        Top-level YAML key is 'Functions'.
        """
        # NOTE: only parse + validate keys here; do NOT read files.
        if not isinstance(item, dict):
            raise TypeError(f"Function spec must be a dict, got {type(item)}")

        name = item.get("name", None)
        if not name:
            raise ValueError("Function spec missing required field 'name'")

        source = item.get("source", None)
        if not isinstance(source, dict):
            raise ValueError(f"Function '{name}': missing/invalid 'source'")

        stype = source.get("type", None)
        if stype != "csv":
            raise ValueError(f"Function '{name}': only source.type='csv' is supported (got {stype!r})")

        path = source.get("path", None)
        xcol = source.get("x", None)
        ycol = source.get("y", None)
        if not path or xcol is None or ycol is None:
            raise ValueError(f"Function '{name}': source requires 'path', 'x', 'y'")

        src = CSVSourceSpec(
            path=str(path),
            x=str(xcol),
            y=str(ycol),
            sort_by=source.get("sort_by", None),
            drop_duplicates=source.get("drop_duplicates", None),
        )

        method = item.get("method", "interp1d")
        if method != "interp1d":
            raise ValueError(f"Function '{name}': only method='interp1d' is supported (got {method!r})")

        opt0 = item.get("options", {}) or {}
        if not isinstance(opt0, dict):
            raise ValueError(f"Function '{name}': options must be a dict")

        opt = Interp1DSpec(
            kind=str(opt0.get("kind", "linear")),
            bounds=str(opt0.get("bounds", "clamp")),
            fill_value=opt0.get("fill_value", None),
            assume_sorted=bool(opt0.get("assume_sorted", False)),
        )

        if opt.kind != "linear":
            raise ValueError(f"Function '{name}': only options.kind='linear' is supported (got {opt.kind!r})")

        return InterpolatorSpec(name=name, source=src, method=method, options=opt)

    # -------------------------
    # Registry
    # -------------------------
    def register(self, spec: InterpolatorSpec) -> None:
        """
        Register a spec and create a LazyCallable wrapper for it.
        """
        name = spec.name
        if name in self._specs:
            raise ValueError(f"Interpolator '{name}' already registered.")
        self._specs[name] = spec

        # wrap a builder closure; builder will do IO + build actual callable
        self._lazy[name] = LazyCallable(name, builder=lambda n=name: self._build(n), logger=self._logger)

    def get(self, name: str) -> Callable:
        """
        Return callable for this interpolator. (LazyCallable until built.)
        """
        if name in self._built:
            return self._built[name]
        if name in self._lazy:
            return self._lazy[name]
        raise KeyError(f"Unknown interpolator '{name}'")

    def as_eval_funcs(self) -> Dict[str, Callable]:
        """
        Dict suitable for injection into expression evaluation globals.
        Keys are interpolator names; values are LazyCallable instances.
        """
        return dict(self._lazy)

    def summary(self) -> Dict[str, Any]:
        """
        Small diagnostic summary: registered / built status.
        """
        return {
            "registered": sorted(self._specs.keys()),
            "built": sorted(self._built.keys()),
        }

    # -------------------------
    # Lazy build (IO + compute)
    # -------------------------
    def _build(self, name: str) -> Callable:
        """
        Build the real interpolator callable for `name`, cache it, and return it.

        Responsibilities:
          - read (x, y) from CSV (may use shared cache)
          - construct interpolator according to method/options
          - return a vectorized callable f(x) -> y
        """
        if name in self._built:
            return self._built[name]
        if name not in self._specs:
            raise KeyError(f"Unknown interpolator '{name}'")

        spec = self._specs[name]

        # Load data
        x, y = self._load_csv_xy(spec.source)

        # Build callable
        if spec.method != "interp1d":
            raise ValueError(f"Function '{name}': unsupported method {spec.method!r}")

        fn = self._make_interp1d(x, y, spec.options)
        self._built[name] = fn
        return fn

    # -------------------------
    # Helpers (placeholders)
    # -------------------------
    def _resolve_path(self, p: str) -> Path:
        """
        Resolve YAML-relative path or absolute path.
        """
        p = str(p)
        if p.startswith("&JP/"):
            # allow internal JP paths if needed; resolve relative to project root (yaml_dir's parent)
            # but keep simple: treat as relative to yaml_dir for now.
            p = p[4:]
        path = Path(p).expanduser()
        if not path.is_absolute():
            path = (self._yaml_dir / path).resolve()
        return path

    def _load_csv_xy(self, spec: CSVSourceSpec) -> Tuple[Any, Any]:
        """
        Read/clean CSV and return arrays (x, y).
        """
        import pandas as _pd
        import numpy as _np

        csv_path = self._resolve_path(spec.path)
        df = _pd.read_csv(csv_path)

        # Optional sort/dedup on a specified column name
        if spec.sort_by is not None and spec.sort_by in df.columns:
            df = df.sort_values(by=spec.sort_by)

        if spec.drop_duplicates is not None and spec.drop_duplicates in df.columns:
            df = df.drop_duplicates(subset=spec.drop_duplicates, keep="first")

        # Allow spec.x/spec.y to be either column names OR python expressions.
        import math as _math
        from ..inner_func import update_funcs as _update_funcs

        local_vars = df.to_dict("series")
        allowed_globals = _update_funcs({"np": _np, "math": _math})

        def _eval_field(field: str):
            if field in df.columns:
                return _np.asarray(df[field].values, dtype=float)
            # treat as expression
            try:
                arr = eval(field, allowed_globals, local_vars)
                return _np.asarray(arr, dtype=float)
            except Exception as e:
                raise ValueError(
                    f"CSV {csv_path}: cannot evaluate field '{field}'. "
                    f"Not a column and eval failed: {e}"
                )

        x = _eval_field(spec.x)
        y = _eval_field(spec.y)

        if x.shape != y.shape:
            raise ValueError(f"CSV {csv_path}: evaluated x/y have different shapes: {x.shape} vs {y.shape}")

        # Ensure x is sorted for interpolation unless explicitly assumed sorted.
        # Even if user sorted by another column, we still need monotonic x.
        order = _np.argsort(x)
        x = x[order]
        y = y[order]

        return x, y

    def _make_interp1d(self, x, y, opt: Interp1DSpec) -> Callable:
        """
        Create interp1d-like callable based on opt.
        """
        import numpy as _np

        x = _np.asarray(x, dtype=float)
        y = _np.asarray(y, dtype=float)

        if x.ndim != 1 or y.ndim != 1 or x.shape[0] < 2 or x.shape != y.shape:
            raise ValueError("interp1d: x and y must be 1D arrays of the same length >= 2")

        xmin = float(_np.min(x))
        xmax = float(_np.max(x))

        def _linear_extrap(xq: _np.ndarray) -> _np.ndarray:
            # linear extrapolation using the first/last segments
            yq = _np.interp(xq, x, y)
            left = xq < xmin
            right = xq > xmax
            if left.any():
                x0, x1 = x[0], x[1]
                y0, y1 = y[0], y[1]
                slope = (y1 - y0) / (x1 - x0)
                yq[left] = y0 + slope * (xq[left] - x0)
            if right.any():
                x0, x1 = x[-2], x[-1]
                y0, y1 = y[-2], y[-1]
                slope = (y1 - y0) / (x1 - x0)
                yq[right] = y1 + slope * (xq[right] - x1)
            return yq

        def f(xq):
            xq = _np.asarray(xq, dtype=float)
            if opt.bounds == "clamp":
                xqc = _np.clip(xq, xmin, xmax)
                return _np.interp(xqc, x, y)
            elif opt.bounds == "extrapolate":
                return _linear_extrap(xq)
            elif opt.bounds == "nan":
                yq = _np.interp(_np.clip(xq, xmin, xmax), x, y)
                out = (xq < xmin) | (xq > xmax)
                if out.any():
                    yq = yq.astype(float, copy=False)
                    yq[out] = _np.nan
                return yq
            elif opt.bounds == "error":
                if _np.any((xq < xmin) | (xq > xmax)):
                    raise ValueError("interp1d: query x is out of bounds")
                return _np.interp(xq, x, y)
            else:
                # fallback
                xqc = _np.clip(xq, xmin, xmax)
                return _np.interp(xqc, x, y)

        return f