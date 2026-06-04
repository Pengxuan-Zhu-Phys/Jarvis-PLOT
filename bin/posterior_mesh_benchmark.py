#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd

from jarvisplot.Figure.density_cell_runtime import density_cell
from jarvisplot.Figure.posterior_mesh import build_posterior_mesh, mesh_diagnostics_frame


@dataclass
class BenchmarkCase:
    name: str
    samples: pd.DataFrame
    support: pd.DataFrame
    frame: pd.DataFrame
    mesh: object
    correlation: float
    refinement_trace: list[dict[str, Any]]
    generator_history: list[np.ndarray]


def _logger():
    return SimpleNamespace(
        debug=lambda *args, **kwargs: None,
        info=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
        error=lambda *args, **kwargs: None,
    )


def _clip_unit_square(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.clip(x, 0.0, 1.0), np.clip(y, 0.0, 1.0)


def _blob_case(rng: np.random.Generator, n: int) -> pd.DataFrame:
    pts = rng.multivariate_normal(
        mean=np.array([0.50, 0.50]),
        cov=np.array([[0.010, 0.0], [0.0, 0.010]]),
        size=n,
    )
    x, y = _clip_unit_square(pts[:, 0], pts[:, 1])
    return pd.DataFrame({"x": x, "y": y, "weight": np.ones(n, dtype=float)})


def _ring_case(rng: np.random.Generator, n: int) -> pd.DataFrame:
    theta = rng.uniform(0.0, 2.0 * np.pi, size=n)
    radius = 0.26 + 0.018 * rng.normal(size=n)
    x = 0.50 + radius * np.cos(theta)
    y = 0.50 + radius * np.sin(theta)
    x, y = _clip_unit_square(x, y)
    return pd.DataFrame({"x": x, "y": y, "weight": np.ones(n, dtype=float)})


def _banana_case(rng: np.random.Generator, n: int) -> pd.DataFrame:
    t = rng.uniform(-1.0, 1.0, size=n)
    x = 0.5 + 0.28 * t
    y = 0.42 + 0.18 * (t**2 - 0.35) + 0.06 * t
    n1 = rng.normal(size=n)
    n2 = rng.normal(size=n)
    x += 0.012 * n1 - 0.008 * n2
    y += 0.008 * n1 + 0.012 * n2
    x, y = _clip_unit_square(x, y)
    return pd.DataFrame({"x": x, "y": y, "weight": np.ones(n, dtype=float)})


def _ridge_case(rng: np.random.Generator, n: int) -> pd.DataFrame:
    t = rng.uniform(0.0, 1.0, size=n)
    x = 0.12 + 0.76 * t
    y = 0.18 + 0.54 * t
    ortho = rng.normal(size=n)
    x += -0.010 * ortho
    y += 0.010 * ortho
    x, y = _clip_unit_square(x, y)
    return pd.DataFrame({"x": x, "y": y, "weight": np.ones(n, dtype=float)})


def _density_cfg(seed: int, bins: int, include_polygons: bool, *, adaptive: bool) -> dict:
    cfg = {
        "method": "bridson",
        "coordinates": {
            "x": {"name": "x", "lim": [0.0, 1.0], "scale": "linear"},
            "y": {"name": "y", "lim": [0.0, 1.0], "scale": "linear"},
            "weight": {"name": "weight"},
        },
        "bridson": {"bin": int(bins), "seed": int(seed)},
        "normalize": True,
        "diagnostics": False,
        "_mesh_debug": {"include_polygons": bool(include_polygons)},
    }
    if adaptive:
        cfg["refinement"] = {
            "enabled": True,
            "iterations": 6,
            "alpha": 0.30,
            "eta": 0.50,
            "anisotropic": True,
            "split_enabled": True,
            "merge_enabled": True,
            "split_quantile": 0.75,
            "merge_quantile": 0.20,
            "split_offset": 0.75,
            "min_separation": 0.005,
            "max_splits": 2,
            "max_merges": 1,
            "max_generators": 64,
            "record_history": True,
            "seed": int(seed),
        }
    return cfg


def _safe_corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    finite = np.isfinite(a) & np.isfinite(b)
    if np.count_nonzero(finite) < 2:
        return float("nan")
    a = a[finite]
    b = b[finite]
    if np.nanstd(a) <= 0.0 or np.nanstd(b) <= 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _run_case(name: str, samples: pd.DataFrame, seed: int, bins: int, *, adaptive: bool) -> BenchmarkCase:
    cfg = _density_cfg(seed=seed, bins=bins, include_polygons=True, adaptive=adaptive)
    out = density_cell(samples, cfg, _logger())
    mesh = density_cell.last_mesh
    if mesh is None:
        raise RuntimeError(f"{name}: posterior mesh was not recorded.")
    frame = mesh_diagnostics_frame(mesh, x=out["x"].to_numpy(dtype=float), y=out["y"].to_numpy(dtype=float))
    correlation = _safe_corrcoef(frame["mass_stress"].to_numpy(dtype=float), frame["geometry_stress"].to_numpy(dtype=float))
    trace = list(mesh.diagnostics.get("refinement_trace", []))
    history = list(mesh.diagnostics.get("generator_history", []))
    return BenchmarkCase(
        name=name,
        samples=samples,
        support=out,
        frame=frame,
        mesh=mesh,
        correlation=correlation,
        refinement_trace=trace,
        generator_history=history,
    )


def _draw_polygons(ax, polygons: Iterable[object], *, color: str = "0.78", alpha: float = 0.45, linewidth: float = 0.55) -> None:
    for geom in polygons:
        if geom is None or getattr(geom, "is_empty", False):
            continue
        geoms = getattr(geom, "geoms", None)
        if geoms is not None:
            for item in geoms:
                _draw_polygons(ax, [item], color=color, alpha=alpha, linewidth=linewidth)
            continue
        if getattr(geom, "geom_type", "") != "Polygon":
            continue
        x, y = geom.exterior.xy
        ax.plot(x, y, color=color, alpha=alpha, linewidth=linewidth, zorder=0)


def _draw_ellipses(ax, frame: pd.DataFrame, *, scale: float = 1.0) -> None:
    for row in frame.itertuples(index=False):
        if not np.isfinite(row.ellipse_width) or not np.isfinite(row.ellipse_height):
            continue
        if row.ellipse_width <= 0.0 or row.ellipse_height <= 0.0:
            continue
        ellipse = Ellipse(
            (float(row.x), float(row.y)),
            width=float(row.ellipse_width) * scale,
            height=float(row.ellipse_height) * scale,
            angle=float(row.ellipse_angle_deg),
            fill=False,
            edgecolor="white",
            linewidth=0.6,
            alpha=0.35,
            zorder=3,
        )
        ax.add_patch(ellipse)


def _draw_vectors(ax, frame: pd.DataFrame, *, kind: str) -> None:
    if kind == "drift":
        u = frame["drift_x_norm"].to_numpy(dtype=float)
        v = frame["drift_y_norm"].to_numpy(dtype=float)
        color = "black"
        scale = 0.16
        alpha = 0.35
    elif kind == "principal":
        u = frame["principal_dx"].to_numpy(dtype=float)
        v = frame["principal_dy"].to_numpy(dtype=float)
        color = "white"
        scale = 0.08
        alpha = 0.55
    else:
        raise ValueError(f"unknown vector kind: {kind}")

    x = frame["x"].to_numpy(dtype=float)
    y = frame["y"].to_numpy(dtype=float)
    ax.quiver(
        x,
        y,
        u * scale,
        v * scale,
        angles="xy",
        scale_units="xy",
        scale=1.0,
        color=color,
        width=0.0022,
        alpha=alpha,
        zorder=4,
    )


def _mass_norm(values: list[np.ndarray]) -> tuple[float, float]:
    finite = np.concatenate([np.asarray(v, dtype=float)[np.isfinite(v)] for v in values if np.asarray(v).size])
    if finite.size == 0:
        return 0.0, 1.0
    lo = float(np.nanpercentile(finite, 5.0))
    hi = float(np.nanpercentile(finite, 95.0))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.nanmin(finite))
        hi = float(np.nanmax(finite))
        if hi <= lo:
            hi = lo + 1.0
    return lo, hi


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_cases(seed: int, bins: int, n: int, *, adaptive: bool) -> list[BenchmarkCase]:
    rng = np.random.default_rng(int(seed))
    cases = [
        ("smooth_blob", _blob_case(rng, n)),
        ("ring", _ring_case(rng, n)),
        ("banana", _banana_case(rng, n)),
        ("ridge", _ridge_case(rng, n)),
    ]
    built = []
    for idx, (name, samples) in enumerate(cases):
        built.append(_run_case(name, samples, seed=seed + idx * 17, bins=bins, adaptive=adaptive))
    return built


def _render_case_snapshot(case: BenchmarkCase, generators: np.ndarray, out_path: Path, *, title: str) -> None:
    mesh = build_posterior_mesh(
        generators,
        case.samples[["x", "y"]].to_numpy(dtype=float),
        case.samples["weight"].to_numpy(dtype=float),
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.0),
        include_polygons=True,
        strict_geometry=False,
        refinement=None,
    )
    frame = mesh_diagnostics_frame(mesh, x=mesh.generators[:, 0], y=mesh.generators[:, 1])
    fig, ax = plt.subplots(1, 1, figsize=(6.8, 6.2), constrained_layout=True)
    _draw_polygons(ax, getattr(mesh, "polygons", None) or [])
    ax.scatter(case.samples["x"], case.samples["y"], s=4, c="0.86", alpha=0.55, linewidths=0.0, zorder=1)
    ax.scatter(frame["x"], frame["y"], s=20, c=frame["geometry_stress"], cmap="viridis", linewidths=0.0, zorder=2)
    _draw_ellipses(ax, frame, scale=0.8)
    _draw_vectors(ax, frame, kind="drift")
    _draw_vectors(ax, frame, kind="principal")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _render_history(case: BenchmarkCase, outdir: Path) -> list[Path]:
    history = case.generator_history
    if len(history) < 2:
        return []
    frames_dir = outdir / f"{case.name}_history_frames"
    _ensure_dir(frames_dir)
    paths: list[Path] = []
    for idx, generators in enumerate(history):
        path = frames_dir / f"frame_{idx:03d}.png"
        title = f"{case.name}: iteration {idx}"
        if idx > 0 and idx - 1 < len(case.refinement_trace):
            info = case.refinement_trace[idx - 1]
            title += f" | splits={info.get('splits', 0)} merges={info.get('merges', 0)} reseed={info.get('reseed', 0)}"
        _render_case_snapshot(case, np.asarray(generators, dtype=float), path, title=title)
        paths.append(path)
    try:
        from PIL import Image

        images = [Image.open(path).convert("P") for path in paths]
        if images:
            gif_path = outdir / f"{case.name}_history.gif"
            images[0].save(
                gif_path,
                save_all=True,
                append_images=images[1:],
                duration=650,
                loop=0,
            )
            for image in images:
                image.close()
    except Exception:
        pass
    return paths


def render_benchmark(cases: list[BenchmarkCase], outdir: Path) -> Path:
    _ensure_dir(outdir)
    mass_lo, mass_hi = _mass_norm([case.frame["mass_stress"].to_numpy(dtype=float) for case in cases])
    geom_lo, geom_hi = _mass_norm([case.frame["geometry_stress"].to_numpy(dtype=float) for case in cases])

    fig, axes = plt.subplots(len(cases), 4, figsize=(22, 4.2 * len(cases)), constrained_layout=True)
    if len(cases) == 1:
        axes = np.asarray([axes])

    combined = []
    for row, case in enumerate(cases):
        frame = case.frame.copy()
        frame.insert(0, "scenario", case.name)
        combined.append(frame)

        ax0, ax1, ax2, ax3 = axes[row]
        samples = case.samples
        support = case.support
        mesh = case.mesh
        polygons = getattr(mesh, "polygons", None) or []

        ax0.scatter(samples["x"], samples["y"], s=5, c="0.82", alpha=0.55, linewidths=0.0, zorder=1)
        _draw_polygons(ax0, polygons)
        ax0.scatter(support["x"], support["y"], s=12, c="black", alpha=0.95, linewidths=0.0, zorder=2)
        _draw_ellipses(ax0, frame, scale=0.8)
        _draw_vectors(ax0, frame, kind="drift")
        _draw_vectors(ax0, frame, kind="principal")
        ax0.set_xlim(0.0, 1.0)
        ax0.set_ylim(0.0, 1.0)
        ax0.set_aspect("equal", adjustable="box")
        ax0.set_title(f"{case.name}: support mesh")
        ax0.set_xlabel("x")
        ax0.set_ylabel("y")

        sc1 = ax1.scatter(
            frame["x"],
            frame["y"],
            c=frame["mass_stress"],
            s=18,
            cmap="magma",
            vmin=mass_lo,
            vmax=mass_hi,
            linewidths=0.0,
        )
        ax1.set_xlim(0.0, 1.0)
        ax1.set_ylim(0.0, 1.0)
        ax1.set_aspect("equal", adjustable="box")
        ax1.set_title(f"{case.name}: mass-driven stress")
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        fig.colorbar(sc1, ax=ax1, fraction=0.046, pad=0.04)

        sc2 = ax2.scatter(
            frame["x"],
            frame["y"],
            c=frame["geometry_stress"],
            s=18,
            cmap="viridis",
            vmin=geom_lo,
            vmax=geom_hi,
            linewidths=0.0,
        )
        ax2.set_xlim(0.0, 1.0)
        ax2.set_ylim(0.0, 1.0)
        ax2.set_aspect("equal", adjustable="box")
        ax2.set_title(f"{case.name}: geometry-driven stress")
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        fig.colorbar(sc2, ax=ax2, fraction=0.046, pad=0.04)

        finite = np.isfinite(frame["mass_stress"]) & np.isfinite(frame["geometry_stress"])
        xs = frame.loc[finite, "mass_stress"].to_numpy(dtype=float)
        ys = frame.loc[finite, "geometry_stress"].to_numpy(dtype=float)
        ax3.scatter(xs, ys, s=16, c=frame.loc[finite, "anisotropy_ratio"], cmap="plasma", alpha=0.8, linewidths=0.0)
        lo = float(np.nanmin([np.nanmin(xs), np.nanmin(ys)])) if xs.size and ys.size else 0.0
        hi = float(np.nanmax([np.nanmax(xs), np.nanmax(ys)])) if xs.size and ys.size else 1.0
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            ax3.plot([lo, hi], [lo, hi], color="0.4", linestyle="--", linewidth=1.0)
        ax3.set_xlabel("mass-driven stress")
        ax3.set_ylabel("geometry-driven stress")
        ax3.set_title(f"{case.name}: comparison\ncorr={case.correlation:.3f}")
        ax3.grid(True, alpha=0.18)

    combined_df = pd.concat(combined, ignore_index=True)
    csv_path = outdir / "posterior_mesh_benchmark.csv"
    combined_df.to_csv(csv_path, index=False)

    for case in cases:
        _render_history(case, outdir)

    fig.suptitle("PosteriorMesh diagnostics: mass-driven versus geometry-driven stress", y=1.01)
    png_path = outdir / "posterior_mesh_benchmark.png"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return png_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark PosteriorMesh diagnostics on synthetic posterior geometries.")
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent.parent / "Results" / "posterior_mesh_benchmark"),
        help="Directory for the benchmark PNG/CSV outputs.",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed for synthetic posteriors.")
    parser.add_argument("--bins", type=int, default=18, help="Bridson bin count used to build the support mesh.")
    parser.add_argument("--samples", type=int, default=1600, help="Number of raw posterior samples per scenario.")
    parser.add_argument("--no-adaptive", dest="adaptive", action="store_false", help="Disable anisotropic adaptive refinement in the benchmark.")
    parser.set_defaults(adaptive=True)
    args = parser.parse_args()

    outdir = Path(args.output_dir).expanduser().resolve()
    cases = build_cases(seed=args.seed, bins=args.bins, n=args.samples, adaptive=bool(args.adaptive))
    png_path = render_benchmark(cases, outdir)
    print(png_path)
    print(outdir / "posterior_mesh_benchmark.csv")
    for case in cases:
        print(f"{case.name}: corr={case.correlation:.6f} history={len(case.generator_history)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
