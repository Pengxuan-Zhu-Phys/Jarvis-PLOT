"""One lightweight version source for CLI, manuals and output provenance."""
from importlib.metadata import PackageNotFoundError, version as distribution_version
from pathlib import Path


def get_version():
    """Prefer installed metadata; use VERSION only in an uninstalled checkout."""
    for name in ('JarvisPLOT', 'Jarvis-PLOT'):
        try:
            return distribution_version(name)
        except PackageNotFoundError:
            continue
    root = Path(__file__).resolve().parent.parent
    if (root/'pyproject.toml').is_file():
        try:
            value = (root/'VERSION').read_text(encoding='utf-8').strip()
        except OSError:
            value = ''
        if value:
            return value
    return '0.0.0'
