"""Literal font metrics for forest's content-sized table columns."""
from functools import lru_cache
from pathlib import Path
import tempfile

_font_directory = None


@lru_cache(maxsize=32)
def font_path(family, weight):
    """Resolve the actual regular/bold face, including macOS TTC collections.

    Matplotlib resolves a TTC file but renders its first face. Extract the
    selected installed face to a process-local temporary file; never package it.
    fontTools is an existing Matplotlib dependency.
    """
    from matplotlib.font_manager import FontProperties, findfont
    try:
        path = findfont(FontProperties(family=family, weight=weight), fallback_to_default=False)
    except ValueError as exc:
        raise ValueError(f'forest font {family!r} is unavailable; install it or explicitly select an installed family') from exc
    if Path(path).suffix.lower() != '.ttc':
        return path
    from fontTools.ttLib import TTCollection
    collection = TTCollection(path)
    target = 700 if weight == 'bold' else 400
    candidates = []
    for index, face in enumerate(collection.fonts):
        names = {n.toUnicode().lower() for n in face['name'].names if n.nameID in (1, 16)}
        if family.lower() not in names:
            continue
        italic = bool(face['head'].macStyle & 2)
        candidates.append((italic, abs(face['OS/2'].usWeightClass-target), index))
    if not candidates:
        collection.close()
        raise ValueError(f'forest font {family!r}: no matching face in {path}')
    index = min(candidates)[2]
    global _font_directory
    if _font_directory is None:
        _font_directory = tempfile.TemporaryDirectory(prefix='jarvisplot-font-')
    output = Path(_font_directory.name)/f'{len(list(Path(_font_directory.name).iterdir()))}-{weight}.ttf'
    collection.fonts[index].save(output)
    collection.close()
    return str(output)


def font_properties(size_pt, family, weight='normal'):
    from matplotlib.font_manager import FontProperties
    return FontProperties(fname=font_path(family, weight), family=family, size=size_pt, weight=weight)


@lru_cache(maxsize=4096)
def text_metrics_mm(text, size_pt, family, weight):
    """Return advance and left/right ink overhang, in physical mm."""
    from matplotlib.textpath import TextToPath
    from matplotlib.path import Path
    from .forest_contract import MM_PER_PT
    if '\n' in text or '\r' in text or '\t' in text:
        raise ValueError(f'forest column text must be a single line without tabs: {text!r}')
    prop = font_properties(size_pt, family, weight)
    converter = TextToPath()
    advance, _, _ = converter.get_text_width_height_descent(text, prop, ismath=False)
    vertices, codes = converter.get_text_path(prop, text, ismath=False)
    left = right = 0.0
    if vertices:
        bounds = Path(vertices, codes).get_extents()
        scale = size_pt/100
        left = max(0, -bounds.x0*scale)
        right = max(0, bounds.x1*scale-advance)
    return advance*MM_PER_PT, left*MM_PER_PT, right*MM_PER_PT


def wrap_text(text, size_pt, family, weight, width_mm):
    """Wrap literal editorial copy at spaces, preserving explicit line breaks."""
    result = []
    for paragraph in text.splitlines():
        current = ''
        for word in paragraph.split():
            candidate = f'{current} {word}' if current else word
            if current and text_metrics_mm(candidate, size_pt, family, weight)[0] > width_mm:
                result.append(current)
                current = word
            else:
                current = candidate
        result.append(current)
    return result
