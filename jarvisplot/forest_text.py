"""Literal font metrics for forest's content-sized table columns."""
from functools import lru_cache
from pathlib import Path
import re
import tempfile

_font_directory = None


@lru_cache(maxsize=32)
def font_path(family, weight, style='normal'):
    """Resolve the actual regular/bold face, including macOS TTC collections.

    Matplotlib resolves a TTC file but renders its first face. Extract the
    selected installed face to a process-local temporary file; never package it.
    fontTools is an existing Matplotlib dependency.
    """
    from matplotlib.font_manager import weight_dict
    from .fonts import match_font
    try:
        match = match_font(family, weight=weight, style=style)
        path = match['path']
    except ValueError as exc:
        raise ValueError(f'forest font {family!r} is unavailable; install it or explicitly select an installed family') from exc
    if Path(path).suffix.lower() != '.ttc':
        return path
    from fontTools.ttLib import TTCollection
    collection = TTCollection(path)
    target = weight_dict.get(weight, weight)
    requested_italic = style != 'normal'
    candidates = []
    for index, face in enumerate(collection.fonts):
        names = {n.toUnicode().lower() for n in face['name'].names if n.nameID in (1, 16)}
        if match['family'].lower() not in names:
            continue
        italic = bool(face['head'].macStyle & 2)
        candidates.append((italic != requested_italic, abs(face['OS/2'].usWeightClass-target), index))
    if not candidates:
        collection.close()
        raise ValueError(f'forest font {family!r}: no matching face in {path}')
    index = min(candidates)[2]
    global _font_directory
    if _font_directory is None:
        _font_directory = tempfile.TemporaryDirectory(prefix='jarvisplot-font-')
    output = Path(_font_directory.name)/f'{len(list(Path(_font_directory.name).iterdir()))}-{weight}.ttf'
    selected = collection.fonts[index]
    # Some installed macOS Helvetica hint programs fail in FreeType at small
    # sizes (including 4 pt logo text). The temporary standalone face retains
    # its original outlines/advances but does not carry those native programs.
    # This also keeps direct external canvas.draw safe without global rc state.
    if 'glyf' in selected:
        selected['glyf'].removeHinting()
    for table in ('fpgm', 'prep', 'cvt ', 'hdmx', 'VDMX', 'LTSH', 'cvar'):
        if table in selected:
            del selected[table]
    selected.save(output)
    collection.close()
    return str(output)


def font_properties(size_pt, family, weight='normal', *, style='normal'):
    from matplotlib.font_manager import FontProperties
    return FontProperties(fname=font_path(family, weight, style), family=family, size=size_pt, weight=weight, style=style)


def text_metrics_mm(text, size_pt, family, weight, math=False, *, prop=None):
    """Return actual literal or opt-in MathText advance/overhang, in mm."""
    # MathText font selection can change between figures as other style cards
    # update rcParams. A metric cached under the prior font configuration must
    # never determine the physical slots of a later figure.
    signature = ()
    if math:
        from matplotlib import rcParams
        signature = tuple((name, rcParams.get('mathtext.'+name)) for name in
                          ('fontset', 'default', 'fallback', 'rm', 'it', 'bf', 'bfit', 'cal', 'sf', 'tt'))
    # Legend font properties may choose mathematical glyphs independently of
    # rcParams, or override the ordinary text family/weight/style. Cache their
    # immutable values instead of retaining a mutable FontProperties object.
    explicit = None if prop is None else (
        tuple(prop.get_family()), prop.get_style(), prop.get_variant(),
        prop.get_weight(), prop.get_stretch(), prop.get_size_in_points(),
        prop.get_file(), prop.get_math_fontfamily(),
    )
    return _text_metrics_mm_cached(text, size_pt, family, weight, math, signature, explicit)


@lru_cache(maxsize=4096)
def _text_metrics_mm_cached(text, size_pt, family, weight, math, math_signature, explicit):
    """Cache against both explicit font properties and active MathText fonts."""
    from matplotlib.textpath import TextToPath
    from matplotlib.path import Path
    from .forest_contract import MM_PER_PT
    if '\n' in text or '\r' in text or '\t' in text:
        raise ValueError(f'forest column text must be a single line without tabs: {text!r}')
    if explicit is None:
        prop = font_properties(size_pt, family, weight)
    else:
        from matplotlib.font_manager import FontProperties
        prop = FontProperties(family=explicit[0], style=explicit[1], variant=explicit[2],
                              weight=explicit[3], stretch=explicit[4], size=explicit[5],
                              fname=explicit[6], math_fontfamily=explicit[7])
    converter = TextToPath()
    from matplotlib.cbook import is_math_text
    ismath = bool(math and is_math_text(text))
    advance, _, _ = converter.get_text_width_height_descent(text, prop, ismath=ismath)
    vertices, codes = converter.get_text_path(prop, text, ismath=ismath)
    left = right = 0.0
    if len(vertices):
        bounds = Path(vertices, codes).get_extents()
        scale = prop.get_size_in_points()/100
        left = max(0, -bounds.x0*scale)
        right = max(0, bounds.x1*scale-advance)
    return advance*MM_PER_PT, left*MM_PER_PT, right*MM_PER_PT


def wrap_text(text, size_pt, family, weight, width_mm, math=False, *, prop=None):
    """Wrap copy at spaces, retaining blank lines and whole math expressions.

    MathText is explicit: dollar-delimited expressions are indivisible atoms,
    including their interior spaces. An oversized atom is kept intact so the
    native artist-bound check can report the required physical width.
    """
    result = []
    for paragraph in text.splitlines():
        current = ''
        words = paragraph.split()
        if math:
            # Split around complete formulas, then tokenize only ordinary text.
            # Prefix/suffix punctuation stays attached and unmatched or escaped
            # dollar signs remain literal rather than silently disappearing.
            chunks = re.split(r'((?<!\\)\$(?:\\.|[^$])*(?<!\\)\$)', paragraph)
            words, atom = [], ''
            for index, chunk in enumerate(chunks):
                if index % 2:
                    atom += chunk
                    continue
                for piece in re.split(r'(\s+)', chunk):
                    if piece.isspace():
                        if atom:
                            words.append(atom)
                            atom = ''
                    else:
                        atom += piece
            if atom:
                words.append(atom)
        for word in words:
            candidate = f'{current} {word}' if current else word
            advance, left, right = text_metrics_mm(candidate, size_pt, family, weight, math=math, prop=prop)
            if current and advance+left+right > width_mm:
                result.append(current)
                current = word
            else:
                current = candidate
        result.append(current)
    return result
