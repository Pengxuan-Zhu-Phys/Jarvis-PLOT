"""Portable physical-font regressions, independent of macOS Helvetica."""
import pytest
from matplotlib.font_manager import FontProperties
from matplotlib.path import Path
from matplotlib.textpath import TextToPath

from jarvisplot.forest_text import font_properties, text_metrics_mm


@pytest.mark.parametrize('math', [False, True])
def test_explicit_font_properties_size_is_authoritative_for_advance_and_overhang(math):
    prop = FontProperties(family='DejaVu Sans', size=13, style='italic', math_fontfamily='stix')
    text = r'$\mathit{j}$' if math else 'j'
    converter = TextToPath()
    advance = converter.get_text_width_height_descent(text, prop, ismath=math)[0]
    vertices, codes = converter.get_text_path(prop, text, ismath=math)
    bounds = Path(vertices, codes).get_extents()
    scale = prop.get_size_in_points()/100
    expected = (advance, max(0, -bounds.x0*scale), max(0, bounds.x1*scale-advance))
    # Deliberately conflicting legacy arguments cannot scale explicit ink at
    # 4 pt while measuring its advance at 13 pt.
    measured = text_metrics_mm(text, 4, 'DejaVu Sans', 'normal', math=math, prop=prop)
    assert measured == pytest.approx(tuple(value*25.4/72 for value in expected))
    assert measured[1] > 0


def test_portable_requested_italic_font_has_an_italic_face():
    from matplotlib.ft2font import FT2Font
    prop = font_properties(5, 'DejaVu Sans', style='italic')
    assert prop.get_style() == 'italic'
    assert 'Oblique' in FT2Font(prop.get_file()).style_name


def test_missing_requested_font_is_not_silently_substituted():
    with pytest.raises(ValueError, match='unavailable'):
        font_properties(5, 'Jarvis deliberately absent regression font')
