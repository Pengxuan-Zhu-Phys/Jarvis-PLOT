"""Export errors are real failures; supported backends retain provenance."""
from types import SimpleNamespace

import matplotlib.pyplot as plt
from PIL import Image, PngImagePlugin
import pytest

from jarvisplot.Figure.figure import Figure, _write_png_metadata
from jarvisplot.version import get_version


def output(tmp_path, backend, formats):
    messages = []
    return SimpleNamespace(
        fmts=formats, dir=str(tmp_path), name='export', dpi=80, fig=backend,
        _output_metadata={}, logger=SimpleNamespace(
            warning=lambda message: messages.append(('warning', message)),
            info=lambda message: messages.append(('info', message))), messages=messages)


def test_png_svg_pdf_and_jpeg_save_with_bounded_metadata_fallback(tmp_path):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    holder = output(tmp_path, fig, ['png', 'svg', 'pdf', 'jpg'])
    try:
        Figure.savefig(holder)
    finally:
        plt.close(fig)
    assert all((tmp_path/f'export.{fmt}').stat().st_size > 100 for fmt in holder.fmts)
    with Image.open(tmp_path/'export.png') as image:
        assert image.info['Jarvis-PLOT version'] == get_version()
    assert f'Jarvis-PLOT version: {get_version()}' in (tmp_path/'export.svg').read_text()
    assert any(level == 'warning' and 'jpg backend' in message for level, message in holder.messages)
    assert len([level for level, _ in holder.messages if level == 'info']) == 4


@pytest.mark.parametrize('error', [ValueError('invalid MathText layout'),
                                   TypeError('invalid metadata value'), OSError('disk full')])
def test_unrelated_export_failures_are_never_retried_or_reported_success(tmp_path, error):
    calls = []
    def save(*args, **kwargs):
        calls.append(kwargs)
        raise error
    holder = output(tmp_path, SimpleNamespace(savefig=save), ['png'])
    with pytest.raises(type(error), match=str(error)):
        Figure.savefig(holder)
    assert len(calls) == 1
    assert holder.messages == []


def test_complete_png_is_not_reencoded(tmp_path, monkeypatch):
    path = tmp_path/'native.png'
    info = PngImagePlugin.PngInfo()
    info.add_text('Creator', 'Jarvis-PLOT')
    Image.new('RGB', (2, 2)).save(path, pnginfo=info)
    before = path.read_bytes()
    monkeypatch.setattr(Image.Image, 'save', lambda *args, **kwargs: pytest.fail('complete native PNG rewritten'))
    _write_png_metadata(str(path), {'Creator': 'Jarvis-PLOT'})
    assert path.read_bytes() == before


def test_library_export_without_injected_logger_still_succeeds(tmp_path):
    fig, _ = plt.subplots()
    holder = output(tmp_path, fig, ['svg'])
    holder.logger = None
    try:
        Figure.savefig(holder)
    finally:
        plt.close(fig)
    assert (tmp_path/'export.svg').is_file()


def test_png_patch_preserves_existing_text_and_reports_io_errors(tmp_path, monkeypatch):
    path = tmp_path/'missing-provenance.png'
    info = PngImagePlugin.PngInfo()
    info.add_text('Caller', 'supplied note')
    Image.new('RGB', (2, 2)).save(path, pnginfo=info, dpi=(150, 150))
    _write_png_metadata(str(path), {'Creator': 'Jarvis-PLOT'})
    with Image.open(path) as image:
        assert image.info['Caller'] == 'supplied note'
        assert image.info['Creator'] == 'Jarvis-PLOT'
        assert image.info['dpi'][0] == pytest.approx(150, abs=.1)
    monkeypatch.setattr(Image.Image, 'save', lambda *a, **k: (_ for _ in ()).throw(OSError('denied')))
    with pytest.raises(OSError, match='denied'):
        _write_png_metadata(str(path), {'Creator': 'changed'})
