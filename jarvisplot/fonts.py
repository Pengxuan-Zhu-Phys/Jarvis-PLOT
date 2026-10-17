"""Font-card data and persistence; discovery imports Matplotlib only on demand.

The immutable packaged card recommends families. User files contain sparse
overrides, never copies of packaged defaults, so future recommendations remain
visible. Reading a card does not scan fonts or create a configuration directory.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
import os
from pathlib import Path

ROLES = ('title', 'axis_label', 'tick', 'legend', 'text', 'annotation', 'caption')
MATH_FONTSETS = ('stix', 'dejavusans')
WEIGHTS = ('ultralight', 'light', 'normal', 'regular', 'book', 'medium', 'roman',
           'semibold', 'demibold', 'demi', 'bold', 'heavy', 'extra bold', 'black')
STYLES = ('normal', 'italic', 'oblique')
ROLE_FIELDS = ('family', 'size_pt', 'weight', 'style')
HELVETICA_FALLBACKS = ('Helvetica', 'Arial', 'Liberation Sans', 'Nimbus Sans', 'DejaVu Sans')


class FontCardError(ValueError):
    """An invalid persisted font card."""


class FontCardUsageError(FontCardError):
    """An invalid command argument or editable address."""


def font_card_path() -> Path:
    home = Path(os.environ.get('JARVIS_HOME') or Path.home() / '.jarvis')
    return home.expanduser().resolve() / 'plot-fonts.json'


def _packaged_card() -> dict:
    path = Path(__file__).with_name('cards') / 'fonts' / 'default.json'
    card = json.loads(path.read_text(encoding='utf-8'))
    _validate(card)
    return card


def _merge(base: dict, override: dict) -> dict:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _address(path: str) -> list[str]:
    parts = str(path).split('.')
    if (len(parts) == 2 and parts[0] == 'families' and parts[1] in ('sans', 'serif')
            or parts == ['math', 'fontset']
            or len(parts) == 3 and parts[0] == 'roles' and parts[1] in ROLES
            and parts[2] in ROLE_FIELDS):
        return parts
    raise FontCardUsageError(f'Unknown editable font-card path {path!r}; use families.sans, '
                        'families.serif, math.fontset or roles.<role>.family/size_pt/weight/style')


def _validate_value(path: str, value) -> None:
    field = path.split('.')[-1]
    if field in ('sans', 'serif', 'family'):
        if not isinstance(value, str) or not value.strip() or value != value.strip():
            raise FontCardUsageError(f'{path}: expected a nonempty font family without surrounding whitespace')
    elif field == 'fontset':
        if value not in MATH_FONTSETS:
            raise FontCardUsageError(f'{path}: choose {", ".join(MATH_FONTSETS)}')
    elif field == 'size_pt':
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                  or not math.isfinite(value) or value <= 0):
            raise FontCardUsageError(f'{path}: expected a positive finite point size or null (inherit style size)')
    elif field == 'weight':
        if not (isinstance(value, str) and value in WEIGHTS or
                not isinstance(value, bool) and isinstance(value, (int, float))
                and math.isfinite(value) and 0 <= value <= 1000):
            raise FontCardUsageError(f'{path}: expected a font weight name or a number from 0 to 1000')
    elif field == 'style' and value not in STYLES:
        raise FontCardUsageError(f'{path}: choose {", ".join(STYLES)}')


def _validate(card: dict) -> None:
    if not isinstance(card, dict) or type(card.get('schema_version')) is not int or card['schema_version'] != 1:
        raise FontCardError('Font card must be an object with schema_version: 1')
    if set(card) != {'schema_version', 'families', 'math', 'roles'}:
        raise FontCardError('Font card fields must be schema_version, families, math and roles')
    for key, expected in (('families', {'sans', 'serif'}), ('math', {'fontset'}), ('roles', set(ROLES))):
        if not isinstance(card[key], dict) or set(card[key]) != expected:
            raise FontCardError(f'{key}: expected fields {sorted(expected)}')
    for key, value in card['families'].items():
        _validate_value(f'families.{key}', value)
    _validate_value('math.fontset', card['math']['fontset'])
    for role, settings in card['roles'].items():
        if not isinstance(settings, dict) or set(settings) != set(ROLE_FIELDS) | {'recommended_size_pt'}:
            raise FontCardError(f'roles.{role}: invalid role fields')
        for key in ROLE_FIELDS:
            _validate_value(f'roles.{role}.{key}', settings[key])
        _validate_value(f'roles.{role}.size_pt', settings['recommended_size_pt'])


def _read_overrides() -> dict:
    path = font_card_path()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except FileNotFoundError:
        # reset may have removed it after the existence check; defaults remain valid.
        return {}
    except json.JSONDecodeError as exc:
        raise FontCardError(f'Invalid font-card JSON in {path}: {exc}') from exc
    if (not isinstance(data, dict) or type(data.get('schema_version')) is not int
            or data['schema_version'] != 1 or not isinstance(data.get('overrides'), dict)):
        raise FontCardError(f'Invalid font overrides in {path}; expected schema_version: 1 and overrides object')
    if set(data) != {'schema_version', 'overrides'}:
        raise FontCardError(f'Unexpected font override fields in {path}')
    overrides = data['overrides']

    def walk(node, prefix=''):
        for key, value in node.items():
            address = f'{prefix}.{key}' if prefix else key
            if isinstance(value, dict):
                if address not in ('families', 'math', 'roles', *(f'roles.{role}' for role in ROLES)):
                    raise FontCardError(f'Unknown font override group {address!r}')
                walk(value, address)
            else:
                _address(address)
                _validate_value(address, value)
    try:
        walk(overrides)
    except FontCardUsageError as exc:
        raise FontCardError(f'Invalid font overrides in {path}: {exc}') from exc
    return overrides


def load_font_card(recommended: bool = False) -> dict:
    card = _packaged_card()
    if not recommended:
        card = _merge(card, _read_overrides())
        _validate(card)
    return card


def _family_name(family: str) -> str:
    return 'STIXGeneral' if family.casefold() in ('stix', 'stixgeneral') else family


def resolve_role(role: str, card: dict | None = None) -> dict:
    if role not in ROLES:
        raise FontCardUsageError(f'Unknown font role {role!r}; choose {", ".join(ROLES)}')
    card = load_font_card() if card is None else card
    result = deepcopy(card['roles'][role])
    result['family'] = _family_name(card['families'].get(result['family'], result['family']))
    result['math_fontfamily'] = card['math']['fontset']
    return result


def font_card_payload(role: str | None = None, recommended: bool = False) -> dict:
    card = load_font_card(recommended=recommended)
    return {
        'card': card, 'recommended': _packaged_card(),
        'roles': {name: resolve_role(name, card) for name in ([role] if role else ROLES)},
        'source': 'recommended' if recommended else 'effective',
        'user_path': str(font_card_path()), 'has_overrides': bool(_read_overrides()) if not recommended else False,
        'precedence': 'explicit YAML/layer > user font card > packaged font card > style card',
        'size_policy': 'null preserves the plot style size; numeric size_pt overrides it; recommended_size_pt is drafting advice',
        'missing_font_policy': 'Helvetica uses the first installed similar sans family; other missing families are errors',
        'helvetica_fallbacks': list(HELVETICA_FALLBACKS),
        'math_engine': 'MathText ($...$); external LaTeX font configuration is independent',
    }


def update_font_card(path: str, value) -> dict:
    parts = _address(path)
    _validate_value(path, value)
    from .cache_registry import FileLock, _write_json
    target = font_card_path()
    with FileLock(target.with_suffix('.lock')):
        overrides = _read_overrides()
        node = overrides
        for key in parts[:-1]:
            node = node.setdefault(key, {})
        node[parts[-1]] = value
        _validate(_merge(_packaged_card(), overrides))
        _write_json(target, {'schema_version': 1, 'overrides': overrides})
        payload = font_card_payload()
    return payload


def reset_font_card(path: str | None = None) -> dict:
    parts = _address(path) if path is not None else None
    target = font_card_path()
    # A fresh installation's reset is read-only, like show.
    if not target.exists():
        return font_card_payload()
    from .cache_registry import FileLock, _write_json
    with FileLock(target.with_suffix('.lock')):
        if parts is None:
            # Reset also repairs a malformed user override file.
            target.unlink(missing_ok=True)
        else:
            overrides = _read_overrides()
            node = overrides
            parents = []
            for key in parts[:-1]:
                parents.append((node, key))
                node = node.get(key, {})
            node.pop(parts[-1], None)
            for parent, key in reversed(parents):
                if not parent.get(key):
                    parent.pop(key, None)
            if overrides:
                _write_json(target, {'schema_version': 1, 'overrides': overrides})
            else:
                target.unlink(missing_ok=True)
        payload = font_card_payload()
    return payload


def font_inventory(query: str | None = None, refresh: bool = False) -> dict:
    import matplotlib
    from matplotlib import font_manager
    if refresh:
        # Public API: rescan installed + Matplotlib bundled fonts in memory.
        # Rebind the module's shortcut, which normally points at the old manager.
        font_manager.fontManager = font_manager.FontManager()
        font_manager.findfont = font_manager.fontManager.findfont
        from .forest_text import font_path
        font_path.cache_clear()
    rows = sorted({(font.name, str(Path(font.fname).resolve()), font.style, str(font.weight))
                   for font in font_manager.fontManager.ttflist
                   if query is None or query.casefold() in font.name.casefold()})
    return {'fonts': [{'family': name, 'path': path, 'style': style, 'weight': weight}
                      for name, path, style, weight in rows],
            'count': len(rows), 'refreshed': refresh, 'cache_dir': matplotlib.get_cachedir()}


def match_font(family: str, weight='normal', style: str = 'normal') -> dict:
    """Choose an installed file without extracting TTC faces (no recursion).

    Only Helvetica has a curated substitution policy. Arbitrary family typos
    still fail, rather than receiving Matplotlib's silent default fallback.
    """
    _validate_value('roles.text.family', family)
    _validate_value('roles.text.weight', weight)
    _validate_value('roles.text.style', style)
    from matplotlib import font_manager
    family = _family_name(family)
    requested = family
    candidates = HELVETICA_FALLBACKS if family.casefold() == 'helvetica' else (family,)
    path = None
    for candidate in candidates:
        try:
            path = font_manager.findfont(
                font_manager.FontProperties(family=[candidate], weight=weight, style=style),
                fallback_to_default=False)
            family = candidate
            break
        except ValueError:
            continue
    if path is None:
        raise ValueError(f'Font {family!r} is unavailable; install it or use '
                         '`jplot fonts list` and explicitly set an installed family')
    from matplotlib.ft2font import FT2Font
    face = FT2Font(path)
    actual = next((font.name for font in font_manager.fontManager.ttflist
                   if font.fname == path and font.name.casefold() == family.casefold()), face.family_name)
    fallback = family.casefold() != requested.casefold() and len(candidates) > 1
    return {'requested_family': requested, 'family': actual, 'weight': weight, 'style': style,
            'path': str(Path(path).resolve()), 'fallback': fallback,
            'fallback_reason': f'{requested} is unavailable; selected installed {actual}' if fallback else None,
            'fallback_candidates': list(candidates)}


def resolve_font(family: str, weight='normal', style: str = 'normal') -> dict:
    result = match_font(family, weight, style)
    path = result['path']
    from matplotlib.ft2font import FT2Font
    face = FT2Font(path)
    # TTC requires the matching face, rather than Matplotlib's first face.
    if Path(path).suffix.lower() == '.ttc':
        from .forest_text import font_path
        face = FT2Font(font_path(result['family'], weight, style))
    return {**result, 'family': face.family_name, 'face': face.style_name}


def check_fonts() -> dict:
    card = load_font_card()
    requests = {}
    for role in ROLES:
        settings = resolve_role(role, card)
        key = (settings['family'], settings['weight'], settings['style'])
        requests.setdefault(key, []).append(role)
    math_family = 'STIXGeneral' if card['math']['fontset'] == 'stix' else 'DejaVu Sans'
    requests.setdefault((math_family, 'normal', 'normal'), []).append('math')
    results = []
    for (family, weight, style), roles in requests.items():
        try:
            result = {**resolve_font(family, weight, style), 'available': True, 'roles': roles}
        except ValueError as exc:
            result = {'requested_family': family, 'weight': weight, 'style': style,
                      'available': False, 'roles': roles, 'error': str(exc)}
        results.append(result)
    return {'ok': all(row['available'] for row in results), 'results': results,
            'user_path': str(font_card_path()), 'math_fontset': card['math']['fontset']}
