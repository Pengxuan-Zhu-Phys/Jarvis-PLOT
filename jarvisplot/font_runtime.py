"""Font-card defaults at the style/render boundary, without data side effects."""
from __future__ import annotations

from copy import deepcopy


def text_settings(role, card):
    from .fonts import match_font, resolve_role
    settings = resolve_role(role, card=card)
    family = match_font(settings['family'], settings['weight'], settings['style'])['family']
    result = {'fontfamily': family, 'fontweight': settings['weight'],
              'fontstyle': settings['style'], 'math_fontfamily': settings['math_fontfamily']}
    if settings['size_pt'] is not None:
        result['fontsize'] = settings['size_pt']
    return result


def rc_font_settings(card):
    """Use a scoped creation/render default, never persist global rc changes."""
    from .fonts import match_font, resolve_role
    text = resolve_role('text', card=card)
    family = match_font(text['family'], text['weight'], text['style'])['family']
    result = {'font.family': [family], 'font.weight': text['weight'],
              'font.style': text['style'], 'mathtext.fontset': card['math']['fontset']}
    if text['size_pt'] is not None:
        result['font.size'] = text['size_pt']
    return result


_FONT_ALIASES = (('fontfamily', 'family'), ('fontsize', 'size'),
                 ('fontweight', 'weight'), ('fontstyle', 'style'))


def prepare_font_overrides(base, override):
    """Remove inherited aliases when a local Text setting owns that property."""
    if not isinstance(base, dict) or not isinstance(override, dict):
        return
    if 'fontproperties' in override:
        for group in _FONT_ALIASES:
            for key in group:
                base.pop(key, None)
        base.pop('math_fontfamily', None)
    for group in _FONT_ALIASES:
        if any(key in override and not (key == 'style' and isinstance(override[key], dict)) for key in group):
            for key in group:
                base.pop(key, None)
    for key, value in override.items():
        if isinstance(value, dict):
            prepare_font_overrides(base.get(key), value)
    if 'fontsize' in override and isinstance(base.get('prop'), dict):
        explicit_prop = override.get('prop')
        if not isinstance(explicit_prop, dict) or 'size' not in explicit_prop:
            base['prop'].pop('size', None)


def merge_text_settings(base, override):
    result = deepcopy(base)
    prepare_font_overrides(result, override)
    result.update(override or {})
    return result


def prepare_frame_fonts(base, override, card):
    """Default new YAML axes and appended text, while local properties win."""
    result = deepcopy(override)
    names = {name for name, value in result.items()
             if name not in {'figure', 'axes', 'axlogo'} and isinstance(value, dict)}
    names.update(name for name in result.get('axes', {}) if name != 'axlogo')
    missing = names - base.keys()
    if missing:
        defaults = {'axes': {name: {} for name in missing}}
        apply_font_defaults(defaults, {}, card)
        for name in missing:
            base[name] = defaults[name]
    for name, value in result.items():
        if isinstance(value, dict) and isinstance(value.get('text'), list):
            role = 'caption' if name == 'axlogo' else 'text'
            value['text'] = [merge_text_settings(text_settings(role, card), entry)
                             for entry in value['text']]
    return result


def apply_font_defaults(frame, style, card):
    """Overlay font attributes on a copied style bundle before explicit YAML."""
    for name in frame.get('axes', {}):
        if name != 'axlogo':
            frame.setdefault(name, {})
    def apply(node, role):
        defaults = text_settings(role, card)
        if 'fontproperties' in node:
            from matplotlib.font_manager import FontProperties
            prop = FontProperties._from_any(node.pop('fontproperties'))
            if 'fontsize' not in defaults and not {'fontsize', 'size'} & node.keys():
                defaults['fontsize'] = prop.get_size_in_points()
        prepare_font_overrides(node, defaults)
        node.update(defaults)
    for name, node in frame.items():
        if name in {'figure', 'axes'} or not isinstance(node, dict):
            continue
        if name == 'axlogo':
            for entry in node.get('text', []):
                apply(entry, 'caption')
            continue
        if 'title_params' in node or name in frame.get('axes', {}):
            apply(node.setdefault('title_params', {}), 'title')
        colorbar = name.startswith('axc') and name != 'axcorr'
        if name != 'axcorr' and not colorbar:
            labels = node.get('labels', {})
            for key in ('xlabel', 'ylabel', 'leftstyle', 'rightstyle', 'bottomstyle'):
                if key in labels or (key in {'xlabel', 'ylabel'} and name != 'axtri'):
                    apply(labels.setdefault(key, {}), 'axis_label')
            if labels or name != 'axtri':
                node['labels'] = labels
        if name.startswith('axc') and name != 'axcorr':
            apply(node.setdefault('label', {}), 'axis_label')
            apply(node.setdefault('offset_text', {}), 'tick')
        ticks = node.setdefault('ticks', {})
        if name == 'axtri':
            for key in ('bottomticklables', 'leftticklables', 'rightticklables'):
                apply(ticks.setdefault(key, {}), 'tick')
        else:
            from .fonts import match_font, resolve_role
            tick = resolve_role('tick', card=card)
            ticks.setdefault('both', {})['labelfontfamily'] = match_font(tick['family'], tick['weight'], tick['style'])['family']
            if tick['size_pt'] is not None:
                # Existing major/minor values otherwise override the card.
                for key in ('both', 'major', 'minor'):
                    ticks.setdefault(key, {})['labelsize'] = tick['size_pt']
        for entry in node.get('text', []):
            apply(entry, 'text')
        legend = None if name == 'axcorr' else node.setdefault('legend', {'enabled': False})
        if isinstance(legend, dict):
            legend.update(legend_defaults(legend, card))
    corrplot = style.get('corrplot')
    if isinstance(corrplot, dict):
        corrplot['__corr_fonts__'] = {role: text_settings(role, card) for role in ('tick', 'annotation')}
    dynesty = style.get('dynesty_runplot')
    if isinstance(dynesty, dict):
        apply(dynesty.setdefault('evidence_summary_text_kwargs', {}), 'annotation')


def legend_defaults(config, card):
    """Only font kwargs; enabling and layout remain owned by the legend card."""
    from .fonts import match_font, resolve_role
    role = resolve_role('legend', card=card)
    family = match_font(role['family'], role['weight'], role['style'])['family']
    prop = {'family': family, 'weight': role['weight'], 'style': role['style'],
            'math_fontfamily': role['math_fontfamily']}
    if role['size_pt'] is not None:
        prop['size'] = role['size_pt']
    existing = config.get('prop') if isinstance(config, dict) else None
    if isinstance(existing, dict):
        prop = {**existing, **prop}
    elif existing is not None and role['size_pt'] is None:
        from matplotlib.font_manager import FontProperties
        prop['size'] = FontProperties._from_any(existing).get_size_in_points()
    return {'prop': prop}


def pin_figure_fonts(figure, frame=None, card=None):
    """Give named text fonts their actual face, including non-first TTC faces."""
    from matplotlib.text import Text
    from .forest_text import font_path
    if frame is not None and card is not None:
        from .fonts import resolve_role
        tick = resolve_role('tick', card=card)
        for name, axes in ((name, axes) for name, axes in getattr(figure, '_jarvis_axes', {}).items()
                           if name != 'axlogo'):
            target = axes.ax if hasattr(axes, 'ax') else axes
            for axis in (target.xaxis, target.yaxis):
                for label in [*axis.get_ticklabels(minor=False), *axis.get_ticklabels(minor=True)]:
                    label.set_fontweight(tick['weight'])
                    label.set_fontstyle(tick['style'])
                    label.set_math_fontfamily(tick['math_fontfamily'])
                offset_config = (frame.get(name) or {}).get('offset_text') or {}
                offset = axis.offsetText
                if not {'fontweight', 'weight', 'fontproperties'} & offset_config.keys():
                    offset.set_fontweight(tick['weight'])
                if not {'fontstyle', 'style', 'fontproperties'} & offset_config.keys():
                    offset.set_fontstyle(tick['style'])
                if not {'math_fontfamily', 'fontproperties'} & offset_config.keys():
                    offset.set_math_fontfamily(tick['math_fontfamily'])
    for artist in figure.findobj(Text):
        if not artist.get_text():
            continue
        prop = artist.get_fontproperties().copy()
        families = prop.get_family()
        if prop.get_file() is None and len(families) == 1:
            prop.set_file(font_path(families[0], prop.get_weight(), prop.get_style()))
            artist.set_fontproperties(prop)


_FOREST_SIZES = {'title_pt': 'title', 'header_pt': 'axis_label', 'label_pt': 'text',
                 'detail_pt': 'text', 'stats_pt': 'annotation', 'tick_pt': 'tick',
                 'footer_pt': 'caption', 'meta_pt': 'caption'}


def forest_typography_defaults(typography, override, card):
    """Keep role data beside native size slots, before physical measurement."""
    from .fonts import match_font, resolve_role
    result = deepcopy(typography)
    text = resolve_role('text', card=card)
    result['family'] = match_font(text['family'], text['weight'], text['style'])['family']
    for key, role in _FOREST_SIZES.items():
        size = resolve_role(role, card=card)['size_pt']
        if size is not None:
            result[key] = size
    result.update(override)
    return result


def forest_role_data(typography, override, card):
    from .fonts import match_font, resolve_role
    roles = {role: resolve_role(role, card=card) for role in card['roles']}
    if 'family' in override:
        # Native forest has one explicit family selector for every text role.
        for settings in roles.values():
            settings['family'] = override['family']
    for settings in roles.values():
        settings['family'] = match_font(settings['family'], settings['weight'], settings['style'])['family']
    return roles


def forest_font_properties(typography, role, size, weight=None, *, style=None):
    from .forest_text import font_properties
    settings = typography.get('_font_roles', {}).get(role, {})
    prop = font_properties(size, settings.get('family', typography['family']),
                           weight if weight is not None else settings.get('weight', 'normal'),
                           style=style if style is not None else settings.get('style', 'normal'))
    if settings.get('math_fontfamily'):
        prop.set_math_fontfamily(settings['math_fontfamily'])
    return prop
