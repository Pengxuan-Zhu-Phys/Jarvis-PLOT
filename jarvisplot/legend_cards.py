"""Data-only legend card loading and YAML resolution, shared with validation."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from .utils.pathing import resolve_project_path


CARDS_DIR = Path(__file__).with_name("cards") / "legends"
CARD_SCHEMA = "https://jarvis-plot.org/schema/v2/core/legend-card.json"


def merge_legend_settings(base, override):
    """Merge mappings, replacing lists/scalars; never mutate either input."""
    if not isinstance(base, dict) or not isinstance(override, dict):
        return deepcopy(override)
    out = deepcopy(base)
    for key, value in override.items():
        out[key] = merge_legend_settings(out.get(key), value)
    return out


def _validate_card(card, location, *, resolved=False):
    from jsonschema import Draft202012Validator
    from .schema_catalog import subschema

    errors = list(Draft202012Validator(subschema(CARD_SCHEMA)).iter_errors(card))
    if errors:
        error = errors[0]
        field = ".".join(map(str, error.absolute_path)) or "$"
        raise ValueError(f"Invalid legend card {location} at {field}: {error.message}")
    if resolved:
        validator = Draft202012Validator(subschema(CARD_SCHEMA)["$defs"]["itemFormat"])
        for name, item in card.get("items", {}).items():
            errors = list(validator.iter_errors(item))
            if errors:
                raise ValueError(f"Invalid legend item {name!r} in {location}: {errors[0].message}")


def load_legend_card(reference="default", *, base_dir=None, _seen=()):
    """Load a built-in name or a JSON path, with local-only inheritance."""
    if not isinstance(reference, str) or not reference.strip():
        raise ValueError("legend.card must be a non-empty name or JSON path")
    if reference.endswith(".json"):
        path = resolve_project_path(reference, base_dir)
    else:
        # A name selects only an installed card; paths must spell .json.
        path = CARDS_DIR / reference / "legend-card.json"
        if reference not in builtin_card_names():
            raise ValueError(
                f"Unknown legend card {reference!r}; choose {', '.join(builtin_card_names())} "
                "or a local .json path"
            )
    path = path.resolve()
    if path in _seen:
        raise ValueError(f"Legend card inheritance cycle at {path}")
    try:
        card = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot load legend card {path}: {exc}") from exc
    _validate_card(card, path)
    parent = card.get("extends")
    resolved = load_legend_card(parent, base_dir=path.parent, _seen=(*_seen, path)) if parent else {}
    resolved = merge_legend_settings(resolved, card)
    resolved.pop("extends", None)
    _validate_card(resolved, path, resolved=True)
    return resolved


def builtin_card_names():
    return sorted(path.parent.name for path in CARDS_DIR.glob("*/legend-card.json"))


def legend_card_catalog():
    """Resolved live cards and item contracts for man and style discovery."""
    return [
        {"name": name, "path": f"cards/legends/{name}/legend-card.json", **load_legend_card(name)}
        for name in builtin_card_names()
    ]


class LegendConfigError(ValueError):
    """Legend configuration violates the single declaration contract."""


def resolve_legend_config(config, *, base_dir=None):
    """Return native layout kwargs and resolved item templates."""
    if isinstance(config, bool):
        config = {"enabled": config}
    if not isinstance(config, dict):
        raise LegendConfigError("legend must be a boolean or mapping")
    retired = set(config) & {"entries", "preset", "handles", "labels", "axes"}
    if retired:
        raise LegendConfigError(
            f"Unsupported axes legend options {sorted(retired)}. "
            "Declare object/label/role on layers; use card for format selection "
            "and order/ncols/loc for overall layout.")
    if "enabled" in config and not isinstance(config["enabled"], bool):
        raise LegendConfigError("legend.enabled must be a boolean")
    if "order" in config and not isinstance(config["order"], list):
        raise LegendConfigError("legend.order must be a list of object names")
    override = deepcopy(config)
    override.pop("enabled", None)
    override.pop("order", None)
    card = load_legend_card(override.pop("card", "default"), base_dir=base_dir)
    item_overrides = override.pop("item_formats", {})
    if not isinstance(item_overrides, dict):
        raise ValueError("legend.item_formats must be a mapping")
    items = merge_legend_settings(card.get("items", {}), item_overrides)
    _validate_card({"schema_version": 1, "legend": {}, "items": items}, "YAML item_formats", resolved=True)
    return merge_legend_settings(card.get("legend", {}), override), items


class LayerLegendError(ValueError):
    """A declaration error with the index of the responsible axes-local layer."""

    def __init__(self, message, layer_index=None):
        super().__init__(message)
        self.layer_index = layer_index


def has_layer_legend(layers):
    return any(isinstance(layer, dict) and "legend" in layer for layer in layers)


def effective_axes_legend(config, layers, *, explicitly_disabled=False):
    """Layer declarations activate inherited disabled defaults, but respect YAML off."""
    if (not isinstance(config, (dict, bool))
            or not has_layer_legend(layers) or explicitly_disabled):
        return deepcopy(config)
    value = deepcopy(config) if isinstance(config, dict) else {}
    value["enabled"] = True
    return value


def assemble_legend_entries(layers, formats, *, order=None):
    """Group layer declarations by object and match card templates by role set.

    References use axes-local indices so unnamed layers are supported and object
    names remain independent of layer names. Never modify the source YAML.
    """
    groups = {}
    for index, layer in enumerate(layers):
        if not isinstance(layer, dict) or "legend" not in layer:
            continue
        declaration = layer["legend"]
        if not isinstance(declaration, dict) or set(declaration) - {"object", "label", "role"}:
            raise LayerLegendError("layer.legend accepts only object, label and optional role", index)
        for field in ("object", "label"):
            if not isinstance(declaration.get(field), str) or not declaration[field].strip():
                raise LayerLegendError(f"layer.legend.{field} must be a non-empty string", index)
        role = declaration.get("role")
        if "role" in declaration and (not isinstance(role, str) or not role.strip()):
            raise LayerLegendError("layer.legend.role must be a non-empty string", index)
        name, label = declaration["object"], declaration["label"]
        group = groups.setdefault(name, {"label": label, "parts": [], "index": index})
        if label != group["label"]:
            raise LayerLegendError(f"Legend object {name!r} has conflicting labels {group['label']!r} and {label!r}", index)
        if role is not None and any(part[0] == role for part in group["parts"]):
            raise LayerLegendError(f"Legend object {name!r} repeats role {role!r}", index)
        group["parts"].append((role, f"@layer:{index}"))

    if order is not None:
        if (not isinstance(order, list) or not all(isinstance(n, str) and n for n in order)
                or len(set(order)) != len(order)):
            raise LayerLegendError("legend.order must be a list of unique object names")
        unknown = set(order) - set(groups)
        if unknown:
            raise LayerLegendError(f"legend.order references unknown objects {sorted(unknown)}")
        names = [*order, *(name for name in groups if name not in order)]
    else:
        names = list(groups)

    entries = []
    for name in names:
        group = groups[name]
        parts = group["parts"]
        entry = {"object": name, "label": group["label"], "key": "overlay"}
        if len(parts) == 1 and parts[0][0] is None:
            # A single real artist/container already carries its native structure.
            entry["layers"] = [parts[0][1]]
        else:
            if any(role is None for role, _ in parts):
                raise LayerLegendError(f"Legend object {name!r} has multiple components; every layer needs role", group["index"])
            roles = {role for role, _ in parts}
            matches = [(fmt, item) for fmt, item in formats.items() if set(item["roles"]) == roles]
            if not matches:
                raise LayerLegendError(
                    f"Legend object {name!r} has roles {sorted(roles)}; no card item matches exactly. "
                    "Supply all required roles or define item_formats for this role set.", group["index"])
            layouts = {(tuple(item["roles"]), item["key"]) for _, item in matches}
            if len(layouts) > 1:
                raise LayerLegendError(
                    f"Legend object {name!r} matches conflicting item formats {[fmt for fmt, _ in matches]}; "
                    "give them distinct role sets or the same roles order and key", group["index"])
            item = matches[0][1]
            by_role = dict(parts)
            entry["layers"] = [by_role[role] for role in item["roles"]]
            entry["key"] = item["key"]
        entries.append(entry)
    return entries


def resolve_axes_legend(config, layers, *, base_dir=None):
    """Compile the sole layer declaration path, shared by validation/rendering."""
    kwargs, formats = resolve_legend_config(config, base_dir=base_dir)
    settings = config if isinstance(config, dict) else {"enabled": config}
    entries = assemble_legend_entries(layers, formats, order=settings.get("order"))
    if settings.get("enabled", True) and not entries:
        raise LegendConfigError(
            "Legend is enabled but no layer declares legend.object/label. "
            "Declare layer.legend objects or set the axes legend to false; "
            "style.label and data[].label do not create legend items.")
    return kwargs, entries


def merge_axes_legend(base, override):
    """A YAML declaration enables a card's disabled default unless it says otherwise."""
    if isinstance(override, bool):
        return {**(deepcopy(base) if isinstance(base, dict) else {}), "enabled": override}
    if isinstance(override, dict):
        value = merge_legend_settings(base if isinstance(base, dict) else {}, override)
        value["enabled"] = override.get("enabled", True)
        return value
    return deepcopy(override)


def style_legend_defaults(style=None):
    """Read axes legend declarations from a style bundle without rendering."""
    from .Figure.style_runtime import resolve_style_bundle_payload

    style = style or ["a4paper_2x1"]
    if (not isinstance(style, list) or len(style) not in (1, 2)
            or not all(isinstance(token, str) for token in style)):
        return {}
    preference = json.loads((CARDS_DIR.parent / "style_preference.json").read_text(encoding="utf-8"))
    family = preference.get(style[0], {})
    bundles = {}
    for variant, reference in family.items():
        path = resolve_project_path(reference)
        if path.is_file():
            bundles[variant] = json.loads(path.read_text(encoding="utf-8"))
    try:
        _, _, bundle = resolve_style_bundle_payload({style[0]: bundles}, style)
    except (KeyError, TypeError):
        return {}
    return {name: node["legend"] for name, node in bundle.get("Frame", {}).items()
            if isinstance(node, dict) and "legend" in node}
