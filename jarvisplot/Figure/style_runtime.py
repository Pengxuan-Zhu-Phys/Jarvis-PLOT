#!/usr/bin/env python3
from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Tuple


PREFERRED_STYLE_VARIANTS = ("rect", "Ternary", "rect_5x1", "dynesty_runplot", "rectcmap", "TernaryCmap")


def normalize_style_tokens(value):
    """``style: a4paper_2x1`` and ``style: [a4paper_2x1]`` name the same card.

    The schema accepts both spellings; everything downstream indexes tokens,
    and a bare string would be read one character at a time.
    """
    if isinstance(value, str):
        return [value]
    if isinstance(value, tuple):
        return list(value)
    return value


def resolve_style_bundle_payload(jpstyles: Mapping[str, Any], value) -> Tuple[str, str, dict]:
    """Resolve style tokens to a full bundle payload."""
    value = normalize_style_tokens(value)
    if len(value) == 2:
        family_name = value[0]
        variant_name = value[1]
        family = jpstyles[family_name]
        bundle = family[variant_name]
    elif len(value) == 1:
        family_name = value[0]
        family = jpstyles.get(family_name, {})
        variant_name = next((key for key in PREFERRED_STYLE_VARIANTS if key in family), None)
        if variant_name is None and family:
            variant_name = next(iter(family.keys()))
        if variant_name is None:
            raise KeyError(f"Style family '{family_name}' has no usable variant")
        bundle = family[variant_name]
    else:
        raise TypeError("Style tokens must contain one family or family+variant")

    return family_name, variant_name, deepcopy(bundle)


def resolve_style_bundle(jpstyles: Mapping[str, Any], value) -> Tuple[str, str, dict, dict]:
    """
    Resolve a style token list into concrete Frame/Style payloads.

    Returns:
        (family_name, variant_name, frame_dict, style_dict)
    """
    family_name, variant_name, bundle = resolve_style_bundle_payload(jpstyles, value)

    return (
        family_name,
        variant_name,
        deepcopy(bundle["Frame"]),
        deepcopy(bundle["Style"]),
    )
