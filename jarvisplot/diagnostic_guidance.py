#!/usr/bin/env python3

"""Actionable guidance for every ``JP-*`` diagnostic.

Mirrors Jarvis-HEP v2 ``diagnostic_guidance.guidance_for``: parameter-level
codes win over path-prefix rules, and every code still gets a non-empty
suggestion even when no specialist rule matches.
"""

from __future__ import annotations

import re
from typing import Optional

__all__ = ["KNOWN_CODES", "guidance_for"]


_EX_ROOT = """version: "0.3"
DataSet:
  - {name: df, path: ./samples.csv, type: csv}
Figures:
  - name: f1
    style: [a4paper_2x1, rect]
    layers:
      - name: pts
        data: [{source: df}]
        axes: ax
        method: scatter
        coordinates:
          x: {expr: m_A}
          y: {expr: tanb}
"""

_EX_SCATTER = """method: scatter
coordinates:
  x: {expr: m_A}
  y: {expr: tanb}
"""

_EX_PCOLOR = """method: pcolormesh
coordinates:
  x: {expr: x}
  y: {expr: y}
  z: {expr: density}
"""

# code -> (suggestion, example | None)
_GUIDANCE_BY_CODE: dict[str, tuple[str, Optional[str]]] = {
    # YAML load / root
    "JP-YML-000": (
        "Check the path; jplot resolves it relative to the current directory.",
        "jplot validate path/to/config.yaml",
    ),
    "JP-YML-001": (
        "Fix the YAML syntax first; nothing else can be checked until the file parses.",
        None,
    ),
    "JP-YML-002": (
        "A config needs at least DataSet and Figures.",
        _EX_ROOT,
    ),
    "JP-YML-003": (
        "The top level of a jplot config is key: value pairs, not a list or scalar.",
        _EX_ROOT,
    ),
    # Schema
    "JP-SCH-001": (
        "Rename the key to the nearest allowed spelling, or remove it.",
        None,
    ),
    "JP-SCH-002": (
        "Add the required key at this location using the expected YAML type.",
        None,
    ),
    "JP-SCH-003": (
        "Replace the value with one of the allowed values listed in the error; spelling and case matter.",
        None,
    ),
    "JP-SCH-004": (
        "Correct the value type (mapping / list / string / number / boolean) as stated.",
        None,
    ),
    "JP-SCH-005": (
        "Rewrite the value so it matches one of the accepted shapes for this field.",
        None,
    ),
    "JP-SCH-009": (
        "Correct the value so it satisfies the constraint stated in the message.",
        None,
    ),
    # Data / figures / refs
    "JP-DAT-004": (
        "Paths resolve relative to the config file's directory "
        "(or use the &JP/ prefix for package-relative paths).",
        "path: ./DATABASE/samples.csv",
    ),
    "JP-DAT-005": (
        "Give every DataSet a unique name; layers address data by that name.",
        "DataSet:\n  - {name: df, path: ./samples.csv, type: csv}",
    ),
    "JP-FIG-003": (
        "Give every Figure a unique name; it becomes the output filename.",
        "Figures:\n  - name: posterior\n    layers: []",
    ),
    "JP-REF-001": (
        "Set data[].source to a declared DataSet.name or an earlier share_data name. "
        "Run jplot data describe on the file if you meant a column, not a source.",
        "data:\n  - source: df",
    ),
    # Columns
    "JP-COL-001": (
        "Use a column that exists in the dataset (see available_columns / did_you_mean), "
        "or run jplot data describe on the file.",
        "coordinates:\n  x: {expr: m_A}",
    ),
    "JP-COL-900": (
        "Column probe could not read the file header; check the path and type.",
        None,
    ),
    # Ownership
    "JP-OWN-001": (
        "Move legend under frame.<axes>.legend; a figure-level legend key is ignored at render time.",
        "frame:\n  ax:\n    legend: {loc: best}",
    ),
    # Expressions (data eval + future validate)
    "JP-EXP-000": (
        "Pass a non-empty expression, e.g. exp(LogL).",
        'jplot data eval "exp(LogL)" --data samples.csv --json',
    ),
    "JP-EXP-001": (
        "Check operators and parentheses; run jplot cap funcs for callables.",
        'jplot data eval "np.log10(m_A)" --data samples.csv --json',
    ),
    "JP-EXP-002": (
        "Use a column from available_columns (or did_you_mean), "
        "or a function from jplot cap funcs.",
        'jplot data eval "exp(LogL)" --data samples.csv --json',
    ),
    # Methods / coordinates
    "JP-MTH-001": (
        "Set layers[].method to a name from jplot cap methods.",
        _EX_SCATTER,
    ),
    "JP-MTH-002": (
        "Add the missing coordinates axis required by this method "
        "(see context.required / optional, or jplot cap methods).",
        _EX_SCATTER,
    ),
    "JP-MTH-003": (
        "This method is not valid for the axes family of the chosen style card; "
        "pick a rect- or tri-compatible method (jplot cap methods / styles).",
        _EX_SCATTER,
    ),
}

# Path-prefix rules (longer / more specific first). Parameter codes above win.
_GUIDANCE_BY_PATH_PREFIX: tuple[tuple[str, str, Optional[str]], ...] = (
    (
        "$.Figures",
        "Fix the figure block; each figure needs a unique name and a layers list.",
        _EX_ROOT,
    ),
    (
        "$.DataSet",
        "Fix the DataSet entry: name, path, and type (csv|hdf5|parquet) are required.",
        "DataSet:\n  - {name: df, path: ./samples.csv, type: csv}",
    ),
    (
        "$.output",
        "output is a mapping; use dir for the figure output directory.",
        "output: {dir: ./plots}",
    ),
)


def guidance_for(code: str, path: str, message: str) -> tuple[str, Optional[str]]:
    """Return ``(suggestion, example)`` for any diagnostic.

    Parameter-level codes in ``_GUIDANCE_BY_CODE`` always win over path-prefix
    rules (HEP D21.14 lesson).
    """
    code = str(code or "").strip()
    path = str(path or "$")
    message = str(message or "")

    if code in _GUIDANCE_BY_CODE:
        return _GUIDANCE_BY_CODE[code]

    # Prefix family fallbacks (SCH/COL/…) when a new code is added without a row.
    if code.startswith("JP-SCH-"):
        if "unknown key" in message.lower() or "unexpected" in message.lower():
            return (
                "Rename or remove the key; only closed-vocabulary keys are allowed here. "
                "See allowed keys in the diagnostic context.",
                None,
            )
        if "required" in message.lower() or "missing" in message.lower():
            return ("Add the required field at this location.", None)
        return ("Correct the value so it satisfies the schema constraint stated above.", None)
    if code.startswith("JP-COL-"):
        return (
            "Use a real column from the dataset (jplot data describe) "
            "or fix the expression symbols.",
            None,
        )
    if code.startswith("JP-MTH-"):
        return (
            "Check method and coordinates against jplot cap methods; "
            "required axes are listed per method.",
            _EX_SCATTER,
        )
    if code.startswith("JP-EXP-"):
        return (
            "Validate the expression with jplot data eval before writing it into YAML.",
            None,
        )
    if code.startswith("JP-OWN-"):
        return (
            "Move the setting to the owner path named in the message "
            "(frame / layer / style card).",
            None,
        )
    if code.startswith("JP-TRF-"):
        return (
            "Use a transform step name from jplot cap transforms; "
            "unknown steps are rejected by the closed vocabulary.",
            "transform:\n  - filter: \"LogL > -100\"",
        )
    if code.startswith("JP-VIZ-"):
        return (
            "Inspect the render health report; adjust limits, color scales, "
            "or filters as indicated.",
            None,
        )

    for prefix, suggestion, example in _GUIDANCE_BY_PATH_PREFIX:
        if path == prefix or path.startswith(prefix + ".") or path.startswith(prefix + "["):
            return suggestion, example

    if "unknown key" in message or "unexpected" in message:
        return (
            "Remove the misspelled key or rename it to one of the allowed keys listed.",
            None,
        )
    if "expected one of" in message or "not a valid value" in message:
        return (
            "Replace the value with one of the allowed values; spelling and case matter.",
            None,
        )
    if "expected a mapping" in message or "expected object" in message:
        return (
            "Replace this scalar/list with an indented YAML mapping (key: value pairs).",
            None,
        )
    if "expected a list" in message or "expected array" in message:
        return (
            "Replace this value with an indented YAML list whose items start with '- '.",
            None,
        )
    if "required" in message or "missing" in message:
        fields = re.findall(r"'([^']+)'", message)
        suffix = f" Add {fields[0]!r}." if fields else ""
        return f"Add the required field at this location.{suffix}", None

    return (
        "Correct the value so it satisfies the constraint stated in the error message.",
        None,
    )


KNOWN_CODES: frozenset[str] = frozenset(_GUIDANCE_BY_CODE)
