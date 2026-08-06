# Specs

This directory defines the stable input and contract documents for Jarvis-PLOT.

## Purpose

Use these docs when you need to know what a valid scene, style, profile, or layer-type contract should look like.

## Current Files

- `SCENE_JSON_SCHEMA.md`: partial semantic scene contract; flowchart subset is implemented in `jarvisplot/flowchart.py`
- `STYLE_SCHEMA.md`: partial style contract (cards exist; formal schema ownership is incomplete)
- `DYNESTY_RUNPLOT.md`: implemented reusable dynesty runplot format
- `TRANSFORMS.md`: implemented transform contract and runtime scope
- `PROFILE_SCHEMA.md`: spec-only profile contract
- `INTERP_2D.md`: implemented 2D support-to-grid interpolation transform
- `LAYER_TYPE_REGISTRY.md`: spec-only layer/method registry contract
- `POSTERIOR_DENSITY.md`: implemented posterior-density reconstruction transforms and HPD contour contract
- `AGENT_DATA_API.md`: spec-only Jarvis-Agent bridge — JSON verbs, headless analysis channel (`likelihood_report` cell/region digests), template slot schemas, figure numeric twins

## Reading Order

1. `SCENE_JSON_SCHEMA.md`
2. the schema doc relevant to the field or contract you are touching
