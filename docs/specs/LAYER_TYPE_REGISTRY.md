# LAYER_TYPE_REGISTRY

Status: spec only

## Purpose

This document defines the intended layer-type contract for Jarvis-PLOT.

Current code does not have a separate layer-type registry.
The real dispatch path is method-based and lives in:

- `jarvisplot/Figure/method_registry.py`
- `jarvisplot/Figure/adapters.py`

## Current Reality

Current YAML layer methods resolve through the method registry, not through a separate semantic layer registry.

That means this document is a target contract, not a description of a finished runtime system.

## Intended Boundary

A future layer-type registry should answer:

- what kind of layer this is
- which axes it is allowed on
- which render primitive it should use
- which style/profile defaults it should inherit

## Non-Goal

Do not confuse layer type with draw method.

Method dispatch is already implemented. A layer-type registry is a broader semantic contract that does not yet exist as code.
