from __future__ import annotations

from jarvisplot.Figure.data_pipelines import DataContext, SharedContent


def test_usage_plan_consume_releases_cache_but_keeps_registry():
    calls = {"compute": 0, "release": 0}

    def compute(_shared):
        calls["compute"] += 1
        return {"value": calls["compute"]}

    def release():
        calls["release"] += 1

    shared = SharedContent()
    shared.register("src", compute, release_fn=release)
    shared.set_usage_plan({"src": 2})

    assert shared.is_tracked("src")
    assert shared.is_registered("src")
    assert shared.remaining_uses("src") == 2

    first = shared.get("src")
    assert first == {"value": 1}
    assert shared.has_cached("src")
    assert calls == {"compute": 1, "release": 0}

    remain = shared.consume("src")
    assert remain == 1
    assert shared.has_cached("src")
    assert calls["release"] == 0

    remain = shared.consume("src")
    assert remain == 0
    assert not shared.has_cached("src")
    assert calls["release"] == 1
    # Factory remains so an accidental re-get recomputes instead of hard-failing.
    assert shared.is_registered("src")

    second = shared.get("src")
    assert second == {"value": 2}
    assert calls["compute"] == 2


def test_consume_untracked_name_is_noop():
    shared = SharedContent()
    shared.update("free", {"x": 1})
    assert shared.consume("free") == 0
    assert shared.has_cached("free")
    assert shared.get("free") == {"x": 1}


def test_invalidate_only_releases_when_value_cached():
    releases = {"n": 0}

    def release():
        releases["n"] += 1

    shared = SharedContent()
    shared.update("named", object(), release_fn=release)
    shared.invalidate("named")
    assert releases["n"] == 1
    assert not shared.has_cached("named")

    # Second invalidate must not re-run release for a missing cache entry.
    shared.invalidate("named")
    assert releases["n"] == 1


def test_data_context_facade_mirrors_shared_lifecycle():
    shared = SharedContent()
    ctx = DataContext(shared)
    ctx.register("alias", lambda _s: [1, 2, 3])
    ctx.set_usage_plan({"alias": 1})

    assert ctx.get("alias") == [1, 2, 3]
    assert ctx.has_cached("alias")
    assert ctx.is_tracked("alias")
    assert ctx.stats()["cached"] == 1

    assert ctx.consume("alias") == 0
    assert not ctx.has_cached("alias")
    assert ctx.is_registered("alias")


def test_set_usage_plan_replaces_previous_counts():
    shared = SharedContent()
    shared.set_usage_plan({"a": 3, "b": 1})
    assert shared.remaining_uses("a") == 3
    shared.set_usage_plan({"a": 1})
    assert shared.remaining_uses("a") == 1
    assert shared.remaining_uses("b") == 0
    assert not shared.is_tracked("b")
