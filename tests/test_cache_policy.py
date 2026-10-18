from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import json
import multiprocessing
from pathlib import Path

import pandas as pd
import pytest

from jarvisplot.cache_registry import CacheRegistry, CacheUsageError, MARKER, PENDING, TEMP_PREFIX
from jarvisplot.cache_store import ProjectCache


def _worker(pipe, workdir, mode):
    """Separate interpreter, explicit barriers; no timing-based lock assertions."""
    try:
        if mode == 'cleaner':
            pipe.send('ready')
            assert pipe.recv() == 'go'
            pipe.send(CacheRegistry().clean())
            return
        pipe.send('starting')
        cache = ProjectCache(workdir)
        cache.put_dataframe('shared', pd.DataFrame({'x': [1, 2]}))
        build = cache.clear_materialized_slot('hdf')
        pd.DataFrame({'x': [3, 4]}).to_parquet(build / 'part.parquet')
        manifest = {'part_files': ['part.parquet']}
        cache.put_materialized_manifest('hdf', manifest)
        part = cache.materialized_slot('hdf') / manifest['part_files'][0]
        pipe.send('ready')
        while True:
            command = pipe.recv()
            if command == 'read':
                pipe.send((cache.get_dataframe('shared')['x'].tolist(),
                           pd.read_parquet(part)['x'].tolist()))
            elif command == 'temporary':
                (cache.data_dir / (TEMP_PREFIX + 'interrupted')).write_bytes(b'partial')
                pipe.send('written')
            elif command == 'close':
                cache.close()
                pipe.send('closed')
                return
            elif command.startswith('write:'):
                prefix = command.split(':')[1]
                for index in range(8):
                    cache.put_named_reference(f'{prefix}-{index}', 'sig', 'shared')
                pipe.send('written')
    except BaseException as exc:
        pipe.send(('error', repr(exc)))
        raise
    finally:
        pipe.close()


def _recv(pipe):
    assert pipe.poll(30), 'cache worker did not respond'
    result = pipe.recv()
    assert not (isinstance(result, tuple) and result[0] == 'error'), result
    return result


@contextmanager
def _process(workdir, mode='reader'):
    ctx = multiprocessing.get_context('spawn')
    parent, child = ctx.Pipe()
    process = ctx.Process(target=_worker, args=(child, str(workdir), mode))
    process.start()
    child.close()
    try:
        yield parent, process
    finally:
        if process.is_alive():
            process.kill()
        process.join(15)
        parent.close()
        assert not process.is_alive()


def _seed(workdir, size=100):
    registry = CacheRegistry()
    with registry.lease(workdir):
        registry.register(workdir)
        root = workdir / '.cache'
        (root / 'data').mkdir(exist_ok=True)
        (root / 'data' / 'payload').write_bytes(b'x' * size)
    return root


def _age(root, days):
    registry = CacheRegistry()
    with registry.lease(root.parent):
        registry.register(root.parent, last_used=(datetime.now(timezone.utc) - timedelta(days=days)).isoformat())


def test_policy_defaults_json_and_environment(tmp_path, monkeypatch):
    registry = CacheRegistry()
    assert registry.policy() == {'max_size_gib': 5, 'max_age_days': 30}
    config = registry.path.with_name('plot-cache.json')
    config.parent.mkdir(parents=True)
    config.write_text(json.dumps({'max_size_gib': 2, 'max_age_days': 10}))
    assert registry.policy() == {'max_size_gib': 2, 'max_age_days': 10}
    monkeypatch.setenv('JP_CACHE_MAX_GIB', '3.5')
    monkeypatch.setenv('JP_CACHE_MAX_AGE_DAYS', '7')
    assert registry.policy() == {'max_size_gib': 3.5, 'max_age_days': 7}


@pytest.mark.parametrize('value', ['0', '-1', 'nan', 'inf', 'abc'])
@pytest.mark.parametrize('env', ['JP_CACHE_MAX_GIB', 'JP_CACHE_MAX_AGE_DAYS'])
def test_invalid_policy_never_deletes_cache(tmp_path, monkeypatch, env, value):
    root = _seed(tmp_path / 'old')
    _age(root, 40)
    monkeypatch.setenv(env, value)
    with pytest.raises(CacheUsageError):
        ProjectCache(str(tmp_path / 'new'))
    assert (root / 'data' / 'payload').is_file()


def test_expiry_and_missing_records_at_session_start(tmp_path):
    old = _seed(tmp_path / 'old')
    recent = _seed(tmp_path / 'recent')
    _age(old, 31)
    _age(recent, 29)
    with ProjectCache(str(tmp_path / 'new')):
        assert not old.exists()
        assert recent.exists()


def test_lru_budget_reclaims_to_eighty_percent_and_preserves_foreign_files(tmp_path, monkeypatch):
    old = _seed(tmp_path / 'old', 4000)
    middle = _seed(tmp_path / 'middle', 4000)
    recent = _seed(tmp_path / 'recent', 4000)
    _age(old, 3)
    _age(middle, 2)
    _age(recent, 1)
    foreign = old / 'foreign.txt'
    foreign.write_text('keep')
    total = sum(row['bytes'] for row in CacheRegistry().list())
    monkeypatch.setenv('JP_CACHE_MAX_GIB', str(total * 0.8 / 1024**3))
    result = CacheRegistry().maintain()
    assert result['removed'] == [str(old), str(middle)]
    assert foreign.read_text() == 'keep'
    assert not (old / 'data').exists()
    assert (recent / 'data' / 'payload').exists()


def test_active_cache_can_exceed_budget_then_is_reclaimed_on_close(tmp_path, monkeypatch):
    cache = ProjectCache(str(tmp_path / 'active'))
    cache.put_dataframe('big', pd.DataFrame({'x': list(range(1000))}))
    monkeypatch.setenv('JP_CACHE_MAX_GIB', str(1000 / 1024**3))
    result = CacheRegistry().maintain()
    assert result['skipped'] == [str(cache.root)]
    assert cache.get_dataframe('big') is not None
    cache.close()
    assert not cache.root.exists()


def test_close_refreshes_last_use(tmp_path):
    cache = ProjectCache(str(tmp_path / 'long-job'))
    _age(cache.root, 40)
    assert str(cache.root) in CacheRegistry().maintain()['skipped']
    cache.close()
    assert cache.root.exists()
    stamp = datetime.fromisoformat(json.loads((cache.root / MARKER).read_text())['last_used'])
    assert (datetime.now(timezone.utc) - stamp).total_seconds() < 10


def test_cross_process_print_clean_and_rebuild_preserve_active_reader(tmp_path):
    workdir = tmp_path / 'shared'
    with _process(workdir) as (pipe, process):
        assert _recv(pipe) == 'starting'
        assert _recv(pipe) == 'ready'
        cache = ProjectCache(str(workdir))
        assert cache.get_dataframe('shared') is not None
        assert cache.get_materialized_manifest('hdf')
        result = cache.clear_used()
        assert result['deferred'] > 0 and result['removed'] == 0
        assert (cache.root / PENDING).is_file()
        assert CacheRegistry().clean()['skipped']
        with pytest.raises(CacheUsageError, match='active'):
            ProjectCache(str(workdir), rebuild=True)
        pipe.send('read')
        assert _recv(pipe) == ([1, 2], [3, 4])
        pipe.send('close')
        assert _recv(pipe) == 'closed'
        process.join(15)
        assert process.exitcode == 0
        assert not (cache.root / PENDING).exists()
        assert not (cache.data_dir / 'shared.pkl').exists()
        assert not (cache.materialized_dir / 'hdf').exists()
        with pytest.raises(RuntimeError, match='closed'):
            cache.get_dataframe('shared')


def test_rebuild_holds_exclusive_lease_until_session_end(tmp_path):
    workdir = tmp_path / 'shared'
    root = _seed(workdir)
    (root / 'foreign.txt').write_text('keep')
    with ProjectCache(str(workdir), rebuild=True) as cache:
        assert not (cache.data_dir / 'payload').exists()
        assert (root / 'foreign.txt').read_text() == 'keep'
        with pytest.raises(BlockingIOError):
            CacheRegistry().lease(workdir, blocking=False)
        assert CacheRegistry().clean()['skipped']
    with CacheRegistry().lease(workdir, blocking=False):
        pass


def test_two_process_cleaners_remove_each_cache_once(tmp_path):
    roots = {_seed(tmp_path / str(i)) for i in range(3)}
    with _process(tmp_path, 'cleaner') as (a, pa), _process(tmp_path, 'cleaner') as (b, pb):
        assert _recv(a) == _recv(b) == 'ready'
        a.send('go')
        b.send('go')
        results = [_recv(a), _recv(b)]
        removed = [p for r in results for p in r['removed']]
        assert len(removed) == len(set(removed)) == 3
        assert set(removed) == {str(p) for p in roots}
        assert not any(r['errors'] for r in results)


def test_crash_releases_reader_and_reaps_deferred_cleanup_and_temporary_files(tmp_path):
    workdir = tmp_path / 'shared'
    with _process(workdir) as (pipe, process):
        assert _recv(pipe) == 'starting'
        assert _recv(pipe) == 'ready'
        pipe.send('temporary')
        assert _recv(pipe) == 'written'
        cache = ProjectCache(str(workdir))
        cache.get_dataframe('shared')
        assert cache.clear_used()['deferred']
        process.kill()
        process.join(15)
        result = CacheRegistry().maintain()
        assert not result['errors']
        assert not (cache.root / PENDING).exists()
        assert not (cache.data_dir / 'shared.pkl').exists()
        assert not list(cache.data_dir.glob(TEMP_PREFIX + '*'))


def test_concurrent_manifest_writers_merge_independent_aliases(tmp_path):
    workdir = tmp_path / 'shared'
    with _process(workdir) as (a, pa), _process(workdir) as (b, pb):
        assert _recv(a) == _recv(b) == 'starting'
        assert _recv(a) == _recv(b) == 'ready'
        a.send('write:first')
        b.send('write:second')
        assert _recv(a) == _recv(b) == 'written'
        manifest = json.loads((workdir / '.cache' / 'manifest.json').read_text())
        assert set(manifest['named']) == {f'{prefix}-{i}' for prefix in ('first', 'second') for i in range(8)}
        a.send('close')
        b.send('close')
        assert _recv(a) == _recv(b) == 'closed'


def test_materialized_generations_survive_other_writer_and_are_reaped_when_idle(tmp_path):
    first = ProjectCache(str(tmp_path))
    build = first.clear_materialized_slot('hdf')
    (build / 'part').write_bytes(b'first')
    m1 = {'part_files': ['part']}
    first.put_materialized_manifest('hdf', m1)
    old_part = first.materialized_slot('hdf') / m1['part_files'][0]
    second = ProjectCache(str(tmp_path))
    build = second.clear_materialized_slot('hdf')
    (build / 'part').write_bytes(b'second')
    m2 = {'part_files': ['part']}
    second.put_materialized_manifest('hdf', m2)
    new_part = second.materialized_slot('hdf') / m2['part_files'][0]
    second.close()
    assert old_part.read_bytes() == b'first'
    assert new_part.read_bytes() == b'second'
    first.close()
    assert not old_part.exists()
    assert new_part.read_bytes() == b'second'


def test_pending_cleanup_does_not_delete_replacement(tmp_path):
    first = ProjectCache(str(tmp_path))
    first.put_dataframe('same', pd.DataFrame({'x': [1]}))
    second = ProjectCache(str(tmp_path))
    first.clear_used()
    second.put_dataframe('same', pd.DataFrame({'x': [2]}))
    second.close()
    assert pd.read_pickle(tmp_path / '.cache' / 'data' / 'same.pkl')['x'].tolist() == [2]


def test_failed_atomic_payload_write_keeps_previous_file(tmp_path, monkeypatch):
    with ProjectCache(str(tmp_path)) as cache:
        cache.put_dataframe('same', pd.DataFrame({'x': [1]}))
        def fail(df, path):
            Path(path).write_bytes(b'partial')
            raise OSError('disk full')
        monkeypatch.setattr(pd.DataFrame, 'to_pickle', fail)
        cache.put_dataframe('same', pd.DataFrame({'x': [2]}))
        assert cache.get_dataframe('same')['x'].tolist() == [1]
        assert cache.is_dataframe_meta_consistent('same', cache.get_dataframe_meta('same'))
        assert not list(cache.data_dir.glob(TEMP_PREFIX + '*'))


def test_eviction_rechecks_reader_arriving_after_inventory(tmp_path, monkeypatch):
    root = _seed(tmp_path / 'old')
    _age(root, 40)
    registry = CacheRegistry()
    original = registry.lease
    calls = 0
    readers = []

    def arriving_reader(workdir, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            readers.append(original(workdir, blocking=False))
        return original(workdir, **kwargs)

    monkeypatch.setattr(registry, 'lease', arriving_reader)
    try:
        result = registry.maintain()
        assert result['removed'] == []
        assert result['skipped'] == [str(root)]
        assert (root / 'data' / 'payload').exists()
    finally:
        for reader in readers:
            reader.close()
    assert registry.maintain()['removed'] == [str(root)]


@pytest.mark.parametrize('mode', ['busy', 'invalid', 'corrupt'])
def test_cache_input_errors_are_reported_without_traceback(tmp_path, monkeypatch, capsys, mode):
    from jarvisplot.client import main
    from loguru import logger

    config = tmp_path / 'plot.yaml'
    config.write_text('Figures: []\n')
    cache = ProjectCache(str(tmp_path))
    if mode == 'invalid':
        monkeypatch.setenv('JP_CACHE_MAX_GIB', 'nan')
    elif mode == 'corrupt':
        CacheRegistry().path.with_name('plot-cache.json').write_text('broken json')
    try:
        assert main([str(config), '--rebuild-cache']) == 2
        captured = capsys.readouterr()
        assert 'Traceback' not in captured.err
        assert ('Cannot rebuild active' if mode == 'busy' else
                'max_size_gib' if mode == 'invalid' else 'Cannot read cache policy') in captured.err
        assert not list(tmp_path.glob('*.png'))
    finally:
        cache.close()
        logger.remove()


@pytest.mark.parametrize('mode', ['busy', 'invalid'])
def test_dryrun_reports_cache_policy_or_lock_failure(tmp_path, monkeypatch, mode):
    from jarvisplot.dryrun_runtime import dryrun_config

    with ProjectCache(str(tmp_path), rebuild=True):
        if mode == 'invalid':
            monkeypatch.setenv('JP_CACHE_MAX_AGE_DAYS', '-1')
        report, bag = dryrun_config({'Figures': []}, base_dir=str(tmp_path), with_data=True)
        assert report['ok'] is False
        assert not bag.ok
        assert not (tmp_path / '.cache' / 'agent_twins').exists()
