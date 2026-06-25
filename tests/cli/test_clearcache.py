import logging
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import oasislmf.cli.clearcache as _cc_mod
from oasislmf.cli.clearcache import ClearCacheCmd

INTERCEPT_FILE = os.path.realpath(_cc_mod.__file__)


def path_patch_side_effect(pkg_root_path):
    """Hits Path(__file__).resolve().parents[1] without hitting other Path uses"""
    _Path = Path

    def _side_effect(p):
        if os.path.realpath(str(p)) == INTERCEPT_FILE:
            mock = MagicMock()
            mock.resolve.return_value.parents.__getitem__.return_value = _Path(pkg_root_path)
            return mock
        return _Path(p)

    return _side_effect


def _run_cmd(argv, pkg_root_path, numba_cache_dir=None):
    env_patch = {}
    if numba_cache_dir is not None:
        env_patch['NUMBA_CACHE_DIR'] = str(numba_cache_dir)

    with patch.dict(os.environ, env_patch):
        with patch('oasislmf.cli.clearcache.Path', side_effect=path_patch_side_effect(pkg_root_path)):
            cmd = ClearCacheCmd(argv=argv)
            cmd.parse_args()
            cmd.action(cmd.args)
    return cmd


def create_cache_files(directory):
    """Create .nbi/.nbc files and returns their paths"""
    files = []
    for name in ('foo', 'bar'):
        for ext in ('.nbi', '.nbc'):
            f = directory / (name + ext)
            f.write_bytes(b'hello world')
            files.append(f)
    nested = directory / 'subpkg'
    nested.mkdir()
    for ext in ('.nbi', '.nbc'):
        f = nested / ('nested_func' + ext)
        f.write_bytes(b'world hello')
        files.append(f)
    return files


@pytest.fixture()
def create_cache_dirs(tmp_path):
    """Return (pkg_root, cache_dir) as isolated temp directories."""
    pkg_root = tmp_path / 'pkg'
    pkg_root.mkdir()
    cache_dir = tmp_path / 'cache'
    cache_dir.mkdir()
    return pkg_root, cache_dir


def test_list_finds_nbi_nbc_files_without_deleting(create_cache_dirs, caplog):
    pkg_root, cache_dir = create_cache_dirs
    pkg_files = create_cache_files(pkg_root)
    cache_files = create_cache_files(cache_dir)

    with caplog.at_level(logging.INFO, logger='oasislmf'):
        _run_cmd(['--list'], pkg_root_path=pkg_root, numba_cache_dir=cache_dir)

    for f in pkg_files + cache_files:
        assert f.exists()
        assert f.name in caplog.text


def test_list_with_no_cache_files_does_not_raise(create_cache_dirs, caplog):
    pkg_root, _ = create_cache_dirs

    with caplog.at_level(logging.INFO, logger='oasislmf'):
        _run_cmd(['--list'], pkg_root_path=pkg_root)

    assert 'No Numba cache files found' in caplog.text


def test_delete_removes_nbi_nbc_from_pkg_root_and_cache_dir(create_cache_dirs):
    pkg_root, cache_dir = create_cache_dirs
    pkg_files = create_cache_files(pkg_root)
    cache_files = create_cache_files(cache_dir)

    bonus_file = pkg_root / 'hello.py'
    bonus_file.write_text('hello world')

    _run_cmd([], pkg_root_path=pkg_root, numba_cache_dir=cache_dir)

    for f in pkg_files + cache_files:
        assert not f.exists()
    assert bonus_file.exists()


def test_delete_without_numba_cache_dir_only_clears_pkg_root(tmp_path):
    pkg_root = tmp_path / 'pkg'
    pkg_root.mkdir()
    unrelated = tmp_path / 'unrelated_cache'
    unrelated.mkdir()

    pkg_files = create_cache_files(pkg_root)
    unrelated_files = create_cache_files(unrelated)

    _run_cmd([], pkg_root_path=pkg_root, numba_cache_dir=None)

    for f in pkg_files:
        assert not f.exists()
    for f in unrelated_files:
        assert f.exists()


def test_clearcache_does_not_affect_numba_files_outside_roots(create_cache_dirs):
    pkg_root, cache_dir = create_cache_dirs
    create_cache_files(pkg_root)
    create_cache_files(cache_dir)

    untouched = cache_dir.parent / 'untouched'
    untouched.mkdir()
    untouched_files = create_cache_files(untouched)

    _run_cmd([], pkg_root_path=pkg_root, numba_cache_dir=cache_dir)

    for f in untouched_files:
        assert f.exists()


_JIT_SCRIPT = textwrap.dedent("""\
    import os, sys
    import numba as nb

    SCALE = int(os.environ.get('_TEST_SCALE', '1'))

    @nb.jit(nopython=True, cache=True)
    def compute(x):
        return x * SCALE

    if __name__ == '__main__':
        result = compute(int(sys.argv[1]))
        print(result, flush=True)
""")


def _run_jit_script(script_path, scale, cache_dir):
    env = {**os.environ, '_TEST_SCALE': str(scale), 'NUMBA_CACHE_DIR': str(cache_dir)}
    env.pop('NUMBA_DISABLE_JIT', None)
    result = subprocess.run(
        [sys.executable, str(script_path), '5'],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0
    return result.stdout.strip()


@pytest.mark.jit_compile
def test_clearcache_allows_env_var_update_to_take_effect(tmp_path):
    script = tmp_path / 'jit_func.py'
    script.write_text(_JIT_SCRIPT)
    cache_dir = tmp_path / 'numba_cache'
    cache_dir.mkdir()
    pkg_root = tmp_path / 'pkg'
    pkg_root.mkdir()

    out1 = _run_jit_script(script, scale=10, cache_dir=cache_dir)
    assert out1 == '50'

    # Assert cache is stored: wrong result should appear
    out2 = _run_jit_script(script, scale=20, cache_dir=cache_dir)
    assert out2 == '50'

    nbi_files = list(cache_dir.rglob('*.nbi'))
    nbc_files = list(cache_dir.rglob('*.nbc'))
    assert nbi_files or nbc_files

    # Verify cache gone and rerun with new env vars
    _run_cmd([], pkg_root_path=pkg_root, numba_cache_dir=cache_dir)
    assert not list(cache_dir.rglob('*.nbi')) and not list(cache_dir.rglob('*.nbc'))

    out3 = _run_jit_script(script, scale=30, cache_dir=cache_dir)
    assert out3 == '150'
