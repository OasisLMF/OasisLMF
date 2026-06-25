import os
import subprocess
import sys
from pathlib import Path

from .command import OasisBaseCommand

# Deletion runs in a subprocess with NUMBA_DISABLE_JIT=1 because Numba
# re-creates cache files instantly if JIT is active in the same process

_CACHE_GLOBS = ('*.nbi', '*.nbc')

_DELETE_SCRIPT = """\
import sys
from pathlib import Path

roots = sys.argv[1:]
files = [f for r in roots for ext in {globs!r} for f in Path(r).rglob(ext)]
removed = failed = 0
for f in files:
    try:
        f.unlink()
        removed += 1
    except OSError as e:
        print(f'WARNING: {{e}}', flush=True)
        failed += 1
print(removed, failed, flush=True)
""".format(globs=_CACHE_GLOBS)


def _find_cache_files(roots):
    return [f for r in roots for ext in _CACHE_GLOBS for f in r.rglob(ext)]


class ClearCacheCmd(OasisBaseCommand):
    """Deletes all Numba cache files."""

    def add_args(self, parser):
        super().add_args(parser)
        parser.add_argument(
            '--list',
            action='store_true',
            help='List Numba cache files without deleting them',
        )

    def action(self, args):
        pkg_root = Path(__file__).resolve().parents[1]
        roots = [pkg_root]
        cache_dir = os.environ.get('NUMBA_CACHE_DIR')
        if cache_dir:
            roots.append(Path(cache_dir))

        if args.list:
            cache_files = _find_cache_files(roots)
            if not cache_files:
                self.logger.info('No Numba cache files found.')
                return
            for f in sorted(cache_files):
                self.logger.info(f'  {f}')
            self.logger.info(f'Numba cache files found: {len(cache_files)}')
            return

        env = {**os.environ, 'NUMBA_DISABLE_JIT': '1'}
        result = subprocess.run(
            [sys.executable, '-c', _DELETE_SCRIPT, *[str(r) for r in roots]],
            env=env,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            self.logger.error(result.stderr)
            return 1

        for line in result.stdout.splitlines():
            if line.startswith('WARNING:'):
                self.logger.warning(line[len('WARNING:'):].strip())
                continue
            removed, failed = map(int, line.split())
            msg = f'Numba cache files removed: {removed}'
            if failed:
                msg += f'\nNumba cache files unable to be removed: {failed}'
            self.logger.info(msg)
