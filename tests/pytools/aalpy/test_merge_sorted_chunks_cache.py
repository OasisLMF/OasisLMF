import os
import subprocess
import sys
import textwrap

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("NUMBA_DISABLE_JIT", "0") == "1",
    reason="exercises the Numba disk cache"
)

SCRIPT = textwrap.dedent("""
    import numba as nb
    import numpy as np

    from oasislmf.pytools.aal.manager import _SUMMARIES_DTYPE, merge_sorted_chunks

    @nb.njit
    def count_rows(memmaps):
        n = 0
        for _ in merge_sorted_chunks(memmaps):
            n += 1
        return n

    chunk = np.zeros(3, dtype=_SUMMARIES_DTYPE)
    chunk["summary_id"] = [1, 2, 3]
    memmaps = [chunk]

    assert len(list(merge_sorted_chunks(memmaps))) == 3
    assert count_rows(memmaps) == 3
""")


def test_jitted_consumer_compiles_against_warm_cache(tmp_path):
    """Regression for #1970: a cold jitted consumer of merge_sorted_chunks must compile in a
    process where the rest of the cache is already warm."""
    script = tmp_path / "consume.py"
    script.write_text(SCRIPT)
    env = {**os.environ, "NUMBA_CACHE_DIR": str(tmp_path / "numba_cache")}

    for run in ("cold", "warm"):
        result = subprocess.run([sys.executable, str(script)], env=env, capture_output=True, text=True)
        assert result.returncode == 0, f"{run} run failed:\n{result.stderr[-3000:]}"
