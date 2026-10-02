import time
from collections import defaultdict
from unittest import TestCase

from oasislmf.execution.resource_monitor import _plot_cpu_time


class _StubAxes:
    def __init__(self):
        self.lines = {}
        self.yaxis = self

    def plot(self, x, y, label=None, **kwargs):
        self.lines[label] = (list(x), list(y))

    def __getattr__(self, name):
        # set_xlabel, legend, grid, set_major_formatter, ...
        return lambda *args, **kwargs: None


class _StubFigure:
    def tight_layout(self):
        pass

    def savefig(self, path, **kwargs):
        pass


class _StubPlt:
    """Minimal stand-in for matplotlib.pyplot that records plotted series."""

    def __init__(self):
        self.ax = _StubAxes()

    def subplots(self, **kwargs):
        return _StubFigure(), self.ax

    def close(self, *args):
        pass


class _StubTicker:
    @staticmethod
    def FuncFormatter(func):
        return func


def _make_rows(n_ts, pids_per_tool, tools=("gulmc", "fmpy", "summarypy")):
    rows = []
    for ts in range(n_ts):
        for tool in tools:
            for pid in range(pids_per_tool):
                # Some processes exit part way through, so the high-water mark
                # must carry their final CPU time forward.
                if pid % 3 == 0 and ts > n_ts // 2:
                    continue
                rows.append({
                    "ts": 1000.0 + ts,
                    "tool": tool,
                    "pid": f"{tool}-{pid}",
                    "cpu_user": ts * 0.5 + pid,
                    "cpu_sys": ts * 0.1,
                })
    return rows


def _expected_series(rows):
    """Reference implementation: the original per-timestamp full scan."""
    t0 = min(r["ts"] for r in rows)
    tool_ts_map = defaultdict(list)
    tool_pid_hwm = defaultdict(dict)
    for ts in sorted(set(r["ts"] for r in rows)):
        for r in [r for r in rows if r["ts"] == ts]:
            cpu_total = r["cpu_user"] + r["cpu_sys"]
            if cpu_total > tool_pid_hwm[r["tool"]].get(r["pid"], 0):
                tool_pid_hwm[r["tool"]][r["pid"]] = cpu_total
        for tool, pid_hwm in tool_pid_hwm.items():
            tool_ts_map[tool].append((ts - t0, sum(pid_hwm.values())))
    return {tool: ([s[0] for s in series], [s[1] for s in series]) for tool, series in tool_ts_map.items()}


class PlotCpuTimeTests(TestCase):
    def test_series_match_reference(self):
        rows = _make_rows(n_ts=40, pids_per_tool=6)
        plt = _StubPlt()

        _plot_cpu_time(rows, "/tmp", plt, _StubTicker)

        self.assertEqual(plt.ax.lines, _expected_series(rows))

    def test_scales_linearly_with_row_count(self):
        # ~1 hour of 1s samples (~200k rows): the old O(T*N) grouping takes ~25s
        # on this, the fixed version well under a second.
        rows = _make_rows(n_ts=3600, pids_per_tool=20)
        plt = _StubPlt()

        start = time.perf_counter()
        _plot_cpu_time(rows, "/tmp", plt, _StubTicker)
        elapsed = time.perf_counter() - start

        self.assertEqual(len(plt.ax.lines["gulmc"][0]), 3600)
        self.assertLess(elapsed, 5)
