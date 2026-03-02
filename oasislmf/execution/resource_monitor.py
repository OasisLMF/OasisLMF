"""
Resource monitor for pytools processes during model runs.

Polls ``ps`` at a configurable interval to capture CPU%, RSS, and VSZ for all
active pytools processes, writes raw data to CSV, and optionally generates a
markdown report with plots (requires matplotlib).
"""

import csv
import logging
import os
import re
import subprocess
import threading
import time
from collections import defaultdict
from datetime import datetime

logger = logging.getLogger(__name__)

# Full list of pytools tool names to monitor (from check_complete in bash.py)
MONITORED_TOOLS = frozenset([
    'evepy', 'modelpy', 'gulpy', 'fmpy', 'gulmc',
    'summarypy', 'plapy', 'katpy', 'eltpy', 'pltpy', 'aalpy', 'lecpy',
])

# Regex to extract a tool name from a process's command-line args
_TOOL_RE = re.compile(r'\b(' + '|'.join(MONITORED_TOOLS) + r')\b')

# Consistent colors per tool (for plots)
TOOL_COLORS = {
    "evepy": "#17becf",
    "modelpy": "#bcbd22",
    "gulpy": "#7f7f7f",
    "fmpy": "#aec7e8",
    "gulmc": "#ffbb78",
    "summarypy": "#1f77b4",
    "pltpy": "#ff7f0e",
    "eltpy": "#2ca02c",
    "katpy": "#d62728",
    "aalpy": "#9467bd",
    "lecpy": "#8c564b",
    "plapy": "#e377c2",
}


class ResourceMonitor:
    """Daemon-thread based resource monitor for pytools processes.

    Args:
        output_dir (str): Directory to write resource_monitor.csv into.
        poll_interval (float): Seconds between polls (default 1.0).
    """

    def __init__(self, output_dir, poll_interval=1.0):
        self.output_dir = output_dir
        self.poll_interval = max(0.1, float(poll_interval))
        self._stop_event = threading.Event()
        self._thread = None
        self._csv_path = os.path.join(output_dir, 'resource_monitor.csv')
        self._bash_pid = None

    def start(self, bash_pid):
        """Start the monitor in a daemon thread.

        Args:
            bash_pid (int): PID of the bash process running the kernel script.
        """
        self._bash_pid = bash_pid
        os.makedirs(self.output_dir, exist_ok=True)
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        logger.info("Resource monitor started (pid=%s, interval=%.1fs, csv=%s)",
                    bash_pid, self.poll_interval, self._csv_path)

    def stop(self):
        """Signal the monitor to stop, wait for the thread, generate report."""
        if self._thread is None:
            return

        self._stop_event.set()
        self._thread.join(timeout=self.poll_interval + 5.0)
        self._thread = None
        logger.info("Resource monitor stopped, csv=%s", self._csv_path)

        self._generate_report()

    # -- internal --------------------------------------------------------

    def _monitor_loop(self):
        """Thread target: poll ps and write CSV rows.

        The CSV file is created lazily on the first poll that finds monitored
        processes.  This avoids a race with the bash kernel script which runs
        ``rm -R -f $LOG_DIR/*`` during its initialisation phase.
        """
        csvfile = None
        writer = None
        try:
            while not self._stop_event.is_set():
                try:
                    rows = self._poll_ps()
                    if rows:
                        if csvfile is None:
                            os.makedirs(self.output_dir, exist_ok=True)
                            csvfile = open(self._csv_path, 'w', newline='')
                            writer = csv.writer(csvfile)
                            writer.writerow(['timestamp', 'tool', 'pid', 'cpu_pct', 'rss_kb', 'vsz_kb'])
                        writer.writerows(rows)
                        csvfile.flush()
                except Exception:
                    logger.debug("Resource monitor poll error", exc_info=True)

                self._stop_event.wait(self.poll_interval)
        except Exception:
            logger.warning("Resource monitor thread failed", exc_info=True)
        finally:
            if csvfile is not None:
                csvfile.close()

    def _poll_ps(self):
        """Run ps and return list of CSV rows for monitored tools."""
        try:
            output = subprocess.check_output(
                ['ps', '-eo', 'pid,pcpu,rss,vsz,args', '--no-headers'],
                stderr=subprocess.DEVNULL,
                text=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            return []

        ts = time.time()
        rows = []
        for line in output.splitlines():
            parts = line.split(None, 4)
            if len(parts) < 5:
                continue
            pid_str, cpu_str, rss_str, vsz_str, args = parts
            match = _TOOL_RE.search(args)
            if not match:
                continue
            tool = match.group(1)
            rows.append([ts, tool, pid_str, cpu_str, rss_str, vsz_str])
        return rows

    # -- report generation -----------------------------------------------

    def _generate_report(self):
        """Read the CSV and generate a markdown report with optional plots."""
        if not os.path.isfile(self._csv_path):
            logger.info("No resource monitor CSV found, skipping report")
            return

        rows = _load_csv(self._csv_path)
        if not rows:
            logger.info("Resource monitor CSV is empty, skipping report")
            return

        report_dir = os.path.join(self.output_dir, 'resource_report')
        os.makedirs(report_dir, exist_ok=True)

        stats = _compute_stats(rows)
        plots = _generate_plots(rows, report_dir)
        md_path = _write_markdown(self._csv_path, report_dir, stats, plots, len(rows))
        logger.info("Resource monitor report: %s (%d samples, %d plots)",
                    md_path, len(rows), len(plots))


# ===== Standalone helper functions (reused from runs/plot_profile.py) =====

def _load_csv(path):
    """Load CSV into a list of dicts with typed values."""
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                rows.append({
                    "ts": float(row["timestamp"]),
                    "tool": row["tool"],
                    "pid": int(row["pid"]),
                    "cpu": float(row["cpu_pct"]),
                    "rss": int(row["rss_kb"]),
                    "vsz": int(row["vsz_kb"]),
                })
            except (ValueError, KeyError):
                continue
    return rows


def _compute_stats(rows):
    """Compute per-tool summary statistics."""
    tools = defaultdict(list)
    for r in rows:
        tools[r["tool"]].append(r)

    stats = {}
    for tool, samples in tools.items():
        pids = set(s["pid"] for s in samples)
        pid_peak_rss = {}
        for s in samples:
            pid_peak_rss[s["pid"]] = max(pid_peak_rss.get(s["pid"], 0), s["rss"])

        ts_groups = defaultdict(list)
        for s in samples:
            ts_groups[s["ts"]].append(s)
        agg_peak_rss = max(sum(s["rss"] for s in g) for g in ts_groups.values()) / 1024
        agg_peak_vsz = max(sum(s["vsz"] for s in g) for g in ts_groups.values()) / 1024

        single_peak_rss = max(pid_peak_rss.values()) / 1024
        peak_cpu = max(s["cpu"] for s in samples)
        avg_rss = sum(s["rss"] for s in samples) / len(samples) / 1024
        avg_cpu = sum(s["cpu"] for s in samples) / len(samples)
        duration = max(s["ts"] for s in samples) - min(s["ts"] for s in samples)

        stats[tool] = {
            "n_instances": len(pids),
            "n_samples": len(samples),
            "single_peak_rss_mb": single_peak_rss,
            "agg_peak_rss_mb": agg_peak_rss,
            "agg_peak_vsz_mb": agg_peak_vsz,
            "peak_cpu": peak_cpu,
            "avg_rss_mb": avg_rss,
            "avg_cpu": avg_cpu,
            "duration_s": duration,
        }
    return stats


def _aggregate_by_tool_and_time(rows):
    """Aggregate rows by (tool, timestamp) -> dict[tool] -> sorted list of tuples."""
    t0 = min(r["ts"] for r in rows)
    groups = defaultdict(list)
    for r in rows:
        groups[(r["tool"], r["ts"])].append(r)

    tool_series = defaultdict(list)
    for (tool, ts), samples in sorted(groups.items()):
        t_rel = ts - t0
        total_rss = sum(s["rss"] for s in samples) / 1024
        total_vsz = sum(s["vsz"] for s in samples) / 1024
        total_cpu = sum(s["cpu"] for s in samples)
        n_procs = len(samples)
        tool_series[tool].append((t_rel, total_rss, total_vsz, total_cpu, n_procs))
    return tool_series


def _per_pid_series(rows):
    """Returns dict[tool] -> dict[pid] -> sorted list of (t_rel, rss_mb, vsz_mb, cpu)."""
    t0 = min(r["ts"] for r in rows)
    data = defaultdict(lambda: defaultdict(list))
    for r in rows:
        t_rel = r["ts"] - t0
        data[r["tool"]][r["pid"]].append((t_rel, r["rss"] / 1024, r["vsz"] / 1024, r["cpu"]))
    for tool in data:
        for pid in data[tool]:
            data[tool][pid].sort()
    return data


def _generate_plots(rows, report_dir):
    """Generate PNG plots if matplotlib is available. Returns dict of plot paths."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        logger.warning("matplotlib not installed — skipping resource monitor plots. "
                       "Install with: pip install oasislmf[extra]")
        return {}

    tool_series = _aggregate_by_tool_and_time(rows)
    pid_data = _per_pid_series(rows)

    plots = {}
    plots["agg_rss"] = _plot_aggregate_rss(tool_series, report_dir, plt, ticker)
    plots["total"] = _plot_total_footprint(tool_series, report_dir, plt, ticker)
    plots["agg_cpu"] = _plot_aggregate_cpu(tool_series, report_dir, plt)
    plots["instances"] = _plot_instance_count(tool_series, report_dir, plt, ticker)
    per_pid = _plot_per_pid_rss(pid_data, report_dir, plt, ticker)
    if per_pid:
        plots["per_pid"] = per_pid
    return plots


def _plot_aggregate_rss(tool_series, out_dir, plt, ticker):
    """Stacked area: aggregate RSS per tool over time."""
    fig, ax = plt.subplots(figsize=(12, 5))
    order = sorted(tool_series.keys(), key=lambda t: tool_series[t][0][0])
    for tool in order:
        series = tool_series[tool]
        ts = [s[0] for s in series]
        rss = [s[1] for s in series]
        ax.plot(ts, rss, label=tool, color=TOOL_COLORS.get(tool), linewidth=1.5)
        ax.fill_between(ts, 0, rss, alpha=0.15, color=TOOL_COLORS.get(tool))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Aggregate RSS (MB)")
    ax.set_title("Aggregate RSS per Tool (summed across all instances)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    fig.tight_layout()
    path = os.path.join(out_dir, "aggregate_rss.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_total_footprint(tool_series, out_dir, plt, ticker):
    """Stacked area of all tools combined."""
    all_ts = sorted(set(s[0] for series in tool_series.values() for s in series))
    order = sorted(tool_series.keys(), key=lambda t: tool_series[t][0][0])
    tool_rss_map = {}
    for tool in order:
        tool_rss_map[tool] = {s[0]: s[1] for s in tool_series[tool]}
    bottoms = [0.0] * len(all_ts)
    fig, ax = plt.subplots(figsize=(12, 5))
    for tool in order:
        rss_map = tool_rss_map[tool]
        vals = [rss_map.get(t, 0) for t in all_ts]
        ax.fill_between(all_ts, bottoms, [b + v for b, v in zip(bottoms, vals)],
                        label=tool, alpha=0.6, color=TOOL_COLORS.get(tool))
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Total RSS (MB)")
    ax.set_title("Total Memory Footprint (stacked by tool)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    fig.tight_layout()
    path = os.path.join(out_dir, "total_footprint.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_aggregate_cpu(tool_series, out_dir, plt):
    """Line plot: aggregate CPU% per tool over time."""
    fig, ax = plt.subplots(figsize=(12, 5))
    order = sorted(tool_series.keys(), key=lambda t: tool_series[t][0][0])
    for tool in order:
        series = tool_series[tool]
        ts = [s[0] for s in series]
        cpu = [s[3] for s in series]
        ax.plot(ts, cpu, label=tool, color=TOOL_COLORS.get(tool), linewidth=1.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Aggregate CPU%")
    ax.set_title("Aggregate CPU% per Tool (summed across all instances)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "aggregate_cpu.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_instance_count(tool_series, out_dir, plt, ticker):
    """Line plot: number of active instances per tool over time."""
    fig, ax = plt.subplots(figsize=(12, 4))
    order = sorted(tool_series.keys(), key=lambda t: tool_series[t][0][0])
    for tool in order:
        series = tool_series[tool]
        ts = [s[0] for s in series]
        n = [s[4] for s in series]
        ax.step(ts, n, label=tool, color=TOOL_COLORS.get(tool), linewidth=1.5, where="post")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Active Instances")
    ax.set_title("Active Process Count per Tool")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    fig.tight_layout()
    path = os.path.join(out_dir, "instance_count.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_per_pid_rss(pid_data, out_dir, plt, ticker):
    """One subplot per tool showing individual PID RSS traces."""
    tools = sorted(pid_data.keys())
    n = len(tools)
    if n == 0:
        return None
    fig, axes = plt.subplots(n, 1, figsize=(12, 3.5 * n), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, tool in zip(axes, tools):
        pids = pid_data[tool]
        for pid, series in sorted(pids.items()):
            ts = [s[0] for s in series]
            rss = [s[1] for s in series]
            ax.plot(ts, rss, linewidth=0.8, alpha=0.7)
        ax.set_ylabel("RSS (MB)")
        ax.set_title(f"{tool} — per-instance RSS ({len(pids)} PIDs)")
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    path = os.path.join(out_dir, "per_pid_rss.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _write_markdown(csv_path, report_dir, stats, plots, total_samples):
    """Write the markdown summary report."""
    md_path = os.path.join(report_dir, "resource_report.md")
    lines = []
    a = lines.append

    a("# Resource Monitor Report")
    a("")
    a(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    a(f"**Total samples:** {total_samples:,}")
    a(f"**Source:** `{os.path.basename(csv_path)}`")
    a("")

    # Summary table
    a("## Summary Statistics")
    a("")
    a("| Tool | Instances | Samples | Peak RSS/inst (MB) | Aggr Peak RSS (MB) | "
      "Peak CPU% | Avg RSS (MB) | Avg CPU% | Duration (s) |")
    a("|------|-----------|---------|--------------------|--------------------|"
      "-----------|--------------|----------|--------------|")
    for tool in sorted(stats.keys()):
        s = stats[tool]
        a(f"| {tool} | {s['n_instances']} | {s['n_samples']} | "
          f"{s['single_peak_rss_mb']:.1f} | {s['agg_peak_rss_mb']:.1f} | "
          f"{s['peak_cpu']:.1f}% | {s['avg_rss_mb']:.1f} | "
          f"{s['avg_cpu']:.1f}% | {s['duration_s']:.1f} |")
    a("")

    # Plots (if generated)
    if plots.get("agg_rss"):
        a("## Aggregate RSS (all instances summed per tool)")
        a("")
        a(f"![Aggregate RSS]({os.path.basename(plots['agg_rss'])})")
        a("")

    if plots.get("total"):
        a("## Total Memory Footprint (stacked)")
        a("")
        a(f"![Total Footprint]({os.path.basename(plots['total'])})")
        a("")

    if plots.get("agg_cpu"):
        a("## Aggregate CPU%")
        a("")
        a(f"![Aggregate CPU]({os.path.basename(plots['agg_cpu'])})")
        a("")

    if plots.get("instances"):
        a("## Active Instance Count")
        a("")
        a(f"![Instance Count]({os.path.basename(plots['instances'])})")
        a("")

    if plots.get("per_pid"):
        a("## Per-Instance RSS Traces")
        a("")
        a(f"![Per PID RSS]({os.path.basename(plots['per_pid'])})")
        a("")

    a("---")
    a(f"*Source: `{csv_path}`*")

    with open(md_path, "w") as f:
        f.write("\n".join(lines))

    return md_path
