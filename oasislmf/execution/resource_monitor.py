"""
Resource monitor for pytools processes during model runs.

Uses ``psutil`` to poll all active pytools processes at a configurable
interval, capturing CPU%, cumulative CPU time (user + system), RSS,
USS (private memory), and PSS (proportional memory, Linux only).
Writes raw data to CSV and optionally generates a markdown report with
plots (requires matplotlib).

USS (Unique Set Size) is the primary memory metric — it represents memory
private to each process and avoids the double-counting problem that RSS has
with shared memory and memory-mapped files.

CPU time (user + system) from ``psutil.Process.cpu_times()`` gives the
actual processor time consumed, equivalent to TIME+ in ``top``.
"""

import csv
import logging
import os
import re
import sys
import threading
import time
from collections import defaultdict
from datetime import datetime

import psutil

logger = logging.getLogger(__name__)

# Full list of pytools tool names to monitor (from check_complete in bash.py)
MONITORED_TOOLS = frozenset([
    'evepy', 'modelpy', 'gulpy', 'fmpy', 'gulmc',
    'summarypy', 'plapy', 'katpy', 'eltpy', 'pltpy', 'aalpy', 'lecpy',
])

# Regex to extract a tool name from a process's command-line
_TOOL_RE = re.compile(r'\b(' + '|'.join(MONITORED_TOOLS) + r')\b')

CSV_HEADER = ['timestamp', 'tool', 'pid', 'cpu_pct', 'cpu_user_s', 'cpu_sys_s', 'rss_kb', 'uss_kb', 'pss_kb']

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

# PSS is only available on Linux
_HAS_PSS = sys.platform.startswith('linux')


class ResourceMonitor:
    """Daemon-thread based resource monitor for pytools processes.

    Args:
        output_dir (str): Directory to write resource_monitor.csv into.
        poll_interval (float): Seconds between polls (default 1.0).
    """

    def __init__(self, output_dir, poll_interval=1.0, generate_report=True, log_root=None):
        self.output_dir = output_dir
        self.poll_interval = max(0.1, float(poll_interval))
        self.generate_report = generate_report
        self.log_root = log_root
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

        if self.generate_report:
            self._generate_report()

    # -- internal --------------------------------------------------------

    def _monitor_loop(self):
        """Thread target: poll processes and write CSV rows.

        The CSV file is created lazily on the first poll that finds monitored
        processes.  This avoids a race with the bash kernel script which runs
        ``rm -R -f $LOG_DIR/*`` during its initialisation phase.
        """
        csvfile = None
        writer = None
        try:
            while not self._stop_event.is_set():
                try:
                    rows = self._poll_processes()
                    if rows:
                        if csvfile is None:
                            os.makedirs(self.output_dir, exist_ok=True)
                            write_header = not os.path.isfile(self._csv_path)
                            csvfile = open(self._csv_path, 'a', newline='')
                            writer = csv.writer(csvfile)
                            if write_header:
                                writer.writerow(CSV_HEADER)
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

    def _poll_processes(self):
        """Use psutil to find monitored pytools processes and collect metrics.

        Returns:
            list: CSV rows matching :data:`CSV_HEADER`.
        """
        ts = time.time()
        rows = []
        for proc in psutil.process_iter(['pid', 'cmdline']):
            try:
                cmdline = proc.info.get('cmdline')
                if not cmdline:
                    continue
                cmdline_str = ' '.join(cmdline)
                match = _TOOL_RE.search(cmdline_str)
                if not match:
                    continue

                tool = match.group(1)
                cpu_pct = proc.cpu_percent()
                cpu_times = proc.cpu_times()
                mem = proc.memory_full_info()
                rss_kb = mem.rss // 1024
                uss_kb = mem.uss // 1024
                pss_kb = (mem.pss // 1024) if _HAS_PSS else -1

                rows.append([
                    ts, tool, proc.pid, cpu_pct,
                    round(cpu_times.user, 3), round(cpu_times.system, 3),
                    rss_kb, uss_kb, pss_kb,
                ])
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
        return rows

    # -- report generation -----------------------------------------------

    def _find_all_csvs(self):
        """Return all resource_monitor.csv paths found in immediate subdirs of log_root."""
        csv_paths = []
        if not self.log_root:
            logger.info("Resource monitor: log_root not set, skipping CSV discovery")
            return csv_paths
        if not os.path.isdir(self.log_root):
            logger.info("Resource monitor: log_root '%s' does not exist or is not a directory", self.log_root)
            return csv_paths
        logger.info("Resource monitor: scanning '%s' for resource_monitor.csv files", self.log_root)
        with os.scandir(self.log_root) as it:
            for entry in sorted(it, key=lambda e: e.name):
                if not entry.is_dir():
                    continue
                csv_path = os.path.join(entry.path, 'resource_monitor.csv')
                if os.path.isfile(csv_path):
                    logger.info("Resource monitor: found %s", csv_path)
                    csv_paths.append(csv_path)
                else:
                    logger.debug("Resource monitor: no CSV in %s", entry.path)
        logger.info("Resource monitor: %d CSV(s) found to combine", len(csv_paths))
        return csv_paths

    def _generate_report(self):
        """Read the CSV(s) and generate a markdown report with optional plots.

        When ``log_root`` is set all ``resource_monitor.csv`` files found in
        immediate sub-directories of ``log_root`` are combined before the
        report is generated, and the report is written to
        ``log_root/resource_report/``.
        """
        if self.log_root:
            logger.info("Resource monitor: generating combined report from log_root='%s'", self.log_root)
            csv_paths = self._find_all_csvs()
            if not csv_paths:
                logger.info("No resource monitor CSVs found under %s, skipping report", self.log_root)
                return
            rows = []
            for path in csv_paths:
                rows.extend(_load_csv(path))
            report_dir = os.path.join(self.log_root, 'resource_report')
            source_csv = os.path.join(self.log_root, 'resource_monitor.csv')
        else:
            if not os.path.isfile(self._csv_path):
                logger.info("No resource monitor CSV found, skipping report")
                return
            rows = _load_csv(self._csv_path)
            report_dir = os.path.join(self.output_dir, 'resource_report')
            source_csv = self._csv_path

        if not rows:
            logger.info("Resource monitor CSV is empty, skipping report")
            return

        os.makedirs(report_dir, exist_ok=True)

        stats = _compute_stats(rows)
        system_memory_mb = psutil.virtual_memory().total / (1024 * 1024)
        peak_total_uss_mb = _compute_peak_total_uss(rows)
        plots = _generate_plots(rows, report_dir, system_memory_mb)
        md_path = _write_markdown(source_csv, report_dir, stats, plots,
                                  len(rows), system_memory_mb, peak_total_uss_mb)
        logger.info("Resource monitor report: %s (%d samples, %d plots)",
                    md_path, len(rows), len(plots))


# ===== Standalone helper functions =====

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
                    "cpu_user": float(row["cpu_user_s"]),
                    "cpu_sys": float(row["cpu_sys_s"]),
                    "rss": int(row["rss_kb"]),
                    "uss": int(row["uss_kb"]),
                    "pss": int(row["pss_kb"]),
                })
            except (ValueError, KeyError):
                continue
    return rows


def _compute_stats(rows):
    """Compute per-tool summary statistics."""
    tools = defaultdict(list)
    for r in rows:
        tools[r["tool"]].append(r)

    has_pss = any(r["pss"] >= 0 for r in rows)

    stats = {}
    for tool, samples in tools.items():
        pids = set(s["pid"] for s in samples)

        # Per-timestamp aggregation
        ts_groups = defaultdict(list)
        for s in samples:
            ts_groups[s["ts"]].append(s)

        agg_peak_rss = max(sum(s["rss"] for s in g) for g in ts_groups.values()) / 1024
        agg_peak_uss = max(sum(s["uss"] for s in g) for g in ts_groups.values()) / 1024

        pid_peak_uss = {}
        for s in samples:
            pid_peak_uss[s["pid"]] = max(pid_peak_uss.get(s["pid"], 0), s["uss"])
        single_peak_uss = max(pid_peak_uss.values()) / 1024

        peak_cpu = max(s["cpu"] for s in samples)
        avg_uss = sum(s["uss"] for s in samples) / len(samples) / 1024
        avg_cpu = sum(s["cpu"] for s in samples) / len(samples)
        duration = max(s["ts"] for s in samples) - min(s["ts"] for s in samples)

        # Total CPU time: take the final (max) cpu_user + cpu_sys per PID,
        # then sum across all PIDs for this tool.
        pid_final_user = {}
        pid_final_sys = {}
        for s in samples:
            pid_final_user[s["pid"]] = max(pid_final_user.get(s["pid"], 0), s["cpu_user"])
            pid_final_sys[s["pid"]] = max(pid_final_sys.get(s["pid"], 0), s["cpu_sys"])
        total_cpu_user = sum(pid_final_user.values())
        total_cpu_sys = sum(pid_final_sys.values())

        stat = {
            "n_instances": len(pids),
            "n_samples": len(samples),
            "single_peak_uss_mb": single_peak_uss,
            "agg_peak_rss_mb": agg_peak_rss,
            "agg_peak_uss_mb": agg_peak_uss,
            "peak_cpu": peak_cpu,
            "avg_uss_mb": avg_uss,
            "avg_cpu": avg_cpu,
            "duration_s": duration,
            "total_cpu_user_s": round(total_cpu_user, 1),
            "total_cpu_sys_s": round(total_cpu_sys, 1),
            "total_cpu_time_s": round(total_cpu_user + total_cpu_sys, 1),
        }

        if has_pss:
            agg_peak_pss = max(
                sum(s["pss"] for s in g) for g in ts_groups.values()
            ) / 1024
            stat["agg_peak_pss_mb"] = agg_peak_pss

        stats[tool] = stat
    return stats


def _compute_peak_total_uss(rows):
    """Compute the peak instantaneous total USS across all processes.

    Groups all rows by timestamp, sums USS at each instant, returns the max in MB.
    """
    ts_uss = defaultdict(int)
    for r in rows:
        ts_uss[r["ts"]] += r["uss"]
    return max(ts_uss.values()) / 1024 if ts_uss else 0


def _aggregate_by_tool_and_time(rows):
    """Aggregate rows by (tool, timestamp).

    Returns:
        dict[tool] -> sorted list of
        (t_rel, rss_mb, uss_mb, cpu_pct, n_procs, sum_cpu_user, sum_cpu_sys).
    """
    t0 = min(r["ts"] for r in rows)
    groups = defaultdict(list)
    for r in rows:
        groups[(r["tool"], r["ts"])].append(r)

    tool_series = defaultdict(list)
    for (tool, ts), samples in sorted(groups.items()):
        t_rel = ts - t0
        total_rss = sum(s["rss"] for s in samples) / 1024
        total_uss = sum(s["uss"] for s in samples) / 1024
        total_cpu = sum(s["cpu"] for s in samples)
        n_procs = len(samples)
        sum_cpu_user = sum(s["cpu_user"] for s in samples)
        sum_cpu_sys = sum(s["cpu_sys"] for s in samples)
        tool_series[tool].append((t_rel, total_rss, total_uss, total_cpu, n_procs,
                                  sum_cpu_user, sum_cpu_sys))
    return tool_series


def _generate_plots(rows, report_dir, system_memory_mb=None):
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

    plots = {}
    plots["agg_uss"] = _plot_aggregate_uss(tool_series, report_dir, plt, ticker)
    plots["agg_rss_vs_uss"] = _plot_aggregate_rss_vs_uss(tool_series, report_dir, plt, ticker)
    plots["total"] = _plot_total_footprint(tool_series, report_dir, plt, ticker, system_memory_mb)
    plots["agg_cpu"] = _plot_aggregate_cpu(tool_series, report_dir, plt)
    plots["cpu_time"] = _plot_cpu_time(rows, report_dir, plt, ticker)
    plots["instances"] = _plot_instance_count(tool_series, report_dir, plt, ticker)
    return plots


def _plot_aggregate_uss(tool_series, out_dir, plt, ticker):
    """Line + fill: aggregate USS (private memory) per tool over time."""
    fig, ax = plt.subplots(figsize=(12, 5))
    order = sorted(tool_series.keys(), key=lambda t: tool_series[t][0][0])
    for tool in order:
        series = tool_series[tool]
        ts = [s[0] for s in series]
        uss = [s[2] for s in series]
        ax.plot(ts, uss, label=tool, color=TOOL_COLORS.get(tool), linewidth=1.5)
        ax.fill_between(ts, 0, uss, alpha=0.15, color=TOOL_COLORS.get(tool))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Aggregate USS (MB)")
    ax.set_title("Aggregate USS per Tool — Private Memory (summed across instances)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    fig.tight_layout()
    path = os.path.join(out_dir, "aggregate_uss.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_aggregate_rss_vs_uss(tool_series, out_dir, plt, ticker):
    """Compare RSS vs USS per tool to visualise shared memory overhead."""
    fig, ax = plt.subplots(figsize=(12, 5))
    order = sorted(tool_series.keys(), key=lambda t: tool_series[t][0][0])
    for tool in order:
        series = tool_series[tool]
        ts = [s[0] for s in series]
        rss = [s[1] for s in series]
        uss = [s[2] for s in series]
        color = TOOL_COLORS.get(tool)
        ax.plot(ts, rss, label=f"{tool} RSS", color=color, linewidth=1.5, alpha=0.4, linestyle='--')
        ax.plot(ts, uss, label=f"{tool} USS", color=color, linewidth=1.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Memory (MB)")
    ax.set_title("RSS vs USS per Tool — gap shows shared/mmap memory")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    fig.tight_layout()
    path = os.path.join(out_dir, "rss_vs_uss.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_total_footprint(tool_series, out_dir, plt, ticker, system_memory_mb=None):
    """Stacked area of USS (private memory) across all tools."""
    all_ts = sorted(set(s[0] for series in tool_series.values() for s in series))
    order = sorted(tool_series.keys(), key=lambda t: tool_series[t][0][0])
    tool_uss_map = {}
    for tool in order:
        tool_uss_map[tool] = {s[0]: s[2] for s in tool_series[tool]}
    bottoms = [0.0] * len(all_ts)
    fig, ax = plt.subplots(figsize=(12, 5))
    for tool in order:
        uss_map = tool_uss_map[tool]
        vals = [uss_map.get(t, 0) for t in all_ts]
        ax.fill_between(all_ts, bottoms, [b + v for b, v in zip(bottoms, vals)],
                        label=tool, alpha=0.6, color=TOOL_COLORS.get(tool))
        bottoms = [b + v for b, v in zip(bottoms, vals)]

    # Peak total USS annotation
    peak_total = max(bottoms)
    peak_idx = bottoms.index(peak_total)
    peak_ts = all_ts[peak_idx]
    ax.axhline(y=peak_total, color='red', linestyle=':', linewidth=1, alpha=0.7)
    ax.annotate(f"Peak: {peak_total:,.0f} MB",
                xy=(peak_ts, peak_total), xytext=(10, 8),
                textcoords="offset points", fontsize=9, color='red',
                arrowprops=dict(arrowstyle='->', color='red', lw=0.8))

    # System memory line
    if system_memory_mb:
        ax.axhline(y=system_memory_mb, color='black', linestyle='--', linewidth=1.2, alpha=0.5)
        ax.text(all_ts[0], system_memory_mb, f" System: {system_memory_mb:,.0f} MB",
                va='bottom', ha='left', fontsize=9, color='black', alpha=0.7)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Total USS (MB)")
    ax.set_title("Total Private Memory Footprint (stacked by tool)")
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


def _plot_cpu_time(rows, out_dir, plt, ticker):
    """Line plot: cumulative CPU time (user + system) per tool over time.

    Uses per-PID high-water marks so that when a process exits, its final
    CPU time is retained in the total rather than dropping to zero.
    """
    t0 = min(r["ts"] for r in rows)

    # Collect all unique timestamps and sort them
    all_ts = sorted(set(r["ts"] for r in rows))

    # Group rows by (tool, ts, pid) and keep max cpu_time per pid
    # Build: tool -> {pid -> best_cpu_time} updated over time
    tool_ts_map = defaultdict(list)  # tool -> [(t_rel, total_cpu_time)]
    tool_pid_hwm = defaultdict(dict)  # tool -> {pid -> max(user+sys)}

    for ts in all_ts:
        # Update high-water marks from rows at this timestamp
        ts_rows = [r for r in rows if r["ts"] == ts]
        for r in ts_rows:
            cpu_total = r["cpu_user"] + r["cpu_sys"]
            tool = r["tool"]
            pid = r["pid"]
            if cpu_total > tool_pid_hwm[tool].get(pid, 0):
                tool_pid_hwm[tool][pid] = cpu_total

        # Record the sum of high-water marks for each tool at this timestamp
        t_rel = ts - t0
        for tool, pid_hwm in tool_pid_hwm.items():
            tool_ts_map[tool].append((t_rel, sum(pid_hwm.values())))

    fig, ax = plt.subplots(figsize=(12, 5))
    order = sorted(tool_ts_map.keys(),
                   key=lambda t: tool_ts_map[t][0][0] if tool_ts_map[t] else 0)
    for tool in order:
        series = tool_ts_map[tool]
        ts = [s[0] for s in series]
        cpu_time = [s[1] for s in series]
        ax.plot(ts, cpu_time, label=tool, color=TOOL_COLORS.get(tool), linewidth=1.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Cumulative CPU Time (s)")
    ax.set_title("Cumulative CPU Time per Tool (user + system, summed across instances)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    fig.tight_layout()
    path = os.path.join(out_dir, "cpu_time.png")
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


def _write_markdown(csv_path, report_dir, stats, plots, total_samples,
                    system_memory_mb=None, peak_total_uss_mb=None):
    """Write the markdown summary report."""
    md_path = os.path.join(report_dir, "resource_report.md")
    has_pss = any("agg_peak_pss_mb" in s for s in stats.values())

    lines = []
    a = lines.append

    a("# Resource Monitor Report")
    a("")
    a(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    a(f"**Total samples:** {total_samples:,}")
    a(f"**Source:** `{os.path.basename(csv_path)}`")
    if system_memory_mb:
        a(f"**System memory:** {system_memory_mb:,.0f} MB")
    if peak_total_uss_mb is not None:
        pct = (peak_total_uss_mb / system_memory_mb * 100) if system_memory_mb else 0
        a(f"**Peak total USS:** {peak_total_uss_mb:,.0f} MB"
          + (f" ({pct:.1f}% of system memory)" if system_memory_mb else ""))
    a("")
    a("> **USS** (Unique Set Size) = private memory per process.  ")
    a("> **RSS** includes shared/mmap pages counted in every process — USS avoids this double-counting.")
    a("")

    # Summary table
    a("## Summary Statistics")
    a("")
    hdr = "| Tool | Instances | Samples | **Peak USS/inst (MB)** | **Aggr Peak USS (MB)** | Aggr Peak RSS (MB) |"
    sep = "|------|-----------|---------|----------------------:|-----------------------:|--------------------:|"
    if has_pss:
        hdr += " Aggr Peak PSS (MB) |"
        sep += "--------------------:|"
    hdr += " Peak CPU% | CPU User (s) | CPU Sys (s) | **CPU Total (s)** | Avg USS (MB) | Avg CPU% | Duration (s) |"
    sep += "----------:|-------------:|------------:|------------------:|-------------:|---------:|-------------:|"
    a(hdr)
    a(sep)
    for tool in sorted(stats.keys()):
        s = stats[tool]
        row = (f"| {tool} | {s['n_instances']} | {s['n_samples']} | "
               f"**{s['single_peak_uss_mb']:.1f}** | **{s['agg_peak_uss_mb']:.1f}** | "
               f"{s['agg_peak_rss_mb']:.1f} |")
        if has_pss:
            row += f" {s.get('agg_peak_pss_mb', 0):.1f} |"
        row += (f" {s['peak_cpu']:.1f}% | {s['total_cpu_user_s']:.1f} | "
                f"{s['total_cpu_sys_s']:.1f} | **{s['total_cpu_time_s']:.1f}** | "
                f"{s['avg_uss_mb']:.1f} | "
                f"{s['avg_cpu']:.1f}% | {s['duration_s']:.1f} |")
        a(row)
    a("")

    # Plots (if generated)
    if plots.get("agg_uss"):
        a("## Aggregate USS — Private Memory per Tool")
        a("")
        a(f"![Aggregate USS]({os.path.basename(plots['agg_uss'])})")
        a("")

    if plots.get("agg_rss_vs_uss"):
        a("## RSS vs USS — Shared Memory Overhead")
        a("")
        a(f"![RSS vs USS]({os.path.basename(plots['agg_rss_vs_uss'])})")
        a("")

    if plots.get("total"):
        a("## Total Private Memory Footprint (stacked)")
        a("")
        a(f"![Total Footprint]({os.path.basename(plots['total'])})")
        a("")

    if plots.get("agg_cpu"):
        a("## Aggregate CPU%")
        a("")
        a(f"![Aggregate CPU]({os.path.basename(plots['agg_cpu'])})")
        a("")

    if plots.get("cpu_time"):
        a("## Cumulative CPU Time (user + system)")
        a("")
        a(f"![CPU Time]({os.path.basename(plots['cpu_time'])})")
        a("")

    if plots.get("instances"):
        a("## Active Instance Count")
        a("")
        a(f"![Instance Count]({os.path.basename(plots['instances'])})")
        a("")

    a("---")
    a(f"*Source: `{csv_path}`*")

    with open(md_path, "w") as f:
        f.write("\n".join(lines))

    return md_path
