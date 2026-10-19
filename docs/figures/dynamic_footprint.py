"""Draw the figures of docs/source/explanation/dynamic-footprint.rst.

Every merged or interpolated curve is computed by FootprintParquetDynamic._build_footprint on the
page's worked examples, so the figures show what the code does. Each figure is written twice, for
the documentation's light and dark themes.

Run from the repository root: python docs/figures/dynamic_footprint.py
"""
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from oasislmf.pytools.getmodel.footprint import FootprintParquetDynamic  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / 'source' / '_static' / 'images' / 'dynamic_footprint'
# fixed ids in the SVGs, so that regenerating unchanged figures leaves the files unchanged
plt.rcParams['svg.hashsalt'] = 'oasislmf-dynamic-footprint'

# validated categorical slots 1-4 and chart ink, per theme
THEMES = {
    'light': {'series': ['#2a78d6', '#eb6834', '#1baf7a', '#eda100'],
              'text': '#0b0b0b', 'secondary': '#52514e', 'muted': '#898781', 'grid': '#e1e0d9', 'axis': '#c3c2b7'},
    'dark': {'series': ['#3987e5', '#d95926', '#199e70', '#c98500'],
             'text': '#f0efec', 'secondary': '#c3c2b7', 'muted': '#898781', 'grid': '#2c2c2a', 'axis': '#383835'},
}


def build(rows, sections, section_merge='max'):
    """The footprint _build_footprint gives areaperil 1: (intensity, probability), lowest intensity first."""
    hazard = pd.DataFrame(rows, columns=['section_id', 'areaperil_id', 'return_period', 'intensity', 'probability'])
    event = pd.DataFrame(sections, columns=['section_id', 'rp_from', 'rp_to', 'interpolation']).assign(event_id=1)
    builder = FootprintParquetDynamic.__new__(FootprintParquetDynamic)
    builder.section_merge = section_merge
    footprint = builder._build_footprint(hazard, event)
    return sorted((int(r['intensity']), float(r['probability'])) for r in footprint if r['areaperil_id'] == 1)


def flat(section, values, probabilities):
    """A section whose distribution is the same at both of its brackets, so interpolation keeps it."""
    return [(section, 1, rp, v, p) for rp in (10, 100) for v, p in zip(values, probabilities)]


def quantile_steps(distribution):
    """x (cumulative probability) and y (intensity) of a quantile function, for a 'post' step plot."""
    levels = np.cumsum([p for _, p in distribution])
    x = np.concatenate(([0.0], levels))
    y = [v for v, _ in distribution] + [distribution[-1][0]]
    return x, y


def cdf_steps(distribution, x_min, x_max):
    """x (intensity) and y (P(intensity <= x)) of a distribution's CDF, for a 'post' step plot."""
    xs = [x_min] + [v for v, _ in distribution] + [x_max]
    ys = [0.0] + list(np.cumsum([p for _, p in distribution])) + [1.0]
    return xs, ys


def style(ax, theme, xlabel, ylabel):
    ink = THEMES[theme]
    ax.set_facecolor('none')
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(ink['axis'])
    ax.tick_params(colors=ink['muted'], labelcolor=ink['secondary'], labelsize=9)
    ax.grid(True, color=ink['grid'], linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel(xlabel, color=ink['secondary'], fontsize=10)
    ax.set_ylabel(ylabel, color=ink['secondary'], fontsize=10)


def figure(theme):
    fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=100)
    fig.patch.set_alpha(0)
    return fig, ax


def interpolation_figure(theme):
    """Two brackets with different numbers of realisations, paired by cumulative probability."""
    rp_from = [(10, 0.5), (20, 0.5)]
    rp_to = [(16, 0.25), (20, 0.5), (24, 0.25)]
    rows = [(1, 1, 10, v, p) for v, p in rp_from] + [(1, 1, 50, v, p) for v, p in rp_to]
    result = build(rows, [(1, 10, 50, 0.5)])
    series = THEMES[theme]['series']

    fig, ax = figure(theme)
    style(ax, theme, 'cumulative probability (level)', 'intensity')
    for level in (0.25, 0.5, 0.75):
        ax.axvline(level, color=THEMES[theme]['muted'], linewidth=0.8, linestyle=':')
    curves = [(rp_from, series[0], '-', 1.6, 'rp_from (10 year): 10, 20'),
              (rp_to, series[1], '-', 1.6, 'rp_to (50 year): 16, 20, 24'),
              (result, series[2], '--', 2.4, 'interpolated half-way: 13, 15, 20, 22')]
    for distribution, colour, line, width, name in curves:
        x, y = quantile_steps(distribution)
        ax.step(x, y, where='post', color=colour, linestyle=line, linewidth=width, label=name)
    for k, (value, _) in enumerate(result):
        ax.annotate(str(value), (0.125 + 0.25 * k, value), textcoords='offset points', xytext=(0, 6),
                    ha='center', color=THEMES[theme]['text'], fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(6, 27)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    legend(ax, theme, loc='upper left')
    return fig


def max_together_figure(theme):
    """Two uncertain sections merged by max: their upper envelope, level by level."""
    section_1 = [(140, 0.25), (160, 0.5), (180, 0.25)]
    section_2 = [(120, 0.25), (150, 0.5), (190, 0.25)]
    rows = flat(1, *zip(*section_1)) + flat(2, *zip(*section_2))
    merged = build(rows, [(1, 10, 100, 0.5), (2, 10, 100, 0.5)], 'max')
    series = THEMES[theme]['series']

    fig, ax = figure(theme)
    style(ax, theme, 'cumulative probability (level)', 'intensity')
    for level in (0.25, 0.75):
        ax.axvline(level, color=THEMES[theme]['muted'], linewidth=0.8, linestyle=':')
    for distribution, colour, line, width, name in [
            (section_1, series[0], '-', 1.6, 'section 1: 140, 160, 180'),
            (section_2, series[1], '-', 1.6, 'section 2: 120, 150, 190'),
            (merged, series[2], '--', 2.6, 'max (together): 140, 160, 190')]:
        x, y = quantile_steps(distribution)
        ax.step(x, y, where='post', color=colour, linestyle=line, linewidth=width, label=name)
    ax.set_xlim(0, 1)
    ax.set_ylim(110, 200)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    legend(ax, theme, loc='upper left')
    return fig


def max_options_figure(theme):
    """P(intensity <= x) of two sections and of their maximum, together and independent."""
    section_1 = [(140, 0.25), (160, 0.5), (180, 0.25)]
    section_2 = [(120, 0.25), (150, 0.5), (190, 0.25)]
    rows = flat(1, *zip(*section_1)) + flat(2, *zip(*section_2))
    sections = [(1, 10, 100, 0.5), (2, 10, 100, 0.5)]
    together = build(rows, sections, 'max')
    independent = build(rows, sections, 'max_independent')
    series = THEMES[theme]['series']

    fig, ax = figure(theme)
    style(ax, theme, 'intensity x', 'P(intensity ≤ x)')
    for distribution, colour, line, width, name in [
            (section_1, series[0], '-', 1.4, 'section 1'),
            (section_2, series[1], '-', 1.4, 'section 2'),
            (together, series[2], '--', 2.4, 'max'),
            (independent, series[3], '-.', 2.4, 'max_independent')]:
        x, y = cdf_steps(distribution, 110, 200)
        ax.step(x, y, where='post', color=colour, linestyle=line, linewidth=width, label=name)
    # the tail: just below 180, P(intensity < 180) is 0.75 together but 0.5625 independent
    ink = THEMES[theme]
    ax.axvline(180, color=ink['muted'], linewidth=0.8, linestyle=':')
    for y, colour in ((0.75, series[2]), (0.5625, series[3])):
        ax.plot([179.3], [y], marker='o', markersize=7, color=colour, markeredgecolor=ink['text'], markeredgewidth=0.8)
    ax.annotate('P(intensity ≥ 180)\n  together: 1 − 0.75 = 0.25\n  independent: 1 − 0.5625 = 0.4375',
                (179.3, 0.5625), xytext=(112, 0.62), color=ink['text'], fontsize=9, va='bottom',
                arrowprops=dict(arrowstyle='->', color=ink['muted'], linewidth=0.8))
    ax.set_xlim(110, 200)
    ax.set_ylim(0, 1.02)
    legend(ax, theme, loc='lower right')
    return fig


def legend(ax, theme, **kwargs):
    ink = THEMES[theme]
    lg = ax.legend(frameon=False, fontsize=9, labelcolor=ink['text'], handlelength=2.6, **kwargs)
    return lg


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, draw in [('interpolation', interpolation_figure), ('max_together', max_together_figure),
                       ('max_options', max_options_figure)]:
        for theme in THEMES:
            fig = draw(theme)
            fig.tight_layout()
            fig.savefig(OUT / f'{name}_{theme}.svg', transparent=True, metadata={'Date': None})
            plt.close(fig)


if __name__ == '__main__':
    main()
