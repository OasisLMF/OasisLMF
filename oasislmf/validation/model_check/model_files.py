__all__ = [
    'VulnerabilityData',
    'FootprintSummary',
    'check_damage_bins',
    'load_vulnerability',
    'check_vulnerability',
    'scan_footprint',
    'check_event_files',
    'check_aggregate_vulnerability',
    'read_lossfactor_amplification_ids',
]

import json
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ...pytools.common.data import areaperil_int, oasis_int, vulnerability_dtype
from ...pytools.common.input_files import read_occurrence_bin, read_periods, read_returnperiods
from ...pytools.getmodel.common import EventIndexBin_dtype, footprint_index_filename
from ...pytools.getmodel.footprint import Footprint
from ...pytools.getmodel.manager import VulnerabilityIndex_dtype, VulnerabilityRow_dtype, get_damage_bins
from ...pytools.getmodel.vulnerability import parquetvulnerability_meta_filename, vulnerability_dataset
from ...pytools.gulmc.aggregate import read_aggregate_vulnerability, read_vulnerability_weights
from ...pytools.pla.structure import read_lossfactors
from .report import ERROR, INFO, WARNING

PROB_TOLERANCE = 1e-4
VALID_DAMAGE_TYPES = {0, 1, 2, 3}


class _Collector:
    """Count occurrences of a problem and keep the first few examples."""

    def __init__(self, max_examples=10):
        self.count = 0
        self.examples = []
        self.max_examples = max_examples

    def add(self, examples, count=None):
        count = len(examples) if count is None else count
        self.count += count
        room = self.max_examples - len(self.examples)
        if room > 0:
            self.examples.extend(list(examples)[:room])

    def report(self, report, level, check, message):
        if self.count:
            report.add(level, check, message, self.examples, count=self.count)
        else:
            report.ok(check)


@dataclass
class VulnerabilityData:
    source: str
    available_ids: np.ndarray
    num_damage_bins: int = None
    rows: np.ndarray = None
    num_intensity_bins: int = None


@dataclass
class FootprintSummary:
    num_intensity_bins: int = None
    events_checked: int = 0
    sampled: bool = False
    missing_events: list = field(default_factory=list)
    hit_areaperils: np.ndarray = None


def check_damage_bins(report, storage):
    try:
        bins = get_damage_bins(storage)
    except FileNotFoundError as e:
        report.error('damage_bin_dict.exists', str(e))
        return None

    n = len(bins)
    row_ids = np.arange(1, n + 1)
    misplaced = bins['bin_index'] != row_ids
    report.missing('damage_bin_dict.contiguous',
                   'bin_index is not 1..N in row order; the runtime indexes damage bins by row position',
                   [f'row {r}: bin_index {b}' for r, b in zip(row_ids[misplaced], bins['bin_index'][misplaced])])
    report.missing('damage_bin_dict.from_to', 'bin_from is greater than bin_to',
                   bins['bin_index'][bins['bin_from'] > bins['bin_to']].tolist())
    outside = (bins['interpolation'] < bins['bin_from'] - PROB_TOLERANCE) | (bins['interpolation'] > bins['bin_to'] + PROB_TOLERANCE)
    report.missing('damage_bin_dict.interpolation', 'interpolation lies outside [bin_from, bin_to]',
                   bins['bin_index'][outside].tolist())
    report.missing('damage_bin_dict.overlap', 'bin_from is below the previous bin_to (bins overlap)',
                   bins['bin_index'][1:][bins['bin_from'][1:] < bins['bin_to'][:-1] - PROB_TOLERANCE].tolist(), level=WARNING)
    report.missing('damage_bin_dict.damage_type', f'damage_type is not one of {sorted(VALID_DAMAGE_TYPES)}',
                   bins['bin_index'][~np.isin(bins['damage_type'], list(VALID_DAMAGE_TYPES))].tolist(), level=WARNING)
    if n and not (bins['bin_from'][0] == 0 and bins['bin_to'][0] == 0):
        report.warning('damage_bin_dict.zero_bin',
                       'first damage bin is not the point bin [0, 0]; an all-zero vulnerability column then gives inf/nan instead of zero loss')
    else:
        report.ok('damage_bin_dict.zero_bin')
    return bins


def _rows_from_idx(vulns_bin, vulns_idx, header_size):
    row_size = VulnerabilityRow_dtype.itemsize
    parts = []
    for entry in vulns_idx:
        start = (int(entry['offset']) - header_size) // row_size
        part = np.empty(int(entry['size']) // row_size, dtype=vulnerability_dtype)
        chunk = vulns_bin[start: start + len(part)]
        part['vulnerability_id'] = entry['vulnerability_id']
        for col in ('intensity_bin_id', 'damage_bin_id', 'probability'):
            part[col] = chunk[col]
        parts.append(part)
    return np.concatenate(parts) if parts else np.empty(0, dtype=vulnerability_dtype)


def _rows_from_parquet(df, num_damage_bins, num_intensity_bins):
    parts = []
    for vuln_id, arr in zip(df['vulnerability_id'].to_numpy(), df['vuln_array'].to_numpy()):
        arr = np.asarray(arr).reshape(num_damage_bins, num_intensity_bins)
        damage_i, intensity_i = np.nonzero(arr)
        part = np.empty(len(damage_i), dtype=vulnerability_dtype)
        part['vulnerability_id'] = vuln_id
        part['intensity_bin_id'] = intensity_i + 1
        part['damage_bin_id'] = damage_i + 1
        part['probability'] = arr[damage_i, intensity_i]
        parts.append(part)
    return np.concatenate(parts) if parts else np.empty(0, dtype=vulnerability_dtype)


def load_vulnerability(report, static_dir, vuln_ids=None):
    """Load vulnerability rows for ``vuln_ids`` (all rows if None) from whichever format a run would pick."""
    header_size = 4
    dataset_fp = os.path.join(static_dir, vulnerability_dataset)
    bin_fp = os.path.join(static_dir, 'vulnerability.bin')
    idx_fp = os.path.join(static_dir, 'vulnerability.idx')
    csv_fp = os.path.join(static_dir, 'vulnerability.csv')
    wanted = None if vuln_ids is None else np.unique(np.asarray(vuln_ids, dtype=np.int64))

    if os.path.exists(dataset_fp):
        with open(os.path.join(static_dir, parquetvulnerability_meta_filename)) as f:
            meta = json.load(f)
        available = np.unique(pd.read_parquet(dataset_fp, columns=['vulnerability_id'])['vulnerability_id'].to_numpy())
        filters = None if wanted is None else [('vulnerability_id', 'in', wanted.tolist())]
        df = pd.read_parquet(dataset_fp, columns=['vulnerability_id', 'vuln_array'], filters=filters)
        rows = _rows_from_parquet(df, meta['num_damage_bins'], meta['num_intensity_bins'])
        return VulnerabilityData(vulnerability_dataset, available, meta['num_damage_bins'], rows, meta['num_intensity_bins'])

    if os.path.exists(bin_fp):
        num_damage_bins = int(np.fromfile(bin_fp, dtype=np.int32, count=1)[0])
        if os.path.exists(idx_fp):
            vulns_idx = np.fromfile(idx_fp, dtype=VulnerabilityIndex_dtype)
            available = np.unique(vulns_idx['vulnerability_id'])
            if ((vulns_idx['original_size'] > 0) & (vulns_idx['original_size'] != vulns_idx['size'])).any():
                report.error('vulnerability.format',
                             'vulnerability.idx describes compressed rows, which the loss kernels do not support')
                return VulnerabilityData('vulnerability.bin+idx', available, num_damage_bins)
            if wanted is not None:
                vulns_idx = vulns_idx[np.isin(vulns_idx['vulnerability_id'], wanted)]
            size_check = os.path.getsize(bin_fp) - header_size
            if size_check % VulnerabilityRow_dtype.itemsize:
                report.error('vulnerability.format', 'vulnerability.bin size is not a whole number of 12-byte rows but a '
                             'vulnerability.idx is present; is the idx stale next to a flat 16-byte-row file?')
                return VulnerabilityData('vulnerability.bin+idx', available, num_damage_bins)
            vulns_bin = np.memmap(bin_fp, dtype=VulnerabilityRow_dtype, offset=header_size, mode='r')
            return VulnerabilityData('vulnerability.bin+idx', available, num_damage_bins,
                                     _rows_from_idx(vulns_bin, vulns_idx, header_size))

        if (os.path.getsize(bin_fp) - header_size) % vulnerability_dtype.itemsize:
            report.error('vulnerability.format', f'vulnerability.bin size is not a whole number of {vulnerability_dtype.itemsize}-byte rows')
            return None
        vulns = np.memmap(bin_fp, dtype=vulnerability_dtype, offset=header_size, mode='r')
        available = np.unique(vulns['vulnerability_id'])
        rows = np.array(vulns if wanted is None else vulns[np.isin(vulns['vulnerability_id'], wanted)])
        return VulnerabilityData('vulnerability.bin', available, num_damage_bins, rows)

    if os.path.exists(csv_fp):
        rows = np.loadtxt(csv_fp, dtype=vulnerability_dtype, delimiter=',', skiprows=1, ndmin=1)
        available = np.unique(rows['vulnerability_id'])
        if wanted is not None:
            rows = rows[np.isin(rows['vulnerability_id'], wanted)]
        return VulnerabilityData('vulnerability.csv', available, int(rows['damage_bin_id'].max()) if len(rows) else 0, rows)

    report.error('vulnerability.exists', f'no vulnerability file (parquet dataset, bin or csv) found in {static_dir}')
    return None


def check_vulnerability(report, vuln, damage_bins, num_intensity_bins, zero_bin_is_point=True):
    rows = vuln.rows
    if rows is None or len(rows) == 0:
        return
    v, i, d, p = rows['vulnerability_id'], rows['intensity_bin_id'], rows['damage_bin_id'], rows['probability']

    def flag(mask, check, message, level=ERROR):
        if mask.any():
            report.add(level, check, message,
                       [f'vuln {a} intensity {b} damage {c}' for a, b, c in zip(v[mask][:10], i[mask][:10], d[mask][:10])],
                       count=int(mask.sum()))
        else:
            report.ok(check)

    flag(d < 1, 'vulnerability.damage_bin_min', 'damage_bin_id < 1 (index wraps to the last damage bin)')
    flag(d > vuln.num_damage_bins, 'vulnerability.damage_bin_header',
         f'damage_bin_id above the {vuln.num_damage_bins} damage bins declared in the vulnerability header (run raises)')
    flag(i < 1, 'vulnerability.intensity_bin_min', 'intensity_bin_id < 1 (index wraps to the highest intensity bin)')
    flag((p < 0) | (p > 1), 'vulnerability.probability_range', 'probability outside [0, 1]')
    if damage_bins is not None:
        flag(d > len(damage_bins), 'vulnerability.damage_bin_dict',
             f'damage_bin_id beyond the {len(damage_bins)} rows of damage_bin_dict (out-of-bounds read at run time)')
    if num_intensity_bins is not None:
        flag(i > num_intensity_bins, 'vulnerability.intensity_bin_max',
             f'intensity_bin_id above the footprint\'s {num_intensity_bins} intensity bins; these rows are silently ignored', WARNING)

    order = np.lexsort((d, i, v))
    sv, si, sd, sp = v[order], i[order], d[order], p[order].astype(np.float64)
    vi_change = (sv[1:] != sv[:-1]) | (si[1:] != si[:-1])
    dup = ~vi_change & (sd[1:] == sd[:-1])
    report.missing('vulnerability.duplicates', 'duplicate (vulnerability_id, intensity_bin_id, damage_bin_id) rows; the last one wins',
                   [f'vuln {a} intensity {b} damage {c}' for a, b, c in zip(sv[1:][dup], si[1:][dup], sd[1:][dup])])

    starts = np.r_[0, np.flatnonzero(vi_change) + 1]
    sums = np.add.reduceat(sp, starts)
    bad = np.abs(sums - 1) > PROB_TOLERANCE
    report.missing('vulnerability.probability_sum', f'probabilities per (vulnerability_id, intensity_bin_id) do not sum to 1 (tol {PROB_TOLERANCE})',
                   [f'vuln {a} intensity {b}: {s:.6f}' for a, b, s in zip(sv[starts][bad], si[starts][bad], sums[bad])])

    if num_intensity_bins is not None:
        in_range = (si[starts] >= 1) & (si[starts] <= num_intensity_bins)
        ids, covered = np.unique(sv[starts][in_range], return_counts=True)
        incomplete = ids[covered < num_intensity_bins]
        msg = f'vulnerability functions do not define all {num_intensity_bins} intensity bins; '
        if zero_bin_is_point:
            report.missing('vulnerability.intensity_coverage', msg + 'undefined intensities give zero loss', incomplete.tolist(), level=INFO)
        else:
            report.missing('vulnerability.intensity_coverage', msg + 'undefined intensities give inf/nan loss because damage bin 1 is not [0, 0]',
                           incomplete.tolist())


def _check_footprint_index(report, static_dir):
    idx_fp = os.path.join(static_dir, footprint_index_filename)
    if not os.path.exists(idx_fp):
        return
    idx = np.fromfile(idx_fp, dtype=EventIndexBin_dtype)
    unsorted = np.flatnonzero(idx['event_id'][1:] < idx['event_id'][:-1])
    report.missing('footprint.index_sorted', 'footprint.idx is not sorted by event_id; events after the break are never found (silent zero loss)',
                   idx['event_id'][unsorted + 1].tolist())
    report.missing('footprint.index_unique', 'footprint.idx has duplicate event_ids',
                   np.unique(idx['event_id'][1:][idx['event_id'][1:] == idx['event_id'][:-1]]).tolist())


def scan_footprint(report, model_storage, static_dir, event_ids, portfolio_areaperils=None, dynamic=False, max_events=None):
    summary = FootprintSummary()
    _check_footprint_index(report, static_dir)

    if event_ids is None:
        report.warning('footprint.events', 'no events file resolved, footprint contents not checked')
        return summary
    event_ids = np.asarray(event_ids)
    if max_events and len(event_ids) > max_events:
        event_ids = event_ids[np.linspace(0, len(event_ids) - 1, max_events).astype(np.int64)]
        summary.sampled = True
        report.info('footprint.sample', f'checked a sample of {max_events} events')

    noncontiguous, ap_zero, bad_intensity, bad_sum, dups = (_Collector() for _ in range(5))
    hits = []
    try:
        with Footprint.load(model_storage) as footprint:
            nib = summary.num_intensity_bins = int(footprint.num_intensity_bins)
            for event_id in event_ids:
                event = footprint.get_event(event_id)
                if event is None or len(event) == 0:
                    summary.missing_events.append(int(event_id))
                    continue
                summary.events_checked += 1
                ap, ib, pr = event['areaperil_id'], event['intensity_bin_id'], event['probability']
                starts = np.r_[0, np.flatnonzero(ap[1:] != ap[:-1]) + 1]
                unique_ap = np.unique(ap)
                if len(starts) != len(unique_ap):
                    noncontiguous.add([int(event_id)])
                if unique_ap[0] == 0:
                    ap_zero.add([int(event_id)])
                if not dynamic:
                    bad = (ib < 1) | (ib > nib)
                    if bad.any():
                        bad_intensity.add([f'event {event_id} areaperil {a} intensity {b}' for a, b in zip(ap[bad][:3], ib[bad][:3])], int(bad.sum()))
                sums = np.add.reduceat(pr.astype(np.float64), starts)
                bad = np.abs(sums - 1) > PROB_TOLERANCE
                if bad.any():
                    bad_sum.add([f'event {event_id} areaperil {a}: {s:.6f}' for a, s in zip(ap[starts][bad][:3], sums[bad][:3])], int(bad.sum()))
                order = np.lexsort((ib, ap))
                dup = (ap[order][1:] == ap[order][:-1]) & (ib[order][1:] == ib[order][:-1])
                if dup.any():
                    dups.add([f'event {event_id} areaperil {a}' for a in ap[order][1:][dup][:3]], int(dup.sum()))
                if portfolio_areaperils is not None:
                    hits.append(np.intersect1d(unique_ap, portfolio_areaperils, assume_unique=True))
    except Exception as e:
        report.error('footprint.load', f'{type(e).__name__}: {e}')
        return summary

    noncontiguous.report(report, ERROR, 'footprint.areaperil_contiguous',
                         'rows for an areaperil are not contiguous within an event (corrupts gulmc item buffers)')
    ap_zero.report(report, WARNING, 'footprint.areaperil_zero', 'events contain areaperil_id 0, which the kernels skip')
    if not dynamic:
        bad_intensity.report(report, ERROR, 'footprint.intensity_range',
                             f'intensity_bin_id outside 1..{summary.num_intensity_bins} (out-of-bounds read at run time)')
    bad_sum.report(report, ERROR, 'footprint.probability_sum',
                   f'probabilities per (event, areaperil) do not sum to 1 (tol {PROB_TOLERANCE})')
    dups.report(report, ERROR, 'footprint.duplicates', 'duplicate (areaperil_id, intensity_bin_id) rows within an event')
    report.missing('footprint.events_present', 'events in the selected event set have no footprint (they produce no loss)',
                   summary.missing_events, level=WARNING)
    if portfolio_areaperils is not None:
        summary.hit_areaperils = np.unique(np.concatenate(hits)) if hits else np.empty(0, dtype=areaperil_int)
    return summary


def check_event_files(report, run_files):
    event_ids = None
    input_dir = os.path.dirname(next(iter(run_files.values()))) if run_files else None
    if 'events' in run_files:
        event_ids = np.fromfile(run_files['events'], dtype=oasis_int)
        sorted_ids = np.sort(event_ids)
        report.missing('events.duplicates', 'duplicate event ids in the events file (losses counted twice)',
                       np.unique(sorted_ids[1:][sorted_ids[1:] == sorted_ids[:-1]]).tolist())
        report.missing('events.positive', 'event ids must be positive', event_ids[event_ids <= 0].tolist())
        if len(event_ids) == 0:
            report.error('events.empty', 'the events file is empty')

    if 'occurrence' in run_files:
        try:
            occ, _, _, no_of_periods = read_occurrence_bin(input_dir, 'occurrence.bin')
        except RuntimeError as e:
            report.error('occurrence.format', str(e))
            return event_ids
        bad = (occ['period_no'] < 1) | (occ['period_no'] > no_of_periods)
        report.missing('occurrence.period_range', f'period_no outside 1..{no_of_periods} (out-of-bounds writes in LEC/PLT)',
                       [f'event {e} period {p}' for e, p in zip(occ['event_id'][bad], occ['period_no'][bad])])
        if event_ids is not None:
            report.missing('occurrence.events_covered', 'events with no occurrence rows are absent from ALT/PLT/EPT outputs',
                           np.setdiff1d(event_ids, occ['event_id']).tolist(), level=WARNING)
        try:
            weights = read_periods(no_of_periods, input_dir)
            report.ok('periods.format')
            total = float(weights['weighting'].sum())
            if 'periods' in run_files and abs(total - 1) > PROB_TOLERANCE:
                report.warning('periods.weights_sum', f'period weights sum to {total:.6f}, not 1')
        except RuntimeError as e:
            report.error('periods.format', str(e))

    if 'returnperiods' in run_files:
        try:
            read_returnperiods(True, input_dir)
            report.ok('returnperiods.format')
        except RuntimeError as e:
            report.error('returnperiods.format', str(e))
    return event_ids


def check_aggregate_vulnerability(report, model_storage, vuln_available_ids, used_vuln_ids=None):
    """Return the aggregate id -> sub-vulnerability ids map (empty if the model has none)."""
    agg = read_aggregate_vulnerability(model_storage)
    weights = read_vulnerability_weights(model_storage)
    if agg is None:
        return {}
    if weights is None:
        report.error('aggregate_vulnerability.weights', 'aggregate_vulnerability is present but weights.bin/csv is missing (gulmc raises)')

    agg_map = {}
    for agg_id, sub_id in zip(agg['aggregate_vulnerability_id'], agg['vulnerability_id']):
        agg_map.setdefault(int(agg_id), []).append(int(sub_id))

    agg_ids = np.array(sorted(agg_map))
    report.missing('aggregate_vulnerability.id_collision',
                   'aggregate ids also exist as vulnerability ids; items using them are treated as aggregates',
                   np.intersect1d(agg_ids, vuln_available_ids).tolist(), level=WARNING)
    used = set(agg_map) if used_vuln_ids is None else set(agg_map) & {int(x) for x in used_vuln_ids}
    missing_used = sorted({s for a in used for s in agg_map[a]} - set(vuln_available_ids.tolist()))
    report.missing('aggregate_vulnerability.sub_ids',
                   'aggregate sub-vulnerability ids used by this portfolio are missing from the vulnerability file', missing_used)
    return agg_map


def read_lossfactor_amplification_ids(static_dir):
    fp = os.path.join(static_dir, 'lossfactors.bin')
    if not os.path.exists(fp):
        return None
    return np.unique(np.array([amp for _, amp in read_lossfactors(static_dir, set(['csv'])).keys()], dtype=np.int64))
