#!/usr/bin/env python
"""Building-packing gulmc -> fmpy parity + performance harness.

Sweeps a range of per-location building counts and, for each, builds an Oasis input set two
ways and runs the pure-Python loss pipeline end to end:

  * disaggregated : one item per building (legacy `do_disaggregation`)
  * packed        : one item per (location, peril, coverage_type) with N buildings multiplexed
                    into the sample dimension (`building_packing`)

For every (N, mode) it generates files, runs gulmc then fmpy as isolated subprocesses, and
records wall time, peak RSS and output sizes. Disaggregation is skipped above --disagg-max
(its item count, and therefore memory/time, grows O(N)).

Parity: the random samples of the two modes differ by construction (disaggregated buildings of
a location share one group seed -> identical draws; packed buildings get independent draws), so
only the draw-independent analytical aggregates are asserted equal: total post-FM mean (sidx -1),
TIV (sidx -3) and max loss (sidx -5).

The model static (footprint / vulnerability / damage bins) defaults to tests/assets/test_model_1;
point --static at any gulmc model. The OED location/account/keys are synthesised to be consistent
with that static.

What the numbers show
--------------------
Packing removes *per-item* overhead — item count, items.bin, the FM tree, and stream item headers
all stay flat as N grows, while disaggregation grows them O(N) (and its gulmc time with them). It
does NOT remove the *per-sample* payload: gul.bin bytes and gulmc's per-building loss / RNG buffers
still scale with N x sample_size, so peak RSS and stream size grow with N even when packed.

At very large N the analytical mean drifts slightly (oasis_float is float32; summing ~N building
values accumulates rounding) — a precision effect, not a correctness bug.

Run examples
------------
    python scripts/building_packing_harness.py --buildings 1,100,10000,1000000 --sample-size 10
    python scripts/building_packing_harness.py --static /path/to/model/static --keep --json-out out.json

stdout carries the report; stderr carries pipeline logging and ods_tools progress bars.
"""
import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DEFAULT_STATIC = REPO / "tests/assets/test_model_1/static"

# OED fixtures consistent with tests/assets/test_model_1 static (areaperil 154/54, vuln 8/2/11/5)
LOC_HEADER = ("PortNumber,AccNumber,LocNumber,IsTenant,BuildingID,CountryCode,Latitude,Longitude,"
              "StreetAddress,PostalCode,OccupancyCode,ConstructionCode,LocPerilsCovered,LocPeril,BuildingTIV,"
              "OtherTIV,ContentsTIV,BITIV,LocCurrency,NumberOfBuildings,IsAggregate,"
              "LocDedCode1Building,LocDedType1Building,LocDed1Building,OEDVersion")
LOC_ROW = ("1,A11111,10002082046,1,1,GB,52.76698052,-0.895469856,1 ABINGDON ROAD,LE13 0HL,1050,5000,"
           "WW1,WW1,220000,0,0,0,GBP,{n},{agg},0,0,{ded},latest version")
ACC = ("PortNumber,AccNumber,AccCurrency,PolNumber,PolPerilsCovered,PolPeril,PolInceptionDate,"
       "PolExpiryDate,LayerNumber,LayerParticipation,LayerLimit,LayerAttachment,OEDVersion\n"
       "1,A11111,GBP,Layer1,WW1,WW1,2018-01-01,2018-12-31,1,1.0,5000000,0,latest version\n")
KEYS = ("LocID,PerilID,CoverageTypeID,AreaPerilID,VulnerabilityID\n"
        "1,WSS,1,154,8\n1,WTC,1,54,2\n1,WSS,3,154,11\n1,WTC,3,54,5\n")
MODEL_SETTINGS = '{"version": "3", "model_settings": {}, "lookup_settings": {}, "model_default_samples": 10}'

ITEMS_ITEMSIZE = 20  # item_id i4 + coverage_id u4 + areaperil i4 + vuln i4 + group u4


def write_oed(d, n_buildings, loc_ded=0.0, is_aggregate=0):
    (d / "location.csv").write_text(
        LOC_HEADER + "\n" + LOC_ROW.format(n=n_buildings, ded=loc_ded, agg=is_aggregate) + "\n")
    (d / "account.csv").write_text(ACC)
    (d / "keys.csv").write_text(KEYS)
    (d / "model_settings.json").write_text(MODEL_SETTINGS)


def generate_files(run_dir, n_buildings, mode, oed):
    """Generate the Oasis input set into run_dir/input via the real GenerateFiles pipeline."""
    from oasislmf.manager import OasisManager
    input_dir = run_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    t = time.time()
    OasisManager().generate_files(
        oasis_files_dir=str(input_dir),
        oed_location_csv=str(oed / "location.csv"),
        oed_accounts_csv=str(oed / "account.csv"),
        keys_data_path=str(oed / "keys.csv"),
        model_settings_json=str(oed / "model_settings.json"),
        building_packing=(mode == "packed"),
        do_disaggregation=(mode == "disagg"),
    )
    gen_wall = time.time() - t
    n_items = (input_dir / "items.bin").stat().st_size // ITEMS_ITEMSIZE
    return gen_wall, n_items


SUBPROC_MARKER = "HARNESS_RESULT "


def _run_subprocess(code, label):
    """Run python -c code; return (result_dict_or_None, ok, tail). result printed after marker."""
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    result = None
    for line in proc.stdout.splitlines():
        if line.startswith(SUBPROC_MARKER):
            result = json.loads(line[len(SUBPROC_MARKER):])
    ok = proc.returncode == 0 and result is not None
    tail = (proc.stderr or proc.stdout)[-600:]
    return result, ok, tail


GULMC_CODE = """
import time, resource, json
from oasislmf.pytools.gulmc import manager
t = time.time()
manager.run(run_dir={run!r}, ignore_file_type=set(), sample_size={S}, loss_threshold=0.0,
            alloc_rule=0, debug=0, random_generator=0, file_in={ev!r}, file_out={out!r},
            peril_filter=[], data_server=None, ignore_correlation=False, ignore_haz_correlation=False,
            effective_damageability=False, max_cached_vuln_cdf_size_MB=200, dynamic_footprint=False)
print("{marker}" + json.dumps({{"wall": time.time() - t,
      "rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}}))
"""

FMPY_CODE = """
import time, resource, json
from oasislmf.pytools.fm import manager
manager.run(create_financial_structure_files=True, allocation_rule={A}, static_path={static!r})
t = time.time()
manager.run(create_financial_structure_files=False, allocation_rule={A}, files_in=[{gul!r}],
            files_out=[{out!r}], net_loss=None, storage_method="sparse", static_path={static!r},
            low_memory=False, sort_output=False)
print("{marker}" + json.dumps({{"wall": time.time() - t,
      "rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}}))
"""


def run_gulmc(run_dir, sample_size):
    code = GULMC_CODE.format(run=str(run_dir), S=sample_size, ev=str(run_dir / "input" / "events.bin"),
                             out=str(run_dir / "gul.bin"), marker=SUBPROC_MARKER)
    return _run_subprocess(code, "gulmc")


def run_fmpy(run_dir, alloc_rule):
    code = FMPY_CODE.format(A=alloc_rule, static=str(run_dir / "input"), gul=str(run_dir / "gul.bin"),
                            out=str(run_dir / "fm.bin"), marker=SUBPROC_MARKER)
    return _run_subprocess(code, "fmpy")


def parse_loss_stream(path):
    """Sum a LOSS stream by special sidx; return totals and the sample mean."""
    raw = Path(path).read_bytes()
    max_sidx = int(np.frombuffer(raw[4:8], dtype="<i4")[0])  # logical sample size S
    pos = 8  # stream header (type + max_sidx)
    totals = {"mean": 0.0, "tiv": 0.0, "max": 0.0}
    per_sample = {}  # sidx -> portfolio total loss for that sample (summed over output items)
    n_out = 0
    while pos + 8 <= len(raw):
        _event, _out = np.frombuffer(raw[pos:pos + 8], dtype="<i4")
        pos += 8
        n_out += 1
        while pos + 8 <= len(raw):
            sidx = int(np.frombuffer(raw[pos:pos + 4], dtype="<i4")[0])
            loss = float(np.frombuffer(raw[pos + 4:pos + 8], dtype="<f4")[0])
            pos += 8
            if sidx == 0:
                break
            if sidx == -1:
                totals["mean"] += loss
            elif sidx == -3:
                totals["tiv"] += loss
            elif sidx == -5:
                totals["max"] += loss
            elif sidx > 0:
                per_sample[sidx] = per_sample.get(sidx, 0.0) + loss
    # per-sample portfolio total across ALL S samples (missing sidx == 0 loss). Averaging over the
    # full S (not just populated samples) is essential: correlated (disagg) runs concentrate loss in
    # fewer, larger samples, so a populated-only mean would be badly biased.
    vals = np.zeros(max(max_sidx, 1), dtype=np.float64)
    for sidx, loss in per_sample.items():
        if 1 <= sidx <= vals.size:
            vals[sidx - 1] = loss
    totals["sample_total"] = float(vals.sum())
    totals["sample_mean"] = float(vals.mean())
    totals["sample_std"] = float(vals.std())
    totals["n_output_items"] = n_out
    return totals


def fmt_bytes(b):
    b = float(b)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if b < 1024 or unit == "TB":
            return f"{b:.0f}{unit}" if unit == "B" else f"{b:.1f}{unit}"
        b /= 1024


def run_cell(workdir, static, n, mode, sample_size, alloc_rule, oed, keep):
    run_dir = workdir / f"{mode}_N{n}"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    rec = {"n_buildings": n, "mode": mode, "ok": False}
    try:
        gen_wall, n_items = generate_files(run_dir, n, mode, oed)
        # link static + a single-event stream (event 1)
        (run_dir / "static").symlink_to(static)
        np.array([1], dtype="i4").tofile(run_dir / "input" / "events.bin")
        rec.update(gen_wall=gen_wall, n_items=int(n_items),
                   items_bytes=(run_dir / "input" / "items.bin").stat().st_size,
                   corr_bytes=(run_dir / "input" / "correlations.bin").stat().st_size)

        g, gok, gtail = run_gulmc(run_dir, sample_size)
        if not gok:
            rec["error"] = "gulmc failed: " + gtail
            return rec
        rec.update(gul_wall=g["wall"], gul_rss_kb=g["rss_kb"],
                   gul_bytes=(run_dir / "gul.bin").stat().st_size)

        f, fok, ftail = run_fmpy(run_dir, alloc_rule)
        if not fok:
            rec["error"] = "fmpy failed: " + ftail
            return rec
        rec.update(fm_wall=f["wall"], fm_rss_kb=f["rss_kb"],
                   fm_bytes=(run_dir / "fm.bin").stat().st_size)
        rec["totals"] = parse_loss_stream(run_dir / "fm.bin")
        rec["ok"] = True
        return rec
    finally:
        if not keep and run_dir.exists():
            shutil.rmtree(run_dir, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--static", default=str(DEFAULT_STATIC), help="gulmc model static dir")
    ap.add_argument("--buildings", default="1,10,100,1000,10000,100000,1000000",
                    help="comma-separated per-location building counts")
    ap.add_argument("--disagg-max", type=int, default=10000,
                    help="skip disaggregation above this N (item count grows O(N))")
    ap.add_argument("--sample-size", type=int, default=10)
    ap.add_argument("--alloc-rule", type=int, default=2, help="fmpy back-allocation rule")
    ap.add_argument("--loc-ded", type=float, default=0.0,
                    help="per-building location deductible (LocDed1Building). Non-zero makes the "
                         "location non-aggregatable, so packed mode disaggregates it (unless --is-aggregate).")
    ap.add_argument("--is-aggregate", type=int, default=0,
                    help="set IsAggregate=1 so the location packs even with location terms")
    ap.add_argument("--keep", action="store_true", help="keep run directories")
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    static = Path(args.static).resolve()
    buildings = [int(x) for x in args.buildings.split(",")]
    workdir = Path(args.workdir) if args.workdir else Path(tempfile.gettempdir()) / "bp_harness_runs"
    workdir.mkdir(parents=True, exist_ok=True)
    oed = workdir / "oed"
    oed.mkdir(exist_ok=True)

    records = []
    for n in buildings:
        write_oed(oed, n, loc_ded=args.loc_ded, is_aggregate=args.is_aggregate)
        modes = ["packed"]
        if n <= args.disagg_max:
            modes = ["disagg", "packed"]
        for mode in modes:
            print(f"[run] N={n} mode={mode} ...", flush=True)
            rec = run_cell(workdir, static, n, mode, args.sample_size, args.alloc_rule, oed, args.keep)
            records.append(rec)
            if not rec["ok"]:
                print(f"   FAILED: {rec.get('error', '?')[:200]}", flush=True)

    # ---- report ----
    print("\n=== gulmc -> fmpy building-packing harness ===")
    aggregatable = bool(args.is_aggregate) or args.loc_ded == 0
    print(f"static={static}  S={args.sample_size}  alloc_rule={args.alloc_rule}  "
          f"loc_ded={args.loc_ded}  is_aggregate={args.is_aggregate}  "
          f"-> locations {'AGGREGATABLE (packed)' if aggregatable else 'NON-aggregatable (packed mode disaggregates)'}\n")
    hdr = f"{'N':>9} {'mode':>7} {'items':>9} {'items.bin':>10} {'gul.bin':>10} " \
        f"{'guls':>7} {'gulMB':>7} {'fms':>7} {'fmMB':>7} {'mean_tot':>14}"
    print(hdr)
    print("-" * len(hdr))
    for r in records:
        if not r["ok"]:
            print(f"{r['n_buildings']:>9} {r['mode']:>7}  FAILED")
            continue
        print(f"{r['n_buildings']:>9} {r['mode']:>7} {r['n_items']:>9} "
              f"{fmt_bytes(r['items_bytes']):>10} {fmt_bytes(r['gul_bytes']):>10} "
              f"{r['gul_wall']:>7.2f} {r['gul_rss_kb'] / 1024:>7.0f} "
              f"{r['fm_wall']:>7.2f} {r['fm_rss_kb'] / 1024:>7.0f} "
              f"{r['totals']['mean']:>14.2f}")

    # ---- parity (analytical aggregates) ----
    print("\n=== parity (packed vs disagg; analytical aggregates) ===")
    by_n = {}
    for r in records:
        if r["ok"]:
            by_n.setdefault(r["n_buildings"], {})[r["mode"]] = r["totals"]
    any_checked = False
    for n, modes in sorted(by_n.items()):
        if "packed" in modes and "disagg" in modes:
            any_checked = True
            p, d = modes["packed"], modes["disagg"]
            ok = all(abs(p[k] - d[k]) <= 1e-3 * max(1.0, abs(d[k])) for k in ("mean", "tiv", "max"))
            verdict = "OK" if ok else "MISMATCH"
            print(f"  N={n:>7}: mean p={p['mean']:.2f} d={d['mean']:.2f} | "
                  f"tiv p={p['tiv']:.2f} d={d['tiv']:.2f} | max p={p['max']:.2f} d={d['max']:.2f}  -> {verdict}")
    if not any_checked:
        print("  (no N ran in both modes)")

    # ---- statistical / modelling difference (the random part) ----
    # Old disaggregation gives all N buildings of a location the same group seed -> identical draws
    # (perfectly correlated). Packing draws each building independently. So the analytical mean is
    # the same, but the spread of the portfolio loss across samples differs: packed should show
    # diversification (smaller std). With independent vs fully-correlated buildings the std ratio
    # tends to ~sqrt(N).
    print("\n=== sample distribution (portfolio loss across samples) ===")
    print(f"  {'N':>7} {'mode':>7} {'sample_mean':>14} {'sample_std':>14}")
    for n, modes in sorted(by_n.items()):
        for mode in ("disagg", "packed"):
            if mode in modes:
                t = modes[mode]
                print(f"  {n:>7} {mode:>7} {t['sample_mean']:>14.2f} {t['sample_std']:>14.2f}")
        if "packed" in modes and "disagg" in modes and modes["packed"]["sample_std"] > 0:
            ratio = modes["disagg"]["sample_std"] / modes["packed"]["sample_std"]
            print(f"  {'':>7} {'-> std ratio disagg/packed':>30} = {ratio:.2f}  (sqrt(N)={n**0.5:.2f})")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(records, indent=2))
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
