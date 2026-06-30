"""
Regression tests for IL ordering sensitivity (issue #2040).

Background
----------
Clients reported that two OED exposure files identical except for row order
produced wildly different insured losses (GUL AALs were identical).

Root causes in release/2.4.13 (the bug also persists in 2.5.4)
--------------------------------------------------------------
1. ``prepare_account_df`` assigned layer_id via ``cumcount()`` on unsorted rows,
   so reordering account.csv changed which PolNumber got layer_id=1, 2, …
   Fix: sort before cumcount.

2. The "drop premature layering" dedup in ``get_il_input_items`` collapsed
   layers that carried genuinely DIFFERENT financial terms (e.g. layer-specific
   CondLimits): at the condition level no agg is yet "layered" so ``layered_id``
   became 0 for every row and distinct policy layers were merged, dropping
   layers and applying the wrong LayerParticipation.  Making the order
   deterministic alone only made the wrong answer stable.
   Fix: keep a row per layer when a (gul_input, agg) has more than one distinct
   profile_id across its layers; collapse only when every layer shares the same
   profile (so the FM structure stays canonical for ordinary accounts).

3. A no-op ``sort_values`` (result not assigned back) after the layer concat
   left the DataFrame in accounts.csv row order, making subsequent
   ``factorize_ndarray`` calls for agg_id order-dependent.
   Fix: assign the sort result.

4. The PolNumber backfill used ``set_index(output_id - 1)`` which misaligned
   when gul_inputs_df had a non-contiguous index.
   Fix: use reset_index/set_index('index').

5. A preserved policy layer can carry a non-canonical PolNumber, so the
   term-based acc_idx merge can leave gaps.
   Fix: fill any remaining acc_idx from the (acc_id, layer_id[, CondTag])
   relationship that fully determines the source account row.
"""
import os
import tempfile
import shutil
import logging

import pandas as pd
from ods_tools.oed import OedExposure
from oasislmf.computation.run.exposure import RunExposure
from oasislmf.preparation.gul_inputs import get_gul_input_items
from oasislmf.preparation.il_inputs import get_il_input_items
from oasislmf.utils.data import prepare_oed_exposure
from oasislmf.utils.status import OASIS_KEYS_STATUS_MODELLED

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Issue-2040 test data: one account, four policies, four CondTags.
# Each CondTag has exactly one policy with a large CondLimit, the others $1.
# Different policy row orders produce the buggy differences.
# ---------------------------------------------------------------------------

_ACC_HEADER = (
    "PortNumber,AccNumber,AccCurrency,PolNumber,PolPerilsCovered,"
    "CondPeril,CondNumber,CondLimit6All,CondDed6All,CondDedType6All,"
    "CondLimitType6All,CondPriority,LayerParticipation,LayerLimit,"
    "LayerAttachment,CondTag,CondClass,OEDVersion\n"
)

# Four policies, each with all four CondTags.
# The "big" CondLimit rotates: CondTag 1→Pol3, 2→Pol2, 3→Pol1, 4→Pol4.
_ACC_ROWS = [
    # Pol1 CondTag=1 (small) CondTag=2 (small) CondTag=3 (big=100M) CondTag=4 (small)
    "1,ACC1,USD,Pol1,AA1,AA1,1,        1.0,0,0,0,1,0.02,1e12,0,1,0,4.0.0\n",
    "1,ACC1,USD,Pol1,AA1,AA1,2,        1.0,0,0,0,1,0.02,1e12,0,2,0,4.0.0\n",
    "1,ACC1,USD,Pol1,AA1,AA1,3,100000000.0,0,0,0,1,0.02,1e12,0,3,0,4.0.0\n",
    "1,ACC1,USD,Pol1,AA1,AA1,4,        1.0,0,0,0,1,0.02,1e12,0,4,0,4.0.0\n",
    # Pol2 CondTag=1 (small) CondTag=2 (big=250M) CondTag=3 (small) CondTag=4 (small)
    "1,ACC1,USD,Pol2,AA1,AA1,1,        1.0,0,0,0,1,0.0925,1e12,0,1,0,4.0.0\n",
    "1,ACC1,USD,Pol2,AA1,AA1,2,250000000.0,0,0,0,1,0.0925,1e12,0,2,0,4.0.0\n",
    "1,ACC1,USD,Pol2,AA1,AA1,3,        1.0,0,0,0,1,0.0925,1e12,0,3,0,4.0.0\n",
    "1,ACC1,USD,Pol2,AA1,AA1,4,        1.0,0,0,0,1,0.0925,1e12,0,4,0,4.0.0\n",
    # Pol3 CondTag=1 (big=150M) CondTag=2 (small) CondTag=3 (small) CondTag=4 (small)
    "1,ACC1,USD,Pol3,AA1,AA1,1,150000000.0,0,0,0,1,0.1675,1e12,0,1,0,4.0.0\n",
    "1,ACC1,USD,Pol3,AA1,AA1,2,        1.0,0,0,0,1,0.1675,1e12,0,2,0,4.0.0\n",
    "1,ACC1,USD,Pol3,AA1,AA1,3,        1.0,0,0,0,1,0.1675,1e12,0,3,0,4.0.0\n",
    "1,ACC1,USD,Pol3,AA1,AA1,4,        1.0,0,0,0,1,0.1675,1e12,0,4,0,4.0.0\n",
    # Pol4 CondTag=1 (small) CondTag=2 (small) CondTag=3 (small) CondTag=4 (big=250M)
    "1,ACC1,USD,Pol4,AA1,AA1,1,        1.0,0,0,0,1,0.044,1e12,0,1,0,4.0.0\n",
    "1,ACC1,USD,Pol4,AA1,AA1,2,        1.0,0,0,0,1,0.044,1e12,0,2,0,4.0.0\n",
    "1,ACC1,USD,Pol4,AA1,AA1,3,        1.0,0,0,0,1,0.044,1e12,0,3,0,4.0.0\n",
    "1,ACC1,USD,Pol4,AA1,AA1,4,250000000.0,0,0,0,1,0.044,1e12,0,4,0,4.0.0\n",
]

_LOC_HEADER = (
    "PortNumber,AccNumber,LocNumber,IsTenant,BuildingID,"
    "CountryCode,Latitude,Longitude,StreetAddress,PostalCode,"
    "OccupancyCode,ConstructionCode,LocPerilsCovered,"
    "BuildingTIV,OtherTIV,ContentsTIV,BITIV,LocCurrency,CondTag,OEDVersion\n"
)
_LOC_ROWS = [
    "1,ACC1,L1,1,1,US,34.0,-118.0,addr1,90001,1050,5000,AA1,1000000,0,0,0,USD,1,4.0.0\n",
    "1,ACC1,L2,1,1,US,34.1,-118.1,addr2,90001,1050,5000,AA1,2000000,0,0,0,USD,2,4.0.0\n",
    "1,ACC1,L3,1,1,US,34.2,-118.2,addr3,90001,1050,5000,AA1,1500000,0,0,0,USD,3,4.0.0\n",
    "1,ACC1,L4,1,1,US,34.3,-118.3,addr4,90001,1050,5000,AA1,2500000,0,0,0,USD,4,4.0.0\n",
    "1,ACC1,L5,1,1,US,34.4,-118.4,addr5,90001,1050,5000,AA1, 800000,0,0,0,USD,1,4.0.0\n",
    "1,ACC1,L6,1,1,US,34.5,-118.5,addr6,90001,1050,5000,AA1,1200000,0,0,0,USD,2,4.0.0\n",
]

LOC_CSV = _LOC_HEADER + "".join(_LOC_ROWS)

# Three orderings of the same account rows: original, reversed, and shuffled.
ACC_CSV_ORIG = _ACC_HEADER + "".join(_ACC_ROWS)
ACC_CSV_REV = _ACC_HEADER + "".join(reversed(_ACC_ROWS))
_shuffled = _ACC_ROWS[3::4] + _ACC_ROWS[1::4] + _ACC_ROWS[2::4] + _ACC_ROWS[0::4]
ACC_CSV_SHUFFLE = _ACC_HEADER + "".join(_shuffled)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_exposure(loc_csv, acc_csv, _tmpdir=None):
    """Write CSV strings to temp files and load as OedExposure."""
    d = _tmpdir or tempfile.mkdtemp()
    loc_fp = os.path.join(d, "location.csv")
    acc_fp = os.path.join(d, "account.csv")
    with open(loc_fp, "w") as f:
        f.write(loc_csv)
    with open(acc_fp, "w") as f:
        f.write(acc_csv)
    exp = OedExposure(
        location=loc_fp,
        account=acc_fp,
        use_field=True,
        check_oed=False,
    )
    prepare_oed_exposure(exp)
    return exp, d


def _build_keys(loc_df):
    rows = []
    for _, row in loc_df.iterrows():
        for cov in (1,):
            rows.append({
                "loc_id": row["loc_id"],
                "peril_id": "AA1",
                "coverage_type_id": cov,
                "area_peril_id": 1,
                "vulnerability_id": 1,
                "status": "success",
                "message": "",
            })
    return pd.DataFrame(rows)


def _generate_fm_files(loc_csv, acc_csv):
    with tempfile.TemporaryDirectory() as src:
        exp, _ = _build_exposure(loc_csv, acc_csv, _tmpdir=src)
        keys_df = _build_keys(exp.location.dataframe)
        gul_df = get_gul_input_items(
            exp.location.dataframe, keys_df, damage_group_id_cols=["loc_id"]
        )
        gul_df = gul_df[gul_df["status"].isin(OASIS_KEYS_STATUS_MODELLED)]

        out = os.path.join(src, "fm_out")
        os.makedirs(out)
        get_il_input_items(gul_df, exp, out, logger)
        result = {}
        for fname in ("fm_xref.csv", "fm_policytc.csv", "fm_profile.csv",
                      "fm_programme.csv"):
            fp = os.path.join(out, fname)
            if os.path.exists(fp):
                result[fname] = pd.read_csv(fp)
    return result


# ---------------------------------------------------------------------------
# Test 1: fm_xref is identical across account row orderings
# ---------------------------------------------------------------------------

def test_fm_xref_is_order_independent():
    """
    fm_xref.csv must be byte-for-byte identical regardless of account.csv
    row order.  fm_xref maps every (GUL item, policy layer) pair to an
    FM output ID.  Before the fix, policies sharing the same CondTag financial
    profile were collapsed arbitrarily, meaning only one or two of four layers
    ended up in fm_xref (the specific ones depending on CSV order).
    """
    orig = _generate_fm_files(LOC_CSV, ACC_CSV_ORIG)
    rev = _generate_fm_files(LOC_CSV, ACC_CSV_REV)
    shuf = _generate_fm_files(LOC_CSV, ACC_CSV_SHUFFLE)

    xref_o = orig["fm_xref.csv"].sort_values(["agg_id", "layer_id"]).reset_index(drop=True)
    xref_r = rev["fm_xref.csv"].sort_values(["agg_id", "layer_id"]).reset_index(drop=True)
    xref_s = shuf["fm_xref.csv"].sort_values(["agg_id", "layer_id"]).reset_index(drop=True)

    pd.testing.assert_frame_equal(
        xref_o, xref_r,
        check_like=False,
        obj="fm_xref (orig vs reversed account order)",
    )
    pd.testing.assert_frame_equal(
        xref_o, xref_s,
        check_like=False,
        obj="fm_xref (orig vs shuffled account order)",
    )


def test_fm_xref_contains_all_policy_layers():
    """
    Every GUL item must appear in fm_xref for ALL four policy layers.
    In this scenario each layer applies a different CondLimit per CondTag, so
    the layers are genuinely distinct and must all be preserved.  Before the
    fix, the 'drop premature layering' dedup collapsed layers, leaving each GUL
    item with only two of the four layers (and the wrong LayerParticipation).
    """
    orig = _generate_fm_files(LOC_CSV, ACC_CSV_ORIG)
    xref = orig["fm_xref.csv"]

    layers_per_item = xref.groupby("agg_id")["layer_id"].apply(
        lambda s: frozenset(s.tolist())
    )
    expected_layers = frozenset([1, 2, 3, 4])
    wrong = {
        int(aid): sorted(layers)
        for aid, layers in layers_per_item.items()
        if layers != expected_layers
    }
    assert not wrong, (
        f"Some GUL items have incorrect layer sets in fm_xref.\n"
        f"Expected each item to appear with layers {sorted(expected_layers)}.\n"
        f"Items with wrong layers: {wrong}\n"
        "This indicates the premature-layering deduplication is discarding layers."
    )


def test_fm_policytc_structure_is_order_independent():
    """
    fm_policytc.csv structure (row count, levels, unique agg_ids and profiles
    per level) must be identical regardless of account.csv row order.
    """
    orig = _generate_fm_files(LOC_CSV, ACC_CSV_ORIG)
    rev = _generate_fm_files(LOC_CSV, ACC_CSV_REV)

    def _structure(df):
        return {
            "row_count": len(df),
            "levels": sorted(df["level_id"].unique().tolist()),
            "agg_ids_per_level": {
                int(lvl): int(grp["agg_id"].nunique())
                for lvl, grp in df.groupby("level_id")
            },
            "profiles_per_level": {
                int(lvl): int(grp["profile_id"].nunique())
                for lvl, grp in df.groupby("level_id")
            },
            "layers_per_level": {
                int(lvl): sorted(grp["layer_id"].unique().tolist())
                for lvl, grp in df.groupby("level_id")
            },
        }

    s_orig = _structure(orig["fm_policytc.csv"])
    s_rev = _structure(rev["fm_policytc.csv"])

    assert s_orig == s_rev, (
        f"fm_policytc.csv structure differs between account orderings.\n"
        f"  Original:  {s_orig}\n"
        f"  Reversed:  {s_rev}\n"
        "This indicates an IL preparation ordering bug."
    )


# ---------------------------------------------------------------------------
# Test 2: End-to-end IL losses are order-independent (RunExposure)
# ---------------------------------------------------------------------------

def test_il_losses_are_order_independent():
    """
    End-to-end: identical portfolios with different account.csv row orders
    must produce exactly equal per-location IL losses.

    Uses RunExposure (deterministic loss factor = 1.0) which exercises the
    full preparation → fmpy pipeline without stochastic sampling.
    """
    tmpdir_o = tempfile.mkdtemp()
    tmpdir_r = tempfile.mkdtemp()
    try:
        for tmpdir, acc_csv, label in [
            (tmpdir_o, ACC_CSV_ORIG, "original"),
            (tmpdir_r, ACC_CSV_REV, "reversed"),
        ]:
            with open(os.path.join(tmpdir, "location.csv"), "w") as f:
                f.write(LOC_CSV)
            with open(os.path.join(tmpdir, "account.csv"), "w") as f:
                f.write(acc_csv)

        def _run(src, run):
            out_fp = os.path.join(run, "loc_summary.csv")
            RunExposure(
                src_dir=src,
                run_dir=run,
                output_file=out_fp,
                output_level="loc",
                loss_factor=[1.0],
                fmpy=False,
                check_oed=False,
            ).run()
            return pd.read_csv(out_fp)

        df_o = _run(tmpdir_o, os.path.join(tmpdir_o, "run"))
        df_r = _run(tmpdir_r, os.path.join(tmpdir_r, "run"))

        # Sort by location identifier for a stable comparison.
        sort_col = "LocNumber" if "LocNumber" in df_o.columns else df_o.columns[0]
        loss_col = "loss_il" if "loss_il" in df_o.columns else "loss"
        df_o = df_o.sort_values(sort_col).reset_index(drop=True)
        df_r = df_r.sort_values(sort_col).reset_index(drop=True)

        pd.testing.assert_frame_equal(
            df_o[[sort_col, loss_col]],
            df_r[[sort_col, loss_col]],
            check_like=False,
            obj="per-location IL losses (orig vs reversed account order)",
        )

        # Absolute magnitude guard: each of the four layers must participate.
        # Hand calculation Σ_layer[LayerParticipation × Σ_loc min(TIV, CondLimit)]
        # = 737,501; the buggy layer-collapse under-counted to ~166,942 (some
        # locations zeroed out). This catches a regression that is still
        # order-independent but numerically wrong.
        assert abs(df_o[loss_col].sum() - 737500.66) < 1.0, (
            f"Total IL {df_o[loss_col].sum():,.2f} != expected 737,500.66 — "
            "policy layers are being dropped/mis-applied (premature-layering collapse)."
        )
    finally:
        shutil.rmtree(tmpdir_o, ignore_errors=True)
        shutil.rmtree(tmpdir_r, ignore_errors=True)
