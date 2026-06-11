import logging
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa

from oasislmf.pytools.common.data import oasis_float, write_ndarray_to_fmt_csv
from oasislmf.pytools.lec.aggreports.outputs.full_uncertainty import output_full_uncertainty
from oasislmf.pytools.lec.aggreports.outputs.mean_damage_ratio import output_mean_damage_ratio
from oasislmf.pytools.lec.aggreports.outputs.sample_mean import output_sample_mean, reorder_losses_by_summary_and_period
from oasislmf.pytools.lec.aggreports.outputs.wheatsheaf import fill_wheatsheaf_items
from oasislmf.pytools.lec.aggreports.outputs.wheatsheaf_mean import fill_wheatsheaf_mean_items, get_wheatsheaf_max_count
from oasislmf.pytools.lec.aggreports.write_tables import write_ept, write_ept_weighted, write_psept, write_psept_weighted, write_wheatsheaf_mean
from oasislmf.pytools.lec.data import (
    AEP, AEPTVAR, AGG_FULL_UNCERTAINTY, AGG_SAMPLE_MEAN, AGG_WHEATSHEAF, AGG_WHEATSHEAF_MEAN,
    FULL, MEANDR, MEANSAMPLE, OCC_FULL_UNCERTAINTY, OCC_SAMPLE_MEAN, OCC_WHEATSHEAF,
    OCC_WHEATSHEAF_MEAN, OEP, OEPTVAR, PERSAMPLEMEAN,
)
from oasislmf.pytools.lec.data import LOSSVEC2MAP_dtype, MEANMAP_dtype, WHEATKEYITEMS_dtype

logger = logging.getLogger(__name__)


@dataclass
class LecConfig:
    """Shared per-run configuration passed to output helpers and output_for_summary_idx."""
    period_weights: np.ndarray
    max_summary_id: int
    sample_size: int
    no_of_periods: int
    num_sidxs: int
    use_return_period: bool
    returnperiods: np.ndarray


def make_output_fn(outmap, output_binary, output_parquet):
    """Return a callable(data, out_type) that writes data in the correct output format."""
    def output_data(data, out_type):
        if output_binary:
            data.tofile(outmap[out_type]["file"])
        elif output_parquet:
            data_df = pd.DataFrame(data)
            data_table = pa.Table.from_pandas(data_df)
            outmap[out_type]["file"].write_table(data_table)
        else:
            write_ndarray_to_fmt_csv(
                outmap[out_type]["file"],
                data,
                outmap[out_type]["headers"],
                outmap[out_type]["fmt"]
            )
    return output_data


# ---------------------------------------------------------------------------
# Shared compute-and-write helpers — arrays pre-allocated by caller
# (memmap for sequential, np.zeros for idx). alloc_* callables handle late-sized arrays.
# ---------------------------------------------------------------------------

def _write_mean_damage_ratio(
    items, items_start_end, row_used_indices_mean, outloss_mean_vals,
    config, eptype, eptype_tvar, output_fn,
):
    has_weights, used_period_no = output_mean_damage_ratio(
        items, items_start_end, row_used_indices_mean,
        outloss_mean_vals, config.period_weights, config.max_summary_id,
    )
    unused_pw = config.period_weights[~used_period_no]
    if has_weights:
        gen = write_ept_weighted(
            items, items_start_end, config.sample_size, MEANDR,
            eptype, eptype_tvar, unused_pw,
            config.use_return_period, config.returnperiods, config.max_summary_id,
        )
    else:
        gen = write_ept(
            items, items_start_end, config.no_of_periods, MEANDR,
            eptype, eptype_tvar,
            config.use_return_period, config.returnperiods, config.max_summary_id,
        )
    for data in gen:
        output_fn(data, "ept")


def _write_full_uncertainty(
    items, items_start_end, row_used_indices_sample, outloss_sample_vals,
    config, eptype, eptype_tvar, output_fn,
):
    has_weights, used_period_no = output_full_uncertainty(
        items, items_start_end, row_used_indices_sample,
        outloss_sample_vals, config.period_weights, config.max_summary_id, config.num_sidxs,
    )
    unused_pw = config.period_weights[~used_period_no]
    if has_weights:
        gen = write_ept_weighted(
            items, items_start_end, 1, FULL,
            eptype, eptype_tvar, unused_pw,
            config.use_return_period, config.returnperiods, config.max_summary_id,
        )
    else:
        gen = write_ept(
            items, items_start_end, config.no_of_periods * config.sample_size, FULL,
            eptype, eptype_tvar,
            config.use_return_period, config.returnperiods, config.max_summary_id,
        )
    for data in gen:
        output_fn(data, "ept")


def _write_wheatsheaf(
    wheatsheaf_items, wheatsheaf_items_start_end, mean_map,
    row_used_indices_sample, outloss_sample_vals,
    config, eptype, eptype_tvar, do_wheat, do_wheat_mean, alloc_wm_items, output_fn,
):
    """alloc_wm_items: no-weights branch only; np.zeros for idx, memmap lambda for seq."""
    has_weights, used_period_no = fill_wheatsheaf_items(
        wheatsheaf_items, wheatsheaf_items_start_end, row_used_indices_sample,
        outloss_sample_vals, config.period_weights, config.max_summary_id, config.num_sidxs,
    )
    unused_pw = config.period_weights[~used_period_no]

    if has_weights:
        if do_wheat:
            gen = write_psept_weighted(
                wheatsheaf_items, wheatsheaf_items_start_end, config.no_of_periods,
                eptype, eptype_tvar, unused_pw,
                config.use_return_period, config.returnperiods,
                config.max_summary_id, config.num_sidxs, config.sample_size, mean_map=mean_map,
            )
            for data in gen:
                output_fn(data, "psept")
        if do_wheat_mean and mean_map is not None:
            gen = write_wheatsheaf_mean(mean_map, eptype, PERSAMPLEMEAN, config.max_summary_id)
            for data in gen:
                output_fn(data, "ept")
    else:
        if do_wheat:
            gen = write_psept(
                wheatsheaf_items, wheatsheaf_items_start_end, config.no_of_periods,
                eptype, eptype_tvar,
                config.use_return_period, config.returnperiods,
                config.max_summary_id, config.num_sidxs,
            )
            for data in gen:
                output_fn(data, "psept")
        if do_wheat_mean:
            maxcounts = get_wheatsheaf_max_count(
                wheatsheaf_items, wheatsheaf_items_start_end, config.max_summary_id)
            if np.any(maxcounts != -1):
                wm_items = alloc_wm_items(np.sum(maxcounts[maxcounts != -1]))
                wm_items_start_end = fill_wheatsheaf_mean_items(
                    wm_items, wheatsheaf_items, wheatsheaf_items_start_end,
                    maxcounts, config.max_summary_id, config.num_sidxs,
                )
                gen = write_ept(
                    wm_items, wm_items_start_end, config.no_of_periods,
                    PERSAMPLEMEAN, eptype, eptype_tvar,
                    config.use_return_period, config.returnperiods, config.max_summary_id,
                    sample_size=config.sample_size,
                )
                for data in gen:
                    output_fn(data, "ept")


def _write_sample_mean(
    reordered_outlosses, row_used_indices_sample, outloss_sample_vals,
    config, eptype, eptype_tvar, alloc_items, output_fn,
):
    """reordered_outlosses shape: no_of_periods * max_summary_id. alloc_items size is post-computation."""
    reorder_losses_by_summary_and_period(
        reordered_outlosses, row_used_indices_sample, outloss_sample_vals,
        config.max_summary_id, config.no_of_periods, config.num_sidxs, config.sample_size,
    )
    row_used_ro = np.flatnonzero(reordered_outlosses["row_used"])
    items = alloc_items(len(row_used_ro))
    items_start_end = np.full((config.max_summary_id, 2), -1, dtype=np.int32)
    has_weights, used_period_no = output_sample_mean(
        items, items_start_end, row_used_ro, reordered_outlosses["value"],
        config.period_weights, config.max_summary_id, config.no_of_periods,
    )
    unused_pw = config.period_weights[~used_period_no]
    if has_weights:
        gen = write_ept_weighted(
            items, items_start_end, config.sample_size, MEANSAMPLE,
            eptype, eptype_tvar, unused_pw,
            config.use_return_period, config.returnperiods, config.max_summary_id,
        )
    else:
        gen = write_ept(
            items, items_start_end, config.no_of_periods, MEANSAMPLE,
            eptype, eptype_tvar,
            config.use_return_period, config.returnperiods, config.max_summary_id,
        )
    for data in gen:
        output_fn(data, "ept")


# ---------------------------------------------------------------------------
# Sequential path — AggReports
# Allocates memmaps, calls the shared helpers above.
# ---------------------------------------------------------------------------

class AggReports():
    def __init__(
        self,
        outmap,
        outloss_mean,
        row_used_mean,
        outloss_sample,
        row_used_sample,
        config,
        lec_files_folder,
        output_binary,
        output_parquet,
    ):
        self.outmap = outmap
        self.outloss_mean = outloss_mean
        self.outloss_sample = outloss_sample
        self.lec_files_folder = lec_files_folder
        self.row_used_indices_mean = np.flatnonzero(row_used_mean)
        self.row_used_indices_sample = np.flatnonzero(row_used_sample)
        self.config = config
        self.output_data = make_output_fn(outmap, output_binary, output_parquet)

    def output_mean_damage_ratio(self, eptype, eptype_tvar, outloss_type):
        """Output Mean Damage Ratio
        Mean Damage Losses - This means do the loss calculation for a year using the event mean
        damage loss computed by numerical integration of the effective damageability distributions.
        Args:
            eptype (int): Exceedance Probability Type
            eptype_tvar (int): Exceedance Probability Type (Tail Value at Risk)
            outloss_type (string): Which loss to output
        """
        if outloss_type not in ("agg_out_loss", "max_out_loss"):
            raise ValueError(f"Error: Unknown outloss_type: {outloss_type}")
        outloss_vals = self.outloss_mean[outloss_type]
        row_used_indices = self.row_used_indices_mean
        items = np.memmap(
            Path(self.lec_files_folder, f"lec_mean_damage_ratio-{outloss_type}-items.bdat"),
            dtype=LOSSVEC2MAP_dtype, mode="w+", shape=(len(row_used_indices),),
        )
        items_start_end = np.full((self.config.max_summary_id, 2), -1, dtype=np.int32)
        _write_mean_damage_ratio(
            items, items_start_end, row_used_indices, outloss_vals,
            self.config, eptype, eptype_tvar, self.output_data,
        )

    def output_full_uncertainty(self, eptype, eptype_tvar, outloss_type):
        """Output Full Uncertainty
        Full Uncertainty – this means do the calculation across all samples (treating the samples
        effectively as repeat years) - this is the most accurate of all the single EP Curves.
        Args:
            eptype (int): Exceedance Probability Type
            eptype_tvar (int): Exceedance Probability Type (Tail Value at Risk)
            outloss_type (string): Which loss to output
        """
        if outloss_type not in ("agg_out_loss", "max_out_loss"):
            raise ValueError(f"Error: Unknown outloss_type: {outloss_type}")
        outloss_vals = self.outloss_sample[outloss_type]
        row_used_indices = self.row_used_indices_sample
        items = np.memmap(
            Path(self.lec_files_folder, f"lec_full_uncertainty-{outloss_type}-items.bdat"),
            dtype=LOSSVEC2MAP_dtype, mode="w+", shape=(len(row_used_indices),),
        )
        items_start_end = np.full((self.config.max_summary_id, 2), -1, dtype=np.int32)
        _write_full_uncertainty(
            items, items_start_end, row_used_indices, outloss_vals,
            self.config, eptype, eptype_tvar, self.output_data,
        )

    def output_wheatsheaf_and_wheatsheafmean(self, eptype, eptype_tvar, outloss_type, output_wheatsheaf, output_wheatsheaf_mean):
        """Output Wheatsheaf and Wheatsheaf Mean
        Wheatsheaf, Per Sample EPT (PSEPT) – this means calculate the EP Curve for each sample and
        leave it at the sample level of detail, resulting in multiple "curves".
        Wheatsheaf Mean, Per Sample mean EPT – this means average the loss at each return period of
        the Per Sample EPT.
        Args:
            eptype (int): Exceedance Probability Type
            eptype_tvar (int): Exceedance Probability Type (Tail Value at Risk)
            outloss_type (string): Which loss to output
            output_wheatsheaf (bool): Bool to Output Wheatsheaf
            output_wheatsheaf_mean (bool): Bool to Output Wheatsheaf Mean
        """
        if outloss_type not in ("agg_out_loss", "max_out_loss"):
            raise ValueError(f"Error: Unknown outloss_type: {outloss_type}")
        outloss_vals = self.outloss_sample[outloss_type]
        row_used_indices = self.row_used_indices_sample
        wh_items = np.memmap(
            Path(self.lec_files_folder, f"lec_wheatsheaf-items-{outloss_type}.bdat"),
            dtype=WHEATKEYITEMS_dtype, mode="w+", shape=(len(row_used_indices),),
        )
        wh_items_start_end = np.full(
            (self.config.max_summary_id * self.config.num_sidxs, 2), -1, dtype=np.int32,
        )
        mean_map = None
        if output_wheatsheaf_mean:
            mean_map = np.memmap(
                Path(self.lec_files_folder, f"lec_wheatsheaf_mean-map-{outloss_type}.bdat"),
                dtype=MEANMAP_dtype, mode="w+",
                shape=(self.config.max_summary_id, len(self.config.returnperiods)),
            )
        wm_items_path = Path(self.lec_files_folder, f"lec_wheatsheaf_mean-items-{outloss_type}.bdat")

        def alloc_wm_items(size): return np.memmap(
            wm_items_path, dtype=LOSSVEC2MAP_dtype, mode="w+", shape=(size,)
        )
        _write_wheatsheaf(
            wh_items, wh_items_start_end, mean_map, row_used_indices, outloss_vals,
            self.config, eptype, eptype_tvar, output_wheatsheaf, output_wheatsheaf_mean,
            alloc_wm_items, self.output_data,
        )

    def output_sample_mean(self, eptype, eptype_tvar, outloss_type):
        """Output Sample Mean
        Sample Mean Losses – this means do the loss calculation for a year using the statistical
        sample event mean.
        Args:
            eptype (int): Exceedance Probability Type
            eptype_tvar (int): Exceedance Probability Type (Tail Value at Risk)
            outloss_type (string): Which loss to output
        """
        if self.config.sample_size == 0:
            logger.warning("aggreports.output_sample_mean, self.sample_size is 0, not outputting any sample mean")
            return
        if outloss_type not in ("agg_out_loss", "max_out_loss"):
            raise ValueError(f"Error: Unknown outloss_type: {outloss_type}")
        outloss_vals = self.outloss_sample[outloss_type]
        reordered_outlosses = np.memmap(
            Path(self.lec_files_folder, f"lec_sample_mean-reordered_outlosses-{outloss_type}.bdat"),
            dtype=np.dtype([("row_used", np.bool_), ("value", oasis_float)]),
            mode="w+",
            shape=(self.config.no_of_periods * self.config.max_summary_id),
        )
        items_path = Path(self.lec_files_folder, f"lec_sample_mean-{outloss_type}-items.bdat")

        def alloc_items(size): return np.memmap(
            items_path, dtype=LOSSVEC2MAP_dtype, mode="w+", shape=(size,)
        )
        _write_sample_mean(
            reordered_outlosses, self.row_used_indices_sample, outloss_vals,
            self.config, eptype, eptype_tvar, alloc_items, self.output_data,
        )


# ---------------------------------------------------------------------------
# Idx path — output_for_summary_idx
# Allocates small np.zeros arrays (single summary at a time), calls the same helpers.
# ---------------------------------------------------------------------------

def output_for_summary_idx(
    summary_id,
    outloss_mean_s,
    row_used_mean_s,
    outloss_sample_s,
    row_used_sample_s,
    output_flags,
    hasOCC,
    hasAGG,
    outmap,
    config,
    output_fn,
):
    """Output all report types for a single summary_id (idx path).

    Called once per summary_id. config.max_summary_id must be 1; arrays are sized
    for a single summary. Write generators emit SummaryId=1, corrected here before each write.
    """
    row_used_indices_mean = np.flatnonzero(row_used_mean_s)
    row_used_indices_sample = np.flatnonzero(row_used_sample_s)
    if not len(row_used_indices_mean) and not len(row_used_indices_sample):
        return

    def _out(data, out_type):
        if len(data):
            data["SummaryId"] = summary_id
        output_fn(data, out_type)

    if outmap["ept"]["compute"] and (hasOCC or hasAGG):
        for eptype, eptype_tvar, outloss_type, wanted in (
            (OEP, OEPTVAR, "max_out_loss", hasOCC),
            (AEP, AEPTVAR, "agg_out_loss", hasAGG),
        ):
            if not wanted:
                continue
            items = np.zeros(len(row_used_indices_mean), dtype=LOSSVEC2MAP_dtype)
            items_start_end = np.full((1, 2), -1, dtype=np.int32)
            _write_mean_damage_ratio(
                items, items_start_end, row_used_indices_mean,
                outloss_mean_s[outloss_type], config, eptype, eptype_tvar, _out,
            )

    for flag, eptype, eptype_tvar, outloss_type in (
        (OCC_FULL_UNCERTAINTY, OEP, OEPTVAR, "max_out_loss"),
        (AGG_FULL_UNCERTAINTY, AEP, AEPTVAR, "agg_out_loss"),
    ):
        if not output_flags[flag]:
            continue
        items = np.zeros(len(row_used_indices_sample), dtype=LOSSVEC2MAP_dtype)
        items_start_end = np.full((1, 2), -1, dtype=np.int32)
        _write_full_uncertainty(
            items, items_start_end, row_used_indices_sample,
            outloss_sample_s[outloss_type], config, eptype, eptype_tvar, _out,
        )

    for flag_w, flag_wm, eptype, eptype_tvar, outloss_type in (
        (OCC_WHEATSHEAF, OCC_WHEATSHEAF_MEAN, OEP, OEPTVAR, "max_out_loss"),
        (AGG_WHEATSHEAF, AGG_WHEATSHEAF_MEAN, AEP, AEPTVAR, "agg_out_loss"),
    ):
        do_wheat = output_flags[flag_w]
        do_wheat_mean = output_flags[flag_wm]
        if not (do_wheat or do_wheat_mean):
            continue
        wheatsheaf_items = np.zeros(len(row_used_indices_sample), dtype=WHEATKEYITEMS_dtype)
        wheatsheaf_items_start_end = np.full((config.num_sidxs, 2), -1, dtype=np.int32)
        mean_map = np.zeros((1, len(config.returnperiods)), dtype=MEANMAP_dtype) if do_wheat_mean else None
        _write_wheatsheaf(
            wheatsheaf_items, wheatsheaf_items_start_end, mean_map,
            row_used_indices_sample, outloss_sample_s[outloss_type],
            config, eptype, eptype_tvar, do_wheat, do_wheat_mean,
            lambda size: np.zeros(size, dtype=LOSSVEC2MAP_dtype),
            _out,
        )

    if config.sample_size == 0:
        return
    for flag, eptype, eptype_tvar, outloss_type in (
        (OCC_SAMPLE_MEAN, OEP, OEPTVAR, "max_out_loss"),
        (AGG_SAMPLE_MEAN, AEP, AEPTVAR, "agg_out_loss"),
    ):
        if not output_flags[flag]:
            continue
        reordered = np.zeros(
            config.no_of_periods,
            dtype=np.dtype([("row_used", np.bool_), ("value", oasis_float)]),
        )
        _write_sample_mean(
            reordered, row_used_indices_sample, outloss_sample_s[outloss_type],
            config, eptype, eptype_tvar,
            lambda size: np.zeros(size, dtype=LOSSVEC2MAP_dtype),
            _out,
        )
