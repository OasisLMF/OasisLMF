import logging
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
from oasislmf.pytools.lec.data import FULL, MEANDR, MEANSAMPLE, PERSAMPLEMEAN
from oasislmf.pytools.lec.data import LOSSVEC2MAP_dtype, MEANMAP_dtype, WHEATKEYITEMS_dtype

logger = logging.getLogger(__name__)


class AggReports():
    def __init__(
        self,
        outmap,
        outloss_mean,
        outloss_sample,
        period_weights,
        max_summary_id,
        sample_size,
        no_of_periods,
        num_sidxs,
        use_return_period,
        returnperiods,
        lec_files_folder,
        output_binary,
        output_parquet,
    ):
        self.outmap = outmap
        self.outloss_mean = outloss_mean
        self.outloss_sample = outloss_sample
        self.period_weights = period_weights
        self.max_summary_id = max_summary_id
        self.sample_size = sample_size
        self.no_of_periods = no_of_periods
        self.num_sidxs = num_sidxs
        self.use_return_period = use_return_period
        self.returnperiods = returnperiods
        self.lec_files_folder = lec_files_folder
        self.output_binary = output_binary
        self.output_parquet = output_parquet

    def output_data(self, data, out_type):
        if self.output_binary:
            data.tofile(self.outmap[out_type]["file"])
        elif self.output_parquet:
            data_df = pd.DataFrame(data)
            data_table = pa.Table.from_pandas(data_df)
            self.outmap[out_type]["file"].write_table(data_table)
        else:
            write_ndarray_to_fmt_csv(
                self.outmap[out_type]["file"],
                data,
                self.outmap[out_type]["headers"],
                self.outmap[out_type]["fmt"]
            )

    def output_mean_damage_ratio(self, eptype, eptype_tvar, outloss_type):
        """Output Mean Damage Ratio
        Mean Damage Losses - This means do the loss calculation for a year using the event mean
        damage loss computed by numerical integration of the effective damageability distributions.
        Args:
            eptype (int): Exceedance Probability Type
            eptype_tvar (int): Exceedance Probability Type (Tail Value at Risk)
            outloss_type (string): Which loss to output
        """
        epcalc = MEANDR

        # Get row indices that are used
        row_used_indices = np.where(self.outloss_mean["row_used"])[0]

        # Allocate storage for the flat data array
        items_fp = Path(self.lec_files_folder, f"lec_mean_damage_ratio-{outloss_type}-items.bdat")
        items = np.memmap(items_fp, dtype=LOSSVEC2MAP_dtype, mode="w+", shape=(len(row_used_indices),))
        # Track start and end indices for each summary_id
        items_start_end = np.full((self.max_summary_id, 2), -1, dtype=np.int32)

        # Select the correct outloss values based on type
        # Required if-else condition as njit cannot resolve outloss_type inside []
        if outloss_type == "agg_out_loss":
            outloss_vals = self.outloss_mean["agg_out_loss"]
        elif outloss_type == "max_out_loss":
            outloss_vals = self.outloss_mean["max_out_loss"]
        else:
            raise ValueError(f"Error: Unknown outloss_type: {outloss_type}")

        # Populate items and items_start_end
        has_weights, used_period_no = output_mean_damage_ratio(
            items,
            items_start_end,
            row_used_indices,
            outloss_vals,
            self.period_weights,
            self.max_summary_id,
        )
        unused_period_weights = self.period_weights[~used_period_no]

        if has_weights:
            gen = write_ept_weighted(
                items,
                items_start_end,
                self.sample_size,
                epcalc,
                eptype,
                eptype_tvar,
                unused_period_weights,
                self.use_return_period,
                self.returnperiods,
                self.max_summary_id
            )
        else:
            gen = write_ept(
                items,
                items_start_end,
                self.no_of_periods,
                epcalc,
                eptype,
                eptype_tvar,
                self.use_return_period,
                self.returnperiods,
                self.max_summary_id

            )

        for data in gen:
            self.output_data(data, "ept")

    def output_full_uncertainty(self, eptype, eptype_tvar, outloss_type):
        """Output Full Uncertainty
        Full Uncertainty – this means do the calculation across all samples (treating the samples
        effectively as repeat years) - this is the most accurate of all the single EP Curves.
        Args:
            eptype (int): Exceedance Probability Type
            eptype_tvar (int): Exceedance Probability Type (Tail Value at Risk)
            outloss_type (string): Which loss to output
        """
        epcalc = FULL

        # Get row indices that are used
        row_used_indices = np.where(self.outloss_sample["row_used"])[0]

        # Allocate storage for the flat data array
        items_fp = Path(self.lec_files_folder, f"lec_full_uncertainty-{outloss_type}-items.bdat")
        items = np.memmap(items_fp, dtype=LOSSVEC2MAP_dtype, mode="w+", shape=(len(row_used_indices),))
        # Track start and end indices for each summary_id
        items_start_end = np.full((self.max_summary_id, 2), -1, dtype=np.int32)

        # Select the correct outloss values based on type
        # Required if-else condition as njit cannot resolve outloss_type inside []
        if outloss_type == "agg_out_loss":
            outloss_vals = self.outloss_sample["agg_out_loss"]
        elif outloss_type == "max_out_loss":
            outloss_vals = self.outloss_sample["max_out_loss"]
        else:
            raise ValueError(f"Error: Unknown outloss_type: {outloss_type}")

        # Populate items and items_start_end
        has_weights, used_period_no = output_full_uncertainty(
            items,
            items_start_end,
            row_used_indices,
            outloss_vals,
            self.period_weights,
            self.max_summary_id,
            self.num_sidxs,
        )
        unused_period_weights = self.period_weights[~used_period_no]

        if has_weights:
            gen = write_ept_weighted(
                items,
                items_start_end,
                1,
                epcalc,
                eptype,
                eptype_tvar,
                unused_period_weights,
                self.use_return_period,
                self.returnperiods,
                self.max_summary_id
            )
        else:
            gen = write_ept(
                items,
                items_start_end,
                self.no_of_periods * self.sample_size,
                epcalc,
                eptype,
                eptype_tvar,
                self.use_return_period,
                self.returnperiods,
                self.max_summary_id
            )

        for data in gen:
            self.output_data(data, "ept")

    def output_wheatsheaf_and_wheatsheafmean(self, eptype, eptype_tvar, outloss_type, output_wheatsheaf, output_wheatsheaf_mean):
        """Output Wheatsheaf and Wheatsheaf Mean
        Wheatsheaf, Per Sample EPT (PSEPT) – this means calculate the EP Curve for each sample and
        leave it at the sample level of detail, resulting in multiple “curves”.
        Wheatsheaf Mean, Per Sample mean EPT – this means average the loss at each return period of
        the Per Sample EPT.
        Args:
            eptype (int): Exceedance Probability Type
            eptype_tvar (int): Exceedance Probability Type (Tail Value at Risk)
            outloss_type (string): Which loss to output
            output_wheatsheaf (bool): Bool to Output Wheatsheaf
            output_wheatsheaf_mean (bool): Bool to Output Wheatsheaf Mean
        """
        epcalc = PERSAMPLEMEAN

        # Get row indices that are used
        row_used_indices = np.where(self.outloss_sample["row_used"])[0]

        wheatsheaf_items_file = Path(self.lec_files_folder, f"lec_wheatsheaf-items-{outloss_type}.bdat")
        wheatsheaf_items = np.memmap(
            wheatsheaf_items_file,
            dtype=WHEATKEYITEMS_dtype,
            mode="w+",
            shape=(len(row_used_indices)),
        )
        # Track start and end indices for each summary_id and sidx
        wheatsheaf_items_start_end = np.full((self.max_summary_id * self.num_sidxs, 2), -1, dtype=np.int32)

        # Select the correct outloss values based on type
        # Required if-else condition as njit cannot resolve outloss_type inside []
        if outloss_type == "agg_out_loss":
            outloss_vals = self.outloss_sample["agg_out_loss"]
        elif outloss_type == "max_out_loss":
            outloss_vals = self.outloss_sample["max_out_loss"]
        else:
            raise ValueError(f"Error: Unknown outloss_type: {outloss_type}")

        # Populate wheatsheaf_items and wheatsheaf_items_start_end
        has_weights, used_period_no = fill_wheatsheaf_items(
            wheatsheaf_items,
            wheatsheaf_items_start_end,
            row_used_indices,
            outloss_vals,
            self.period_weights,
            self.max_summary_id,
            self.num_sidxs,
        )
        unused_period_weights = self.period_weights[~used_period_no]

        if has_weights:
            mean_map = None

            if output_wheatsheaf_mean:
                mean_map_file = Path(self.lec_files_folder, f"lec_wheatsheaf_mean-map-{outloss_type}.bdat")
                mean_map = np.memmap(
                    mean_map_file,
                    dtype=MEANMAP_dtype,
                    mode="w+",
                    shape=(self.max_summary_id, len(self.returnperiods)),
                )

            if output_wheatsheaf:
                gen = write_psept_weighted(
                    wheatsheaf_items,
                    wheatsheaf_items_start_end,
                    self.no_of_periods,
                    eptype,
                    eptype_tvar,
                    unused_period_weights,
                    self.use_return_period,
                    self.returnperiods,
                    self.max_summary_id,
                    self.num_sidxs,
                    self.sample_size,
                    mean_map=mean_map,
                )
                for data in gen:
                    self.output_data(data, "psept")

            if output_wheatsheaf_mean:
                gen = write_wheatsheaf_mean(
                    mean_map,
                    eptype,
                    epcalc,
                    self.max_summary_id,
                )
                for data in gen:
                    self.output_data(data, "ept")
        else:
            if output_wheatsheaf:
                gen = write_psept(
                    wheatsheaf_items,
                    wheatsheaf_items_start_end,
                    self.no_of_periods,
                    eptype,
                    eptype_tvar,
                    self.use_return_period,
                    self.returnperiods,
                    self.max_summary_id,
                    self.num_sidxs,
                )
                for data in gen:
                    self.output_data(data, "psept")

            if not output_wheatsheaf_mean:
                return

            maxcounts = get_wheatsheaf_max_count(
                wheatsheaf_items,
                wheatsheaf_items_start_end,
                self.max_summary_id,
            )

            wheatsheaf_mean_items_file = Path(self.lec_files_folder, f"lec_wheatsheaf_mean-items-{outloss_type}.bdat")
            wheatsheaf_mean_items = np.memmap(
                wheatsheaf_mean_items_file,
                dtype=LOSSVEC2MAP_dtype,
                mode="w+",
                shape=(np.sum(maxcounts[maxcounts != -1])),
            )

            wheatsheaf_mean_items_start_end = fill_wheatsheaf_mean_items(
                wheatsheaf_mean_items,
                wheatsheaf_items,
                wheatsheaf_items_start_end,
                maxcounts,
                self.max_summary_id,
                self.num_sidxs,
            )

            gen = write_ept(
                wheatsheaf_mean_items,
                wheatsheaf_mean_items_start_end,
                self.no_of_periods,
                epcalc,
                eptype,
                eptype_tvar,
                self.use_return_period,
                self.returnperiods,
                self.max_summary_id,
                sample_size=self.sample_size
            )

            for data in gen:
                self.output_data(data, "ept")

    def output_sample_mean(self, eptype, eptype_tvar, outloss_type):
        """Output Sample Mean
        Sample Mean Losses – this means do the loss calculation for a year using the statistical
        sample event mean.
        Args:
            eptype (int): Exceedance Probability Type
            eptype_tvar (int): Exceedance Probability Type (Tail Value at Risk)
            outloss_type (string): Which loss to output
        """
        if self.sample_size == 0:
            logger.warning("aggreports.output_sample_mean, self.sample_size is 0, not outputting any sample mean")
            return
        epcalc = MEANSAMPLE

        # outloss_sample has all SIDXs plus -2 and -3
        reordered_outlosses_file = Path(self.lec_files_folder, f"lec_sample_mean-reordered_outlosses-{outloss_type}.bdat")
        reordered_outlosses = np.memmap(
            reordered_outlosses_file,
            dtype=np.dtype([
                ("row_used", np.bool_),
                ("value", oasis_float),
            ]),
            mode="w+",
            shape=(self.no_of_periods * self.max_summary_id),
        )

        # Select the correct outloss values based on type
        # Required if-else condition as njit cannot resolve outloss_type inside []
        if outloss_type == "agg_out_loss":
            outloss_vals = self.outloss_sample["agg_out_loss"]
        elif outloss_type == "max_out_loss":
            outloss_vals = self.outloss_sample["max_out_loss"]
        else:
            raise ValueError(f"Error: Unknown outloss_type: {outloss_type}")

        # Get row indices that are used
        row_used_indices = np.where(self.outloss_sample["row_used"])[0]

        # Reorder outlosses by summary_id and period_no
        reorder_losses_by_summary_and_period(
            reordered_outlosses,
            row_used_indices,
            outloss_vals,
            self.max_summary_id,
            self.no_of_periods,
            self.num_sidxs,
            self.sample_size,
        )

        # Get row indices that are used
        row_used_indices = np.where(reordered_outlosses["row_used"])[0]

        # Allocate storage for the flat data array
        items_fp = Path(self.lec_files_folder, f"lec_sample_mean-{outloss_type}-items.bdat")
        items = np.memmap(items_fp, dtype=LOSSVEC2MAP_dtype, mode="w+", shape=(len(row_used_indices),))
        # Track start and end indices for each summary_id
        items_start_end = np.full((self.max_summary_id, 2), -1, dtype=np.int32)

        # Populate items and items_start_end
        has_weights, used_period_no = output_sample_mean(
            items,
            items_start_end,
            row_used_indices,
            reordered_outlosses["value"],
            self.period_weights,
            self.max_summary_id,
            self.no_of_periods,
        )
        unused_period_weights = self.period_weights[~used_period_no]

        if has_weights:
            gen = write_ept_weighted(
                items,
                items_start_end,
                self.sample_size,
                epcalc,
                eptype,
                eptype_tvar,
                unused_period_weights,
                self.use_return_period,
                self.returnperiods,
                self.max_summary_id
            )
        else:
            gen = write_ept(
                items,
                items_start_end,
                self.no_of_periods,
                epcalc,
                eptype,
                eptype_tvar,
                self.use_return_period,
                self.returnperiods,
                self.max_summary_id
            )

        for data in gen:
            self.output_data(data, "ept")
