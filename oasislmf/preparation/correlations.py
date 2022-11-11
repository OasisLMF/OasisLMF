"""
This file defines the functions that maps the supported perils with the correlation settings. This data is usually
obtained from the model_settings.
"""
import pandas as pd
from typing import Optional


def map_data(data: Optional[dict]) -> Optional[pd.DataFrame]:
    """
    Maps data from the model settings to to have Peril ID, peril_correlation_group, and correlation_value.

    Args:
        data: (dict) the data loaded from the model settings

    Returns: (pd.DataFrame) the mapped data
    """
    if data is not None:
        supported_perils = data.get("lookup_settings", {}).get("supported_perils", [])
        correlation_settings = data.get("correlation_settings", [])

        for supported_peril in supported_perils:
            supported_peril["peril_correlation_group"] = supported_peril.get("peril_correlation_group", 0)

        supported_perils_df = pd.DataFrame(supported_perils)
        correlation_settings_df = pd.DataFrame(correlation_settings)

        # merge allows duplicates of the "peril_correlation_group" in the supported perils
        # merge does not allow duplicates of the "peril_correlation_group" in the correlation settings
        if len(supported_perils_df) > 0 and len(correlation_settings_df) > 0:
            mapped_data = pd.merge(supported_perils_df, correlation_settings_df, on="peril_correlation_group")
            return mapped_data


def get_correlation_input_items(gul_inputs_df: pd.DataFrame, correlation_map_df: pd.DataFrame) -> pd.DataFrame:
    """
    Gets the correlation values with the peril ID from the model_settings.

    Args:
        correlation_map_df: (pd.DataFrame) data from the model settings to to have Peril ID, peril_correlation_group,
                                           and correlation_value
        gul_inputs_df: (pd.DataFrame) the data of the gul inputs to be mapped

    Returns: (pd.DataFrame) the mapped data of correlations
    """
    gul_inputs_df = gul_inputs_df.merge(correlation_map_df, left_on='peril_id', right_on='id').reset_index()
    gul_inputs_df["correlation_value"] = gul_inputs_df["correlation_value"].astype(float)
    gul_inputs_df = gul_inputs_df.reindex(columns=list(gul_inputs_df))

    correlation_df = gul_inputs_df[["item_id", "peril_correlation_group", "correlation_value"]]
    return correlation_df.sort_values('item_id')
