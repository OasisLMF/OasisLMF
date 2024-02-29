"""
This file defines the functions that maps the supported perils with the correlation settings. This data is usually
obtained from the model_settings.
"""
from typing import Optional

import pandas as pd


def map_data(data: Optional[dict], logger) -> Optional[pd.DataFrame]:
    """
    Maps data from the model settings to to have Peril ID, peril_correlation_group, and damage_correlation_value.

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

        if len(correlation_settings_df) > 0:
            # correlations_settings are defined
            if "damage_correlation_value" not in correlation_settings_df.columns:
                logger.info("Correlation settings: No `damage_correlation_value` found")
                correlation_settings_df["damage_correlation_value"] = 0

            if "hazard_correlation_value" not in correlation_settings_df.columns:
                logger.info("Correlation settings: No `hazard_correlation_value` found")
                correlation_settings_df["hazard_correlation_value"] = 0

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
                                           and damage_correlation_value
        gul_inputs_df: (pd.DataFrame) the data of the gul inputs to be mapped

    Returns: (pd.DataFrame) the mapped data of correlations
    """
    correlation_df = (
        gul_inputs_df
        .merge(correlation_map_df, left_on='peril_id', right_on='id')
        .reset_index()
        .astype({"damage_correlation_value": "float32", "hazard_correlation_value": "float32"})
        [["item_id", "peril_correlation_group", "damage_correlation_value", "hazard_group_id", "hazard_correlation_value"]]
        .sort_values('item_id')
    )

    return correlation_df
