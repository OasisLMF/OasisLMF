"""
This file defines the functions that maps the supported perils with the correlation settings. This data is usually
obtained from the model_settings.
"""
from typing import Optional

import pandas as pd

from oasislmf.utils.exceptions import OasisException


def map_data(data: Optional[dict], logger) -> Optional[pd.DataFrame]:
    """
    Maps data from the model settings to to have Peril ID, peril_correlation_group, and damage_correlation_value.

    Args:
        data: (dict) the data loaded from the model settings

    Returns: (pd.DataFrame) the mapped data
    """
    if data is not None:
        supported_perils = data.get("lookup_settings", {}).get("supported_perils", [])
        correlations_legacy = data.get("correlation_settings", [])
        correlation_settings = data.get("model_settings", {}).get("correlation_settings", correlations_legacy)

        for supported_peril in supported_perils:  # supported_perils is expected to be a list of dict
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


def get_coverage_dependency_settings(data: Optional[dict], logger) -> list:
    """Extract coverage dependency pairs from the model settings.

    Reads ``model_settings.coverage_dependency_settings``. Each entry links a source
    coverage type to a dependent coverage type; in gulmc the dependent coverage's hazard
    sampling is then driven by the source coverage's per-sample damage ratio.

    Args:
        data (dict): the model settings dictionary (may be None).
        logger: logger.

    Returns:
        list[tuple[int, int]]: list of (source_coverage_type, dependent_coverage_type) pairs.

    Raises:
        OasisException: if an entry is malformed, is a self-reference, or lists a dependent
            coverage type more than once (each dependent must have exactly one source).
    """
    if not data:
        return []
    # canonical location is the nested model_settings block (where correlation_settings now
    # lives; its top-level form is deprecated legacy). No legacy fallback for this new setting.
    settings = data.get("model_settings", {}).get("coverage_dependency_settings", [])

    pairs = []
    seen_dependents = set()
    for entry in settings:
        try:
            source_cov_type = int(entry["source_coverage_type"])
            dependent_cov_type = int(entry["dependent_coverage_type"])
        except (KeyError, TypeError, ValueError) as e:
            raise OasisException(f"Invalid coverage_dependency_settings entry {entry}: {e}")
        if source_cov_type == dependent_cov_type:
            raise OasisException(
                f"Invalid coverage_dependency_settings entry {entry}: a coverage type cannot depend on itself.")
        if dependent_cov_type in seen_dependents:
            raise OasisException(
                f"Invalid coverage_dependency_settings: coverage type {dependent_cov_type} is listed as a dependent "
                "more than once; each dependent coverage type must have exactly one source.")
        seen_dependents.add(dependent_cov_type)
        pairs.append((source_cov_type, dependent_cov_type))
    return pairs
