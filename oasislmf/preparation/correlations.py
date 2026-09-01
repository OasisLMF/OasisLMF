"""Defines the functions that maps the supported perils with the correlation settings. This data is usually
obtained from the model_settings.
"""
from typing import Optional

import pandas as pd

from oasislmf.utils.exceptions import OasisException


def map_data(data: Optional[dict], logger) -> Optional[pd.DataFrame]:
    """Maps data from the model settings to to have Peril ID, peril_correlation_group, and damage_correlation_value.

    Args:
        data: (dict) the data loaded from the model settings
        logger: logger used to warn when the model settings hold no correlation data

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


def get_coverage_dependency_settings(data: Optional[dict]) -> list:
    """Extract coverage dependency pairs from the model settings.

    Reads ``model_settings.coverage_dependency_settings``. Each entry links a source
    coverage type to a dependent coverage type; in gulmc the dependent coverage's damage is
    then driven by the source coverage's per-sample sampled damage bin, through the dependent's
    conditional (damage-transition) vulnerability.

    Args:
        data (dict): the model settings dictionary (may be None).

    Returns:
        list[tuple[int, int]]: list of (source_coverage_type, dependent_coverage_type) pairs.

    Raises:
        OasisException: if an entry is malformed, is a self-reference, lists a dependent coverage
            type more than once (each dependent must have exactly one source), or closes a
            dependency cycle.

    Examples:
        Contents (3) driven by buildings (1):

        >>> get_coverage_dependency_settings({"model_settings": {"coverage_dependency_settings": [
        ...     {"source_coverage_type": 1, "dependent_coverage_type": 3}]}})
        [(1, 3)]

        A source may drive several dependents, and a dependent may itself be a source, so the pairs
        form a forest — here buildings drive contents (3) and other (2), and contents drive BI (4):

        >>> get_coverage_dependency_settings({"model_settings": {"coverage_dependency_settings": [
        ...     {"source_coverage_type": 1, "dependent_coverage_type": 3},
        ...     {"source_coverage_type": 3, "dependent_coverage_type": 4},
        ...     {"source_coverage_type": 1, "dependent_coverage_type": 2}]}})
        [(1, 3), (3, 4), (1, 2)]

        A cycle is refused, naming the entry that closes it and the types it runs through. Every
        entry here is individually valid — no self-reference, no repeated dependent — so the cycle
        only becomes visible when the third entry closes 1 -> 3 -> 4 -> 1:

        >>> from oasislmf.utils.exceptions import OasisException
        >>> try:
        ...     get_coverage_dependency_settings({"model_settings": {"coverage_dependency_settings": [
        ...         {"source_coverage_type": 1, "dependent_coverage_type": 3},
        ...         {"source_coverage_type": 3, "dependent_coverage_type": 4},
        ...         {"source_coverage_type": 4, "dependent_coverage_type": 1}]}})
        ... except OasisException as e:
        ...     print(e)  # doctest: +ELLIPSIS
        Invalid coverage_dependency_settings entry ... closes a dependency cycle over coverage types [1, 4, 3]; ...
    """
    if not data:
        return []
    settings = data.get("model_settings", {}).get("coverage_dependency_settings", [])

    pairs = []
    source_of = {}                        # dependent coverage type -> its source
    for entry in settings:
        try:
            source_cov_type = int(entry["source_coverage_type"])
            dependent_cov_type = int(entry["dependent_coverage_type"])
        except (KeyError, TypeError, ValueError) as e:
            raise OasisException(f"Invalid coverage_dependency_settings entry {entry}: {e}")
        if source_cov_type == dependent_cov_type:
            raise OasisException(
                f"Invalid coverage_dependency_settings entry {entry}: a coverage type cannot depend on itself.")
        if dependent_cov_type in source_of:
            raise OasisException(
                f"Invalid coverage_dependency_settings: coverage type {dependent_cov_type} is listed as a dependent "
                "more than once; each dependent coverage type must have exactly one source.")

        # Each dependent has exactly one source, so the pairs form a functional graph and this entry
        # closes a cycle if its source already reaches its dependent. Walking up from the source
        # terminates because the dependent is not yet a key. gulmc rejects cycles too, but by
        # coverage_id, which does not point back at the entry that caused it.
        chain, node = [dependent_cov_type], source_cov_type
        while node != dependent_cov_type:
            chain.append(node)
            if node not in source_of:
                break
            node = source_of[node]
        else:
            raise OasisException(
                f"Invalid coverage_dependency_settings entry {entry} closes a dependency cycle over "
                f"coverage types {chain}; the source/dependent pairs must form a directed acyclic graph.")

        source_of[dependent_cov_type] = source_cov_type
        pairs.append((source_cov_type, dependent_cov_type))
    return pairs
