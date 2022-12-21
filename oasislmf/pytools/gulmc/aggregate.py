"""
This file contains specific functionality needed for aggregate vulnerabilities.

"""
import numba as nb
from numba import njit
from numba.typed import Dict, List
from numba.types import int32 as nb_int32

from oasislmf.pytools.getmodel.common import areaperil_int

AGG_VULN_WEIGHTS_KEY_TYPE = nb.types.Tuple((nb.from_dtype(areaperil_int), nb.types.int32))
AGG_VULN_WEIGHTS_VAL_TYPE = nb.types.int32


@njit(cache=True)
def gen_empty_agg_vuln_to_vuln_ids():
    """TODO

    Returns:
        _type_: _description_
    """
    return Dict.empty(nb_int32, List.empty_list(nb_int32))


@njit(cache=True)
def gen_empty_areaperil_vuln_ids_to_weights():
    """TODO

    Returns:
        _type_: _description_
    """
    return Dict.empty(AGG_VULN_WEIGHTS_KEY_TYPE, AGG_VULN_WEIGHTS_VAL_TYPE)


@njit(cache=True)
def map_agg_vuln_ids_to_agg_vuln_idxs(agg_vulns, agg_vuln_to_vuln_ids, vuln_dict):
    """For each aggregate vulnerability listed in `agg_vulns`, map the individual vulnerability_ids that compose it
    to the indices where they are stored in `vuln_array`.

    Args:
        agg_vulns (List[int32])
    """
    agg_vuln_to_vuln_idxs = Dict.empty(nb_int32, List.empty_list(nb_int32))

    for agg in agg_vulns:
        agg_vuln_to_vuln_idxs[agg] = List([vuln_dict[vuln] for vuln in agg_vuln_to_vuln_ids[agg]])

    return agg_vuln_to_vuln_idxs


@njit(cache=True)
def map_areaperil_vuln_id_to_weight_to_areaperil_vuln_idx_to_weight(areaperil_to_vulns, ap_vuln_weights, vuln_dict):
    """Make map from aggregate vulnerability id to the list of sub-vulnerability ids of which they are composed, where the
    value of the sub-vulnerability id is the internal pointer to the dense array where they are stored.
    """
    ap_vuln_idx_weights = {}

    for ap in areaperil_to_vulns:
        for vuln in areaperil_to_vulns[ap]:
            ap_vuln_idx_weights[tuple((ap, vuln_dict[vuln]))] = ap_vuln_weights[tuple((ap, vuln))]

    return ap_vuln_idx_weights
