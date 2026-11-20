import logging

from oasislmf.pytools.common.blended_areaperil import WEIGHT_TYPE_NAMES, read_csv, validate, write_bin
from oasislmf.pytools.common.data import resolve_file

logger = logging.getLogger(__name__)


def blended_areaperil_tobin(stack, file_in, file_out, file_type, no_validation=False):
    """Convert a blended areaperil csv to the binary form, its weight type taken from the csv.

    The table is validated as at run time and written sorted by (blended_areaperil_id,
    placement_id). With ``no_validation`` the rows are written as read.
    """
    source = resolve_file(file_in, "r", stack) if str(file_in) == "-" else file_in
    blend = read_csv(source, label=str(file_in))
    if not no_validation:
        blend = validate(blend, label=str(file_in))
    logger.info(f"{len(blend.table)} rows, {WEIGHT_TYPE_NAMES[blend.weight_type]} weights")
    write_bin(file_out, blend.table, blend.weight_type)
