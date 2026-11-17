from oasislmf.pytools.common.blended_areaperil import read_bin, record_dtype, to_records
from oasislmf.pytools.common.data import resolve_file, write_ndarray_to_fmt_csv


def blended_areaperil_tocsv(stack, file_in, file_out, file_type, noheader):
    """Write a binary blended areaperil table as csv, with the columns of its weight type."""
    blend = read_bin(resolve_file(file_in, "rb", stack), label=str(file_in))
    _, headers, fmt = record_dtype(blend.weight_type)
    if not noheader:
        file_out.write(",".join(headers) + "\n")
    if len(blend.table):
        write_ndarray_to_fmt_csv(file_out, to_records(blend.table, blend.weight_type), headers, fmt)
