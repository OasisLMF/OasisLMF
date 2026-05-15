# Loss-factors binary → CSV converter.
#
# The binary is memory-mapped (np.memmap) so only pages that are read are loaded —
# memory usage is proportional to the events being output, not file size.
# The format has no index: events are stored sequentially as
# [event_id i4][count i4][amplification_id i4, factor f4 × count].
# Events are iterated in one pass; their rows are accumulated into a pre-allocated
# output buffer (_BATCH_ROWS rows) that is flushed to write_ndarray_to_fmt_csv in
# one call per batch. Single events that exceed the buffer are written directly.

import sys
import numpy as np

from oasislmf.pytools.common.data import resolve_file, write_ndarray_to_fmt_csv
from oasislmf.pytools.converters.data import TOOL_INFO
from oasislmf.pytools.pla.common import amp_factor_dtype, event_count_dtype

_BATCH_ROWS = 1 << 13  # 8 K rows


def lossfactors_tocsv(stack, file_in, file_out, file_type, noheader):
    headers = TOOL_INFO[file_type]["headers"]
    dtype = TOOL_INFO[file_type]["dtype"]
    fmt = TOOL_INFO[file_type]["fmt"]

    file_in = resolve_file(file_in, "rb", stack)
    if file_in == sys.stdin.buffer:
        raw = np.frombuffer(file_in.read(), dtype='u1')
    else:
        raw = np.memmap(file_in, dtype='u1', mode='r')

    if not noheader:
        file_out.write(",".join(headers) + "\n")

    batch_data = np.empty(_BATCH_ROWS, dtype=dtype)
    batch_pos = 0
    pos = 4  # skip opts header (int32)

    while pos + event_count_dtype.itemsize <= len(raw):
        hdr = raw[pos:pos + event_count_dtype.itemsize].view(event_count_dtype)[0]
        event_id = int(hdr['event_id'])
        count = int(hdr['count'])
        pos += event_count_dtype.itemsize

        body = raw[pos:pos + count * amp_factor_dtype.itemsize].view(amp_factor_dtype)
        pos += count * amp_factor_dtype.itemsize

        if batch_pos + count > _BATCH_ROWS:
            if batch_pos > 0:
                write_ndarray_to_fmt_csv(file_out, batch_data[:batch_pos], headers, fmt)
                batch_pos = 0
            if count > _BATCH_ROWS:
                large_buf = np.empty(count, dtype=dtype)
                large_buf['event_id'] = event_id
                large_buf['amplification_id'] = body['amplification_id']
                large_buf['factor'] = body['factor']
                write_ndarray_to_fmt_csv(file_out, large_buf, headers, fmt)
                continue

        batch_data['event_id'][batch_pos:batch_pos + count] = event_id
        batch_data['amplification_id'][batch_pos:batch_pos + count] = body['amplification_id']
        batch_data['factor'][batch_pos:batch_pos + count] = body['factor']
        batch_pos += count

    if batch_pos > 0:
        write_ndarray_to_fmt_csv(file_out, batch_data[:batch_pos], headers, fmt)
