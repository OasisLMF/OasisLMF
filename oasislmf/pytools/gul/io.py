"""
This file contains the utilities for all the I/O necessary in gulpy.

"""
import os
import numpy as np
import numba as nb
from numba import njit

from numba.types import int_
from numba.typed import Dict, List

from oasislmf.pytools.getmodel.manager import get_damage_bins
from oasislmf.pytools.getmodel.common import oasis_float, areaperil_int
from oasislmf.pytools.gul.common import (
    ProbMean, damagecdfrec, damagecdfrec_stream,
    gulSampleslevelHeader, gulSampleslevelRec, oasis_float_to_int32_size,
    gulSampleslevelRec_size, gulSampleslevelHeader_size,
    BIN_MAP_KEY_TYPE, ITEM_MAP_KEY_TYPE
)
from oasislmf.pytools.gul.utils import append_to_dict_value


@njit(cache=True, fastmath=True)
def init_structures():
    item_bin_ids = Dict.empty(ITEM_MAP_KEY_TYPE, List.empty_list(nb.types.int32))
    bin_map = Dict.empty(BIN_MAP_KEY_TYPE, nb.types.int32)
    bin_lookup = List.empty_list(BIN_MAP_KEY_TYPE)

    return item_bin_ids, bin_map, bin_lookup


def read_getmodel_stream(run_dir, stream_in, buff_size=65536):
    """Read the getmodel output stream yielding data event by event. 

    Args:
        run_dir (str): path to the run directory.
        stream_in (buffer-like): input stream, e.g. `sys.stdin.buffer`.
        buff_size (int): size in bytes of the read buffer. Default is 65536.

    Raises:
        ValueError: If the stream type is not 1.

    Yields:
        int32,  numpy.array[damagecdf], numpy.array[int], numpy.array[ProbMean]:
          event_id, array of damagecdf entries (areaperil_id, vulnerability_id) for this event,
          number of bins in all cdfs of event_id, all the cdfs used in event_id.
    """
    # determine stream type
    stream_type = np.frombuffer(stream_in.read(4), dtype='i4')

    # TODO: make sure the bit1 and bit 2-4 compliance is checked
    # see https://github.com/OasisLMF/ktools/blob/master/docs/md/CoreComponents.md
    if stream_type[0] != 1:
        raise ValueError(f"FATAL: Invalid stream type: expect 1, got {stream_type[0]}.")

    # get damage bins from fileFIXME
    static_path = os.path.join(run_dir, 'static')
    damage_bins = get_damage_bins(static_path=static_path)

    # maximum number of damage bins (individual items can have up to `total_bins` bins)
    if damage_bins.shape[0] == 0:
        max_Nbins = 1000
    else:
        max_Nbins = damage_bins.shape[0]

    # maximum number of entries is buff_size divided by the minimum entry size
    # (corresponding to a 1-bin only cdf)
    min_size_cdf_entry = damagecdfrec_stream.size + ProbMean.size + 4

    # each record from getmodel stream is expected to contain:
    # 1 damagecdfrec_stream obj, 1 int (Nbins), a number `Nbins` of ProbMean objects

    # use a memoryview of size twice the buff_size to ensure `read_into1` always reads the maximum amount
    # data possible (the largest between the pipe limit and the remaining memory to fill the memoryview)
    mv = memoryview(bytearray(buff_size * 2))
    int32_mv = np.ndarray(buff_size * 2 // 4, buffer=mv, dtype='i4')

    cursor = 0
    valid_buf = 0
    last_event_id = -1
    damagecdfs, Nbinss, recs = [], [], []
    len_read = 1
    size_cdf_entry = min_size_cdf_entry

    item_bin_ids, bin_map, bin_lookup = init_structures()

    while True:
        if len_read > 0:
            # read stream from valid_buf onwards
            len_read = stream_in.readinto1(mv[valid_buf:])
            # extend the valid_buf by the same amount of data that was read
            valid_buf += len_read

        # read the streamed data into formatted data
        cursor, i, yield_event, event_id, last_event_id, bin_map, bin_lookup, item_bin_ids = stream_to_data(
            int32_mv, valid_buf, size_cdf_entry, max_Nbins, last_event_id, bin_map, bin_lookup, item_bin_ids)

        if yield_event:
            # event is fully read, append the last chunk of data to the list of this event
            # damagecdfs.append(damagecdf[:i])
            # Nbinss.append(Nbins[:i])
            # recs.append(rec[:i])

            # convert to np array to allow slicing `prob_to` and `bin_mean`
            bin_lookup = np.array(bin_lookup)
            # bin_lookup_ndarr = np.empty((len(bin_lookup),), dtype=ProbMean)

            # bin_lookup_ndarr[:]['prob_to'] = bin_lookup[:, 0]
            # bin_lookup_ndarr[:]['bin_mean'] = bin_lookup[:, 1]

            yield last_event_id, bin_lookup, item_bin_ids

            # start a new list for the new event, storing the first element
            # damagecdfs, Nbinss, recs = [damagecdf[i:i + 1]], [Nbins[i:i + 1]], [rec[i:i + 1]]
            last_event_id = event_id

            item_bin_ids, bin_map, bin_lookup = init_structures()

        # else:
        #     # the current event is not finished, keep appending data about this event
        #     damagecdfs.append(damagecdf[:i + 1])
        #     Nbinss.append(Nbins[:i + 1])
        #     recs.append(rec[:i + 1])

        # convert cursor to bytes
        cursor_buf = cursor * int32_mv.itemsize

        if valid_buf == cursor_buf:
            # # this is the last cycle, all data has been read, append the last chunk of data
            # damagecdfs.append(damagecdf[:i])
            # Nbinss.append(Nbins[:i])
            # recs.append(rec[:i])

            # np.concatenate(damagecdfs), np.concatenate(Nbinss), np.concatenate(recs)

            # convert to np array to allow slicing `prob_to` and `bin_mean`
            bin_lookup = np.array(bin_lookup)
            # bin_lookup_ndarr = np.empty((len(bin_lookup),), dtype=ProbMean)

            # bin_lookup_ndarr[:]['prob_to'] = bin_lookup[:, 0]
            # bin_lookup_ndarr[:]['bin_mean'] = bin_lookup[:, 1]

            yield last_event_id, bin_lookup, item_bin_ids
            break

        else:
            # this is not the last cycle
            # move the un-read data to the beginning of the memoryview
            mv[:valid_buf - cursor_buf] = mv[cursor_buf:valid_buf]

            # update the length of the valid data
            valid_buf -= cursor_buf


@njit(cache=True, fastmath=True)
def stream_to_data(int32_mv, valid_buf, size_cdf_entry, max_Nbins, last_event_id,
                   bin_map, bin_lookup, item_bin_ids):
    """Parse streamed data into data arrays.

    Args:
        int32_mv (ndarray): int32 view of the buffer
        valid_buf (int): number of bytes with valid data
        size_cdf_entry (int): size (in bytes) of a single record
        max_Nbins (int): Maximum number of probability bins
        last_event_id (int): event_id of the last event that was completed

    Returns:
        int, int, bool, int, numpy.array[damagecdf], numpy.array[int32], numpy.array[ProbMean], int:
          number of int numbers read from the int32_mv ndarray, number of cdf data entries read,
          whether the current event (id=`event_id`) has been fully read, id of the event being read,
          chunk of damagecdf entries, chunk of Nbins entries, chunk of cdf record entries,
          event_id of the last event that was completed
    """
    valid_len = valid_buf // size_cdf_entry
    yield_event = False

    # Nbins = np.zeros(valid_len, dtype='i4')
    # damagecdf = np.zeros(valid_len, dtype=damagecdfrec)
    # rec = np.zeros((valid_len, max_Nbins), dtype=ProbMean)
    # bin_map = Dict.empty(nb.types.UniTuple(nb.types.float32, 2), int_)

    i = 0
    cursor = 0
    while cursor < valid_len:
        event_id, cursor = int32_mv[cursor], cursor + 1

        if event_id != last_event_id:
            # a new event has started
            if last_event_id > 0:
                # if this is not the beginning of the very first event, yield the event we just completed
                yield_event = True
                cursor -= 1  # go back by one to re-read event_id

                return cursor, i, yield_event, event_id, last_event_id, bin_map, bin_lookup, item_bin_ids

            # <-- this line is only executed once for last_event_id == 0 because if >0 it returns.
            last_event_id = event_id
            # if so, change the if to be clearer eg if == 0: last_event_id = event_id else: yield... return.

        # damagecdf[i]['areaperil_id'], cursor = int32_mv[cursor:cursor + 1].view(areaperil_int)[0], cursor + 1
        # damagecdf[i]['vulnerability_id'], cursor = int32_mv[cursor], cursor + 1

        areaperil_id, cursor = int32_mv[cursor:cursor + 1].view(areaperil_int)[0], cursor + 1
        vulnerability_id, cursor = int32_mv[cursor], cursor + 1
        # Nbins[i] = int32_mv[cursor] # TODO probably not needed anymore
        Nbins = int32_mv[cursor]  # TODO probably not needed anymore
        cursor += 1

        # for j in range(Nbins[i]):
        #     rec[i, j]['prob_to'] = int32_mv[cursor: cursor + oasis_float_to_int32_size].view(oasis_float)[0]
        #     cursor += oasis_float_to_int32_size
        #     rec[i, j]['bin_mean'] = int32_mv[cursor: cursor + oasis_float_to_int32_size].view(oasis_float)[0]
        #     cursor += oasis_float_to_int32_size

        bin_ids = List.empty_list(nb.types.int32)
        for j in range(Nbins):
            prob_to = int32_mv[cursor: cursor + oasis_float_to_int32_size].view(oasis_float)[0]
            cursor += oasis_float_to_int32_size
            bin_mean = int32_mv[cursor: cursor + oasis_float_to_int32_size].view(oasis_float)[0]
            cursor += oasis_float_to_int32_size

            bin_key = tuple((prob_to, bin_mean))

            if bin_key not in bin_map:
                bin_map[bin_key] = np.int32(len(bin_map))

                # better to store the bin as a List and not as tuple
                # since this allows us to easily cast all the bins in a 2d numpy array later
                # myl = List.empty_list()
                # myl.append(List([prob_to, bin_mean]))
                bin_lookup.append(bin_key)

            bin_ids.append(bin_map[bin_key])

        # let's not store Nbins for now. let's recompute it with len(bin_ids)
        item_bin_ids[tuple((areaperil_id, vulnerability_id))] = bin_ids

        # to be deleted
        # if event_id != last_event_id:
        #     # a new event has started
        #     if last_event_id > 0:
        #         # if this is not the beginning of the very first event, yield the event we just completed
        #         yield_event = True

        #         return cursor, i, yield_event, event_id, last_event_id, bin_map, bin_lookup, item_bin_ids

        #     # <-- this line is only executed once for last_event_id == 0 because if >0 it returns.
        #     last_event_id = event_id
        #     # if so, change the if to be clearer eg if == 0: last_event_id = event_id else: yield... return.
        # to be deleted end

        i += 1  # maybe not used anymore

    return cursor, i - 1, yield_event, event_id, last_event_id, bin_map, bin_lookup, item_bin_ids


class LossWriter():
    # TODO clean this up and remove unused features
    def __init__(self, lossout, len_sample, buff_size=65536) -> None:

        # number of bytes to read at a given time.
        # number_size = 8 works only if loss in gulSampleslevelRec is float32.
        self.number_size = max(gulSampleslevelHeader.size, gulSampleslevelRec.size)  # bytes

        self.len_sample = len_sample
        self.lossout = lossout
        self.buff_size = buff_size  # bytes

        # compute how many numbers of size `number_size` fit in the buffer
        # for safety, take 1000 less than the compute number to allow flushing the buffer not too often
        # if -1 instead of -1000 is taken, it requires checking whether to flush or not for every write to mv.
        self.buff_safety = self.number_size * 1000
        self.nb_number = (self.buff_size + self.buff_safety) // self.number_size
        self.flush_number = self.nb_number - 4

        # define the raw memory view, the int32 view of it, and their respective cursors
        mv_size_bytes = buff_size * 2
        # mv_size_bytes = buff_size + self.number_size * 10  # doesn't work, produces wrong output (zeroes)
        self.mv = memoryview(bytearray(mv_size_bytes))
        self.int32_mv = np.ndarray(mv_size_bytes // self.number_size, buffer=self.mv, dtype='i4')
        # cannot use because the header is int int
        # self.loss_mv = np.ndarray(self.nb_number, buffer=self.mv, dtype=gulSampleslevelRec.dtype)
        # cannot use two views loss_mv and header_mv because it only works if oasis_float is float32.
        # if oasis_float is set to float64, the cursor will not map correctly both mv.
        self.cursor_bytes = 0
        self.cursor = 0

        # size(oasis_float)/size(i4)
        # TODO find a way to do that programmatically and test if this works with oasis_float=float64
        self.oasis_float_to_int32_size = 1

    def flush(self):
        # print("FLUSHING ", self.cursor, " ", self.cursor_bytes)
        self.lossout.write(self.mv[:self.cursor_bytes])
        self.cursor_bytes = 0
        self.cursor = 0
        # print("FLUSHED ", self.cursor, " ", self.cursor_bytes)

    def write_sample_header(self, event_id, item_id):
        self.int32_mv[self.cursor] = event_id
        self.cursor += 1
        self.int32_mv[self.cursor] = item_id
        self.cursor += 1
        self.cursor_bytes += gulSampleslevelHeader.size

    def write_sample_rec(self, sidx, loss):

        if self.cursor >= self.flush_number:
            self.flush()

        self.int32_mv[self.cursor] = sidx
        self.cursor += 1
        self.int32_mv[self.cursor:self.cursor + self.oasis_float_to_int32_size].view(oasis_float)[:] = loss
        self.cursor += self.oasis_float_to_int32_size
        self.cursor_bytes += gulSampleslevelRec.size


@njit(cache=True, fastmath=True)
def write_sample_header(event_id, item_id, int32_mv, cursor, cursor_bytes):
    """Write to buffer the header for the samples of this (event, item).

    Args:
        event_id (int32): event id.
        item_id (int32): item id.
        int32_mv (numpy.ndarray): int32 view of the memoryview where the output is buffered.
        cursor (int): index of int32_mv where to start writing.
        cursor_bytes (int): number of bytes written in int32_mv.

    Returns:
        int, int: updated values of cursor and cursor_bytes
    """
    int32_mv[cursor], cursor = event_id, cursor + 1
    int32_mv[cursor], cursor = item_id, cursor + 1
    cursor_bytes += gulSampleslevelHeader_size

    return cursor, cursor_bytes


@njit(cache=True, fastmath=True)
def write_sample_rec(sidx, loss, int32_mv, cursor, cursor_bytes):
    """Write to buffer a (sidx, loss) sample record.

    Args:
        sidx (int32): sidx number.
        loss (oasis_float): loss.
        int32_mv (numpy.ndarray): int32 view of the memoryview where the output is buffered.
        cursor (int): index of int32_mv where to start writing.
        cursor_bytes (int): number of bytes written in int32_mv.

    Returns:
        int, int: updated values of cursor and cursor_bytes
    """
    int32_mv[cursor], cursor = sidx, cursor + 1
    int32_mv[cursor:cursor + oasis_float_to_int32_size].view(oasis_float)[:] = loss
    cursor += oasis_float_to_int32_size
    cursor_bytes += gulSampleslevelRec_size

    return cursor, cursor_bytes


@njit(cache=True, fastmath=True)
def write_negative_sidx(max_loss_idx, max_loss, chance_of_loss_idx, chance_of_loss,
                        tiv_idx, tiv, std_dev_idx, std_dev, mean_idx, gul_mean,
                        int32_mv, cursor, cursor_bytes):
    """Write to buffer the negative sidx samples.

    Args:
        max_loss_idx (int32): max_loss_idx sidx number.
        max_loss (oasis_float): max_loss.
        chance_of_loss_idx (int32): chance_of_loss_idx sidx number.
        chance_of_loss_idx (oasis_float): chance_of_loss
        tiv_idx (int32): tiv_idx sidx number.
        tiv (oasis_float): tiv.
        std_dev_idx (int32): std_dev_idx sidx number.
        std_dev (oasis_float): std_dev.
        mean_idx (int32): mean_idx sidx number.
        gul_mean (oasis_float): gul_mean.
        int32_mv (numpy.ndarray): int32 view of the memoryview where the output is buffered.
        cursor (int): index of int32_mv where to start writing.
        cursor_bytes (int): number of bytes written in int32_mv.

    Returns:
        int, int: updated values of cursor and cursor_bytes
    """
    int32_mv[cursor], cursor = max_loss_idx, cursor + 1
    int32_mv[cursor:cursor + oasis_float_to_int32_size].view(oasis_float)[:] = max_loss
    cursor += oasis_float_to_int32_size
    cursor_bytes += gulSampleslevelRec_size

    int32_mv[cursor], cursor = chance_of_loss_idx, cursor + 1
    int32_mv[cursor:cursor + oasis_float_to_int32_size].view(oasis_float)[:] = chance_of_loss
    cursor += oasis_float_to_int32_size
    cursor_bytes += gulSampleslevelRec_size

    int32_mv[cursor], cursor = tiv_idx, cursor + 1
    int32_mv[cursor:cursor + oasis_float_to_int32_size].view(oasis_float)[:] = tiv
    cursor += oasis_float_to_int32_size
    cursor_bytes += gulSampleslevelRec_size

    int32_mv[cursor], cursor = std_dev_idx, cursor + 1
    int32_mv[cursor:cursor + oasis_float_to_int32_size].view(oasis_float)[:] = std_dev
    cursor += oasis_float_to_int32_size
    cursor_bytes += gulSampleslevelRec_size

    int32_mv[cursor], cursor = mean_idx, cursor + 1
    int32_mv[cursor:cursor + oasis_float_to_int32_size].view(oasis_float)[:] = gul_mean
    cursor += oasis_float_to_int32_size
    cursor_bytes += gulSampleslevelRec_size

    return cursor, cursor_bytes
