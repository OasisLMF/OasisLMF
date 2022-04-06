"""
This file contains the utilities for all the I/O necessary in gulpy.

"""
import os
import numpy as np

from numba import njit

from oasislmf.pytools.getmodel.manager import get_damage_bins
from oasislmf.pytools.getmodel.common import oasis_float, areaperil_int
from oasislmf.pytools.gul.common import ProbMean, damagecdfrec, damagecdfrec_stream
from oasislmf.pytools.gul.common import gulSampleslevelHeader, gulSampleslevelRec, oasis_float_to_int_size

loss_rel_size = 1
gulSampleslevelRec_size = gulSampleslevelRec.size
gulSampleslevelHeader_size = gulSampleslevelHeader.size


def read_getmodel_stream(run_dir, streams_in, buff_size=65536):
    """Read the getmodel output stream.

    Args:
        run_dir ([str]): Path to the run directory.
        streams_in ([]): input streams

    Raises:
        ValueError: If the stream type is not 1.

    Yields:
        All the cdfs related to 1 event.

        damagecdfs: array-like([damagecdf])
        Nbinss: array-like([int]) 
        recs: array-like([ProbMean])
    """
    # determine stream type
    stream_type = np.frombuffer(streams_in.read(4), dtype='i4')

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
    while True:
        if len_read > 0:
            # read stream from valid_buf onwards
            len_read = streams_in.readinto1(mv[valid_buf:])
            # extend the valid_buf by the same amount of data that was read
            valid_buf += len_read

        # read the streamed data into formatted data
        cursor, i, yield_event, event_id, damagecdf, Nbins, rec, last_event_id = stream_to_data(
            int32_mv, valid_buf, size_cdf_entry, max_Nbins, last_event_id)

        if yield_event:
            # event is fully read, append the last chunk of data to the list of this event
            damagecdfs.append(damagecdf[:i])
            Nbinss.append(Nbins[:i])
            recs.append(rec[:i])

            yield last_event_id, np.concatenate(damagecdfs), np.concatenate(Nbinss), np.concatenate(recs)

            # start a new list for the new event, storing the first element
            damagecdfs, Nbinss, recs = [damagecdf[i:i + 1]], [Nbins[i:i + 1]], [rec[i:i + 1]]
            last_event_id = event_id

        else:
            # the current event is not finished, keep appending data about this event
            damagecdfs.append(damagecdf[:i + 1])
            Nbinss.append(Nbins[:i + 1])
            recs.append(rec[:i + 1])

        # convert cursor to bytes
        cursor_buf = cursor * int32_mv.itemsize

        if valid_buf == cursor_buf:
            # this is the last cycle, all data has been read, append the last chunk of data
            damagecdfs.append(damagecdf[:i])
            Nbinss.append(Nbins[:i])
            recs.append(rec[:i])

            yield last_event_id, np.concatenate(damagecdfs), np.concatenate(Nbinss), np.concatenate(recs)
            break

        else:
            # this is not the last cycle
            # move the un-read data to the beginning of the memoryview
            mv[:valid_buf - cursor_buf] = mv[cursor_buf:valid_buf]

            # update the length of the valid data
            valid_buf -= cursor_buf


@njit(cache=True, fastmath=True)
def stream_to_data(int32_mv, valid_buf, size_cdf_entry, max_Nbins, last_event_id):
    """Read streamed data into formatted data.

    Args:
        int32_mv (ndarray): int32 view of the buffer
        valid_buf (int): number of bytes with valid data
        size_cdf_entry (int): size (in bytes) of a single record
        max_Nbins (int): Maximum number of probability bins
        last_event_id (int): event_id of the last event that was completed

    Returns:
        cursor (int): number of int numbers read from the int32_mv ndarray.
        i (int): number of cdf data entries read.
        yield_event (bool): if True, the current event (id=`event_id`) is completed.
        event_id (int): id of the event being read.
        damagecdf (array-like[damagecdf]):
        Nbins (array-like[damagecdf]):
        rec (array-like[damagecdf]):
        last_event_id (int): event_id of the last event that was completed

    # TODO: don't store event_id (it's wasted space). need to change this everywhere `damagecdfrec` is used

    """
    valid_len = valid_buf // size_cdf_entry
    yield_event = False

    Nbins = np.zeros(valid_len, dtype='i4')
    damagecdf = np.zeros(valid_len, dtype=damagecdfrec)
    rec = np.zeros((valid_len, max_Nbins), dtype=ProbMean)

    i = 0
    cursor = 0
    while cursor < valid_len:
        event_id, cursor = int32_mv[cursor], cursor + 1
        damagecdf[i]['areaperil_id'], cursor = int32_mv[cursor:cursor + 1].view(areaperil_int)[0], cursor + 1
        damagecdf[i]['vulnerability_id'], cursor = int32_mv[cursor], cursor + 1
        Nbins[i] = int32_mv[cursor]
        cursor += 1

        for j in range(Nbins[i]):
            rec[i, j]['prob_to'] = int32_mv[cursor: cursor + oasis_float_to_int_size].view(oasis_float)[0]
            cursor += oasis_float_to_int_size
            rec[i, j]['bin_mean'] = int32_mv[cursor: cursor + oasis_float_to_int_size].view(oasis_float)[0]
            cursor += oasis_float_to_int_size

        if event_id != last_event_id:
            # a new event has started
            if last_event_id > 0:
                # this is not the beginning of first event
                yield_event = True

                return cursor, i, yield_event, event_id, damagecdf, Nbins, rec, last_event_id

            last_event_id = event_id

        i += 1

    return cursor, i - 1, yield_event, event_id, damagecdf, Nbins, rec, last_event_id


class LossWriter():

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
        mv_size_bytes = buff_size * 10
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
        self.loss_rel_size = 1

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
        self.int32_mv[self.cursor:self.cursor + self.loss_rel_size].view(oasis_float)[:] = loss
        self.cursor += self.loss_rel_size
        self.cursor_bytes += gulSampleslevelRec.size


@njit(cache=True, fastmath=True)
def write_sample_header(event_id, item_id, int32_mv, cursor, cursor_bytes):
    int32_mv[cursor], cursor = event_id, cursor + 1
    int32_mv[cursor], cursor = item_id, cursor + 1
    cursor_bytes += gulSampleslevelHeader_size

    return cursor, cursor_bytes


@njit(cache=True, fastmath=True)
def write_sample_rec(sidx, loss, int32_mv, cursor, cursor_bytes):

    int32_mv[cursor], cursor = sidx, cursor + 1
    int32_mv[cursor:cursor + loss_rel_size].view(oasis_float)[:] = loss
    cursor += loss_rel_size
    cursor_bytes += gulSampleslevelRec_size

    return cursor, cursor_bytes


@njit(cache=True, fastmath=True)
def write_negative_sidx(max_loss_idx, max_loss, chance_of_loss_idx, chance_of_loss,
                        tiv_idx, tiv, std_dev_idx, std_dev, mean_idx, gul_mean,
                        int32_mv, cursor, cursor_bytes):

    int32_mv[cursor], cursor = max_loss_idx, cursor + 1
    int32_mv[cursor:cursor + loss_rel_size].view(oasis_float)[:] = max_loss
    cursor += loss_rel_size
    cursor_bytes += gulSampleslevelRec_size

    int32_mv[cursor], cursor = chance_of_loss_idx, cursor + 1
    int32_mv[cursor:cursor + loss_rel_size].view(oasis_float)[:] = chance_of_loss
    cursor += loss_rel_size
    cursor_bytes += gulSampleslevelRec_size

    int32_mv[cursor], cursor = tiv_idx, cursor + 1
    int32_mv[cursor:cursor + loss_rel_size].view(oasis_float)[:] = tiv
    cursor += loss_rel_size
    cursor_bytes += gulSampleslevelRec_size

    int32_mv[cursor], cursor = std_dev_idx, cursor + 1
    int32_mv[cursor:cursor + loss_rel_size].view(oasis_float)[:] = std_dev
    cursor += loss_rel_size
    cursor_bytes += gulSampleslevelRec_size

    int32_mv[cursor], cursor = mean_idx, cursor + 1
    int32_mv[cursor:cursor + loss_rel_size].view(oasis_float)[:] = gul_mean
    cursor += loss_rel_size
    cursor_bytes += gulSampleslevelRec_size

    return cursor, cursor_bytes
