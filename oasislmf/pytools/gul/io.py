"""
This file contains the utilities for all the I/O necessary in gulpy.

"""
import os
from tokenize import group
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
    ITEMS_DATA_MAP_TYPE, COVERAGE_ID_TYPE, items_data_type, NP_BASE_ARRAY_SIZE
)
from oasislmf.pytools.gul.utils import append_to_dict_value
from oasislmf.pytools.gul.random import generate_hash

ProbMean_size = ProbMean.size


@njit(cache=True)
def gen_structs():
    group_id_rng_index = Dict.empty(nb.types.int32, nb.types.int64)
    rec_idx_ptr = List([0])

    return group_id_rng_index, rec_idx_ptr


def read_getmodel_stream(run_dir, stream_in, item_map, coverages, compute, seeds, buff_size=65536):
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
    min_size_cdf_entry = damagecdfrec_stream.size + 4 + ProbMean.size

    # each record from getmodel stream is expected to contain:
    # 1 damagecdfrec_stream obj, 1 int (Nbins), a number `Nbins` of ProbMean objects

    # use a memoryview of size twice the buff_size to ensure `read_into1` always reads the maximum amount
    # data possible (the largest between the pipe limit and the remaining memory to fill the memoryview)
    mv = memoryview(bytearray(buff_size * 2))
    int32_mv = np.ndarray(buff_size * 2 // 4, buffer=mv, dtype='i4')

    cursor = 0
    valid_buf = 0
    last_event_id = -1
    len_read = 1
    size_cdf_entry = min_size_cdf_entry

    group_id_rng_index, rec_idx_ptr = gen_structs()
    rng_index = 0
    damagecdf_i = 0
    compute_i = 0
    items_data_i = 0
    # rec_idx_ptr = [0]
    recs = []
    end_of_stream = False

    items_data = np.empty(2 ** NP_BASE_ARRAY_SIZE, dtype=items_data_type)
    while not end_of_stream:
        if len_read > 0:
            # read stream from valid_buf onwards
            # TODO use selELECT
            len_read = stream_in.readinto1(mv[valid_buf:])
            # extend the valid_buf by the same amount of data that was read
            valid_buf += len_read

        # read the streamed data into formatted data
        cursor, yield_event, event_id, rec, rec_idx_ptr, last_event_id, compute_i, items_data_i, items_data, rng_index, group_id_rng_index, damagecdf_i = stream_to_data(
            int32_mv, valid_buf, size_cdf_entry, max_Nbins, last_event_id, item_map, coverages, compute_i, compute, items_data_i, items_data, seeds, rng_index, group_id_rng_index, damagecdf_i, rec_idx_ptr)

        # convert cursor to bytes
        cursor_buf = cursor * int32_mv.itemsize

        recs.append(rec[:rec_idx_ptr[-1]])  # <-- check if index is OK

        if yield_event:
            # event is fully read, append the last chunk of data to the list of this event
            # damagecdfs.append(damagecdf[:i])
            # Nbinss.append(Nbins[:i])
            # rec_idx_ptrs.append(rec_idx_ptr[:i])

            # item_ids_by_coverage_id = Dict.empty(int_, List.empty_list(ITEM_ID_TYPE))
            # list_coverage_ids = List.empty_list(COVERAGE_ID_TYPE)

            # convert list_coverage_ids to np.array to apply np.unique() to it
            # coverage_ids_tmp = np.empty(Ncoverage_ids, dtype=COVERAGE_ID_TYPE)
            # for j in range(Ncoverage_ids):
            #     coverage_ids_tmp[j] = list_coverage_ids[j]

            # recs.append(rec[:rec_idx_ptr[-1]+1]) # <-- check if index is OK

            yield last_event_id, compute_i, items_data, np.concatenate(recs), rec_idx_ptr, rng_index
            # print("> YIELDED EVENT")
            # if valid_buf == cursor_buf:
            #     # this was the last event
            #     print("********** BREAK")
            #     break

            # start a new list for the new event, storing the first element
            # damagecdfs, Nbinss, rec_idx_ptrs = [damagecdf[i:i + 1]], [Nbins[i:i + 1]], [rec_idx_ptr[i:i + 1]]
            last_event_id = event_id

            group_id_rng_index, rec_idx_ptr = gen_structs()
            rng_index = 0
            damagecdf_i = 0
            compute_i = 0
            items_data_i = 0
            coverages['cur_items'].fill(0)
            # rec_idx_ptr = [0]
            recs = []
            # print("> NEW  EVENT")

        else:
            if valid_buf == cursor_buf and len_read == 0:
                # the stream has ended, this is the last event
                # print("********** BREAK")
                end_of_stream = True

                yield last_event_id, compute_i, items_data, np.concatenate(recs), rec_idx_ptr, rng_index
                # print("> YIELDED LAST EVENT")

        # print("==> CONTINUE  EVENT")

        # if last_event_id > 1400:
        #     print("valid_buf, cursor_buf, len_read: ", valid_buf, cursor_buf, len_read)

        # else:
        # the current event is not finished, keep appending data about this event
        # recs.append(rec[:rec_idx_ptr[-1]+1]) # <-- check if index is OK

        # damagecdfs.append(damagecdf[:i + 1])
        # Nbinss.append(Nbins[:i + 1])
        # rec_idx_ptrs.append(rec_idx_ptr[:i + 1])

        # this is not the last cycle. if here, valid_buf != cursor_buf
        # move the un-read data to the beginning of the memoryview
        mv[:valid_buf - cursor_buf] = mv[cursor_buf:valid_buf]

        # update the length of the valid data
        valid_buf -= cursor_buf

        # if valid_buf == cursor_buf:
        #     # this is the last cycle, all data has been read, append the last chunk of data
        #     # REPEATED CODE

        #     items_data_by_coverage_id = Dict.empty(int_, List.empty_list(ITEMS_DATA_MAP_TYPE))
        #     list_coverage_ids = List.empty_list(COVERAGE_ID_TYPE)

        #     rng_index = 0
        #     seeds = []
        #     group_id_rng_index = {}
        #     Ncoverage_ids = 0

        #     for damagecdf_i, damagecdf in enumerate(damagecdfs):
        #         item_key = tuple((damagecdf['areaperil_id'], damagecdf['vulnerability_id']))

        #         for item in item_map[item_key]:
        #             item_id, coverage_id, group_id = item

        #             if group_id not in group_id_rng_index:
        #                 group_id_rng_index[group_id] = rng_index
        #                 seeds.append(generate_hash(group_id, last_event_id))
        #                 this_rng_index = rng_index
        #                 rng_index += 1
        #             else:
        #                 this_rng_index = group_id_rng_index[group_id]

        #             # append always, will filter the unqiue list_coverage_ids later
        #             # for `list_coverage_ids` list is preferable over set because order is important
        #             list_coverage_ids.append(coverage_id)
        #             Ncoverage_ids += 1

        #             append_to_dict_value(items_data_by_coverage_id, coverage_id,
        #                                  (item_id, damagecdf_i, this_rng_index), ITEMS_DATA_MAP_TYPE)
        #             # append_to_dict_value(item_ids_by_coverage_id, coverage_id, item_id, nb.types.int32)

        #     # item_ids_by_coverage_id = Dict.empty(int_, List.empty_list(ITEM_ID_TYPE))
        #     list_coverage_ids = List.empty_list(COVERAGE_ID_TYPE)

        #     # convert list_coverage_ids to np.array to apply np.unique() to it
        #     coverage_ids_tmp = np.empty(Ncoverage_ids, dtype=COVERAGE_ID_TYPE)
        #     for j in range(Ncoverage_ids):
        #         coverage_ids_tmp[j] = list_coverage_ids[j]

        #     coverage_ids = np.unique(coverage_ids_tmp)
        #     Nunique_coverage_ids = coverage_ids.shape[0]

        #     yield last_eventid, items_data_by_coverage_id, coverage_ids, Nunique_coverage_ids, rec, rec_idx_ptr, seeds

        #     # END REPEATED CODE

        #     # yield last_event_id, np.concatenate(damagecdfs), np.concatenate(Nbinss), np.concatenate(recs)
        #     break

        # else:
        #     # this is not the last cycle
        #     # move the un-read data to the beginning of the memoryview
        #     mv[:valid_buf - cursor_buf] = mv[cursor_buf:valid_buf]

        #     # update the length of the valid data
        #     valid_buf -= cursor_buf


@njit(cache=True, fastmath=True)
def insert_item_data_val(items_data, item_id, damagecdf_i, rng_index):
    # assumes all item_ids are different
    for i in range(items_data.shape[0] - 1):
        if items_data[i]['item_id'] > item_id:
            items_data[i+1:] = items_data[i:-1]
            items_data[i]['item_id'] = item_id
            items_data[i]['damagecdf_i'] = damagecdf_i
            items_data[i]['rng_index'] = rng_index
            return
    items_data[-1]['item_id'] = item_id
    items_data[-1]['damagecdf_i'] = damagecdf_i
    items_data[-1]['rng_index'] = rng_index


@njit(cache=True, fastmath=True)
def stream_to_data(int32_mv, valid_buf, size_cdf_entry, max_Nbins, last_event_id, item_map, coverages,
                   compute_i, compute, items_data_i, items_data, seeds, rng_index, group_id_rng_index, damagecdf_i, rec_idx_ptr):
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
    yield_event = False
    # print("valid_buf, size_cdf_entry, valid_len", valid_buf, size_cdf_entry, valid_len)
    # Nbins = np.zeros(valid_len, dtype='i4')
    # damagecdf = np.zeros(valid_len, dtype=damagecdfrec)
    # conservative choice: size as if the entire buffer is filled with cdf bins
    # `rec` is a temporary buffer to store the cdf being read
    # TODO: rec could be re-used, so it'd be possible to pass it in input to this functio.
    rec = np.zeros(valid_buf // ProbMean_size, dtype=ProbMean)
    # rec_idx_ptr = np.zeros(valid_len + 1, dtype='i4')
    
    cursor = 0
    # damagecdf_i = 0

    # init a counter for the local `rec` tempoary
    last_rec_idx_ptr = 0
    while cursor * int32_mv.itemsize + size_cdf_entry <= valid_buf:

        event_id, cursor = int32_mv[cursor], cursor + 1
        
        if event_id != last_event_id:
            # a new event has started
            if last_event_id > 0:
                # if this is not the beginning of the very first event, yield the event we just completed
                yield_event = True

                # print(valid_buf, len(rec), rec_idx_ptr[-1])
                return cursor-1, yield_event, event_id, rec, rec_idx_ptr, last_event_id, compute_i, items_data_i, items_data, rng_index, group_id_rng_index, damagecdf_i

            last_event_id = event_id

        # damagecdf[i]['areaperil_id'], cursor = int32_mv[cursor:cursor + 1].view(areaperil_int)[0], cursor + 1
        # damagecdf[i]['vulnerability_id'], cursor = int32_mv[cursor], cursor + 1
        areaperil_id, cursor = int32_mv[cursor:cursor + 1].view(areaperil_int)[0], cursor + 1
        vulnerability_id, cursor = int32_mv[cursor], cursor + 1
        Nbins_to_read, cursor = int32_mv[cursor], cursor + 1
        
        if cursor * int32_mv.itemsize + Nbins_to_read * ProbMean_size > valid_buf:
            # if it is possible that the next record is not fully contained in the valid buf
            # (Nbins_to_read * ProbMean_size is the largest possible record), return to get more
            # data in the buffer and put cursor back at the beginning of this cdf
            yield_event = False  # probably not needed, TODO check

            return cursor-4, yield_event, event_id, rec, rec_idx_ptr, last_event_id, compute_i, items_data_i, items_data, rng_index, group_id_rng_index, damagecdf_i

        # read damage cdf bins
        start_rec = last_rec_idx_ptr
        end_rec = start_rec + Nbins_to_read
        for j in range(start_rec, end_rec, 1):
            rec[j]['prob_to'] = int32_mv[cursor: cursor + oasis_float_to_int32_size].view(oasis_float)[0]
            cursor += oasis_float_to_int32_size

            rec[j]['bin_mean'] = int32_mv[cursor: cursor + oasis_float_to_int32_size].view(oasis_float)[0]
            cursor += oasis_float_to_int32_size

        rec_idx_ptr.append(rec_idx_ptr[-1] + Nbins_to_read)
        last_rec_idx_ptr = end_rec

        # cursor_buf = cursor * 4
        # Nbins[i] = Nbins_to_read

        # take these in input
        # items_data_by_coverage_id = Dict.empty(int_, List.empty_list(ITEMS_DATA_MAP_TYPE))
        # list_coverage_ids = List.empty_list(COVERAGE_ID_TYPE)

        # rng_index = 0
        # seeds = []
        # group_id_rng_index = {}
        # Ncoverage_ids = 0

        # for damagecdf_i, damagecdf in enumerate(damagecdfs):
        item_key = tuple((areaperil_id, vulnerability_id))

        for item in item_map[item_key]:
            item_id, coverage_id, group_id = item
            # print(areaperil_id, vulnerability_id, item_id, coverage_id, group_id)
            # check if this group_id was already processed
            # it assumes that hash only depends on event_id and group_id
            # and that only 1 event_id is processed at a time.
            if group_id not in group_id_rng_index:
                group_id_rng_index[group_id] = rng_index
                seeds[rng_index] = generate_hash(group_id, last_event_id)
                this_rng_index = rng_index
                rng_index += 1
            else:
                this_rng_index = group_id_rng_index[group_id]

            # append always, will filter the unqiue list_coverage_ids later
            # for `list_coverage_ids` list is preferable over set because order is important
            coverage = coverages[coverage_id]
            if coverage['cur_items'] == 0:
                # no items were collected for this coverage: set up the structure
                compute[compute_i], compute_i = coverage_id, compute_i + 1
                while items_data.shape[0] < items_data_i + coverage['max_items']:  # MT don't understand this
                    # is this just a hand-wavy approach: double the size whenever you need more?
                    temp_items_data = np.empty(items_data.shape[0] * 2, dtype=items_data.dtype)
                    temp_items_data[:items_data_i] = items_data[:items_data_i]
                    items_data = temp_items_data

                coverage['start_items'], items_data_i = items_data_i, items_data_i + coverage['max_items']


            coverage['cur_items'] += 1
            insert_item_data_val(items_data[coverage['start_items']: coverage['start_items'] + coverage['cur_items']], item_id, damagecdf_i, this_rng_index)
            # print(items_data[coverage['start_items']: coverage['start_items'] + coverage['cur_items']], coverage['start_items'], coverage['cur_items'])

            # append_to_dict_value(item_ids_by_coverage_id, coverage_id, item_id, nb.types.int32)

        damagecdf_i += 1

    return cursor, yield_event, event_id, rec, rec_idx_ptr, last_event_id, compute_i, items_data_i, items_data, rng_index, group_id_rng_index, damagecdf_i


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
def write_sample_header(event_id, item_id, int32_mv, cursor):
    """Write to buffer the header for the samples of this (event, item).

    Args:
        event_id (int32): event id.
        item_id (int32): item id.
        int32_mv (numpy.ndarray): int32 view of the memoryview where the output is buffered.
        cursor (int): index of int32_mv where to start writing.

    Returns:
        int: updated values of cursor
    """
    int32_mv[cursor], cursor = event_id, cursor + 1
    int32_mv[cursor], cursor = item_id, cursor + 1

    return cursor


@njit(cache=True, fastmath=True)
def write_sample_rec(sidx, loss, int32_mv, cursor):
    """Write to buffer a (sidx, loss) sample record.

    Args:
        sidx (int32): sidx number.
        loss (oasis_float): loss.
        int32_mv (numpy.ndarray): int32 view of the memoryview where the output is buffered.
        cursor (int): index of int32_mv where to start writing.

    Returns:
        int: updated values of cursor
    """
    int32_mv[cursor], cursor = sidx, cursor + 1
    int32_mv[cursor:cursor + oasis_float_to_int32_size].view(oasis_float)[:] = loss
    cursor += oasis_float_to_int32_size

    return cursor


@njit(cache=True, fastmath=True)
def write_negative_sidx(max_loss_idx, max_loss, chance_of_loss_idx, chance_of_loss,
                        tiv_idx, tiv, std_dev_idx, std_dev, mean_idx, gul_mean,
                        int32_mv, cursor):
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

    Returns:
        int: updated values of cursor
    """
    int32_mv[cursor], cursor = max_loss_idx, cursor + 1
    int32_mv[cursor:cursor + oasis_float_to_int32_size].view(oasis_float)[:] = max_loss
    cursor += oasis_float_to_int32_size

    int32_mv[cursor], cursor = chance_of_loss_idx, cursor + 1
    int32_mv[cursor:cursor + oasis_float_to_int32_size].view(oasis_float)[:] = chance_of_loss
    cursor += oasis_float_to_int32_size

    int32_mv[cursor], cursor = tiv_idx, cursor + 1
    int32_mv[cursor:cursor + oasis_float_to_int32_size].view(oasis_float)[:] = tiv
    cursor += oasis_float_to_int32_size

    int32_mv[cursor], cursor = std_dev_idx, cursor + 1
    int32_mv[cursor:cursor + oasis_float_to_int32_size].view(oasis_float)[:] = std_dev
    cursor += oasis_float_to_int32_size

    int32_mv[cursor], cursor = mean_idx, cursor + 1
    int32_mv[cursor:cursor + oasis_float_to_int32_size].view(oasis_float)[:] = gul_mean
    cursor += oasis_float_to_int32_size

    return cursor
