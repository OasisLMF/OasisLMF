import ray
import os
import asyncio
import numpy as np
import numba as nb
import numba.typed
from numba.types import int32 as nb_int32, int64 as nb_int64

from .common import areaperil_int, MAX_LOSS_IDX, CHANCE_OF_LOSS_IDX, TIV_IDX, STD_DEV_IDX, MEAN_IDX
from .manager import (logger, get_damage_bins, oasis_float, get_coverages, coverage_type, gul_get_items,
                      generate_item_map, get_random_generator, Item, NUM_IDX)
from .io import generate_hash, NP_BASE_ARRAY_SIZE, items_data_type
from .core import compute_mean_loss, get_gul, split_tiv, setmaxloss
from .utils import binary_search

ProbMean = nb.from_dtype(np.dtype([('prob_to', oasis_float),
                                   ('bin_mean', oasis_float)
                                   ]))

sidx_loss_type = nb.from_dtype(np.dtype([('sidx', np.int32),
                                   ('loss', oasis_float)
                                   ]))

cdf_info_type = nb.from_dtype(np.dtype([('area_peril_id', areaperil_int),
                                       ('vulnerability_id', np.int32),
                                       ('bin_start', np.int64)
                                       ]))


items_out_type = nb.from_dtype(np.dtype([('item_id', np.int32),
                                          ('loss_start', np.int64),
                                          ]))


@nb.jit(cache=True, fastmath=True)
def ndarray_prepare_add(_array, cur_size, elm_to_fit):
    while cur_size + elm_to_fit > _array.shape[0]:
        tmp = np.empty(shape=2*_array.shape[0], dtype = _array.dtype)
        tmp[:cur_size] = _array[:cur_size]
        _array = tmp
    return _array


@nb.njit(cache=True)
def gen_structs():
    """Generate some data structures needed for the whole computation.

    Returns:
        Dict(int,int), List: map of group ids to random seeds,
          list storing the index where a specific cdf record starts in the `rec` numpy array.

    """
    group_id_rng_index = numba.typed.Dict.empty(nb_int32, nb_int64)

    return group_id_rng_index


@nb.njit(cache=True)
def prepare(event_id, cdf_info, recs_size, item_map, items_data, compute, seeds, group_id_rng_index, coverages):
    rng_index = 0
    damagecdf_i = 0
    items_data_i = 0
    compute_i = 0
    rec_idx_ptr = np.empty(shape=cdf_info.shape[0] + 1, dtype=np.int64)

    for i in range(cdf_info.shape[0]):
        # register the items to their coverage
        areaperil_id = cdf_info[i]['area_peril_id']
        vulnerability_id = cdf_info[i]['vulnerability_id']
        rec_idx_ptr[damagecdf_i] = cdf_info[i]['bin_start']

        item_key = tuple((areaperil_id, vulnerability_id))

        for item in item_map[item_key]:
            item_id, coverage_id, group_id = item

            # if this group_id was not seen yet, process it.
            # it assumes that hash only depends on event_id and group_id
            # and that only 1 event_id is processed at a time.
            if group_id not in group_id_rng_index:
                group_id_rng_index[group_id] = rng_index
                seeds[rng_index] = generate_hash(group_id, event_id)
                this_rng_index = rng_index
                rng_index += 1
            else:
                this_rng_index = group_id_rng_index[group_id]

            coverage = coverages[coverage_id]
            if coverage['cur_items'] == 0:
                # no items were collected for this coverage yet: set up the structure
                compute[compute_i], compute_i = coverage_id, compute_i + 1

                while items_data.shape[0] < items_data_i + coverage['max_items']:
                    # if items_data needs to be larger to store all the items, double it in size
                    temp_items_data = np.empty(items_data.shape[0] * 2, dtype=items_data.dtype)
                    temp_items_data[:items_data_i] = items_data[:items_data_i]
                    items_data = temp_items_data

                coverage['start_items'], items_data_i = items_data_i, items_data_i + coverage['max_items']

            # append the data of this item
            item_i = coverage['start_items'] + coverage['cur_items']
            items_data[item_i]['item_id'] = item_id
            items_data[item_i]['damagecdf_i'] = damagecdf_i
            items_data[item_i]['rng_index'] = this_rng_index

            coverage['cur_items'] += 1

        damagecdf_i += 1

    rec_idx_ptr[damagecdf_i] = recs_size
    return items_data, rec_idx_ptr, compute_i, rng_index


@nb.njit(cache=True, fastmath=True)
def write_losses(sample_size, loss_threshold, losses, item_ids, alloc_rule, tiv,
                 items_out, total_item_i, losses_out, loss_i):
    """Write the computed losses.

    Args:
        event_id (int32): event id.
        sample_size (int): number of random samples to draw.
        loss_threshold (float): threshold above which losses are printed to the output stream.
        losses (numpy.array[oasis_float]): losses for all item_ids
        item_ids (numpy.array[ITEM_ID_TYPE]): ids of items whose losses are in `losses`.
        alloc_rule (int): back-allocation rule.
        tiv (oasis_float): total insured value.
        int32_mv (numpy.ndarray): int32 view of the memoryview where the output is buffered.
        cursor (int): index of int32_mv where to start writing.

    Returns:
        int: updated values of cursor
    """
    # note that Nsamples = sample_size + NUM_IDX

    if alloc_rule == 2:
        losses[1:] = setmaxloss(losses[1:])

    # split tiv has to be executed after setmaxloss, if alloc_rule==2.
    if alloc_rule >= 1 and tiv > 0:
        # check whether the sum of losses-per-sample exceeds TIV
        # if so, split TIV in proportion to the losses
        for sample_i in range(1, losses.shape[0]):
            split_tiv(losses[sample_i], tiv)

    # output the losses for all the items
    for item_j in range(item_ids.shape[0]):

        # write header
        items_out[total_item_i]['item_id'] = item_ids[item_j]
        items_out[total_item_i]['loss_start'] = loss_i
        total_item_i += 1

        # write negative sidx
        for sidx in [MAX_LOSS_IDX, CHANCE_OF_LOSS_IDX, TIV_IDX, STD_DEV_IDX, MEAN_IDX]:
            losses_out[loss_i]['sidx'] = sidx
            losses_out[loss_i]['loss'] = losses[sidx, item_j]
            loss_i += 1

        # write the random samples (only those with losses above the threshold)
        for sample_idx in range(1, sample_size + 1):
            if losses[sample_idx, item_j] >= loss_threshold:
                losses_out[loss_i]['sidx'] = sidx
                losses_out[loss_i]['loss'] = losses[sidx, item_j]
                loss_i += 1

    return total_item_i, loss_i




@nb.njit(cache=True, fastmath=True)
def compute_event_losses(coverages, coverage_ids, items_data,
                         sample_size, recs, rec_idx_ptr, damage_bins,
                         loss_threshold, losses, alloc_rule, rndms, debug, items_out, losses_out):
    """Compute losses for an event.

    Args:
        event_id (int32): event id.
        coverages (numpy.array[oasis_float]): array with the coverage values for each coverage_id.
        coverage_ids (numpy.array[: array of **uniques** coverage ids used in this event.
        items_data (numpy.array[items_data_type]): items-related data.
        last_processed_coverage_ids_idx (int): index of the last coverage_id stored in `coverage_ids` that was fully processed
          and printed to the output stream.
        sample_size (int): number of random samples to draw.
        recs (numpy.array[ProbMean]): all the cdfs used in event_id.
        rec_idx_ptr (numpy.array[int]): array with the indices of `rec` where each cdf record starts.
        damage_bins (List[Union[damagebindictionaryCsv, damagebindictionary]]): loaded data from the damage_bin_dict file.
        loss_threshold (float): threshold above which losses are printed to the output stream.
        losses (numpy.array[oasis_float]): array (to be re-used) to store losses for all item_ids.
        alloc_rule (int): back-allocation rule.
        rndms (numpy.array[float64]): 2d array of shape (number of seeds, sample_size) storing the random values
          drawn for each seed.
        debug (bool): if True, for each random sample, print to the output stream the random value
          instead of the loss.
        buff_size (int): size in bytes of the output buffer.
        int32_mv (numpy.ndarray): int32 view of the memoryview where the output is buffered.
        cursor (int): index of int32_mv where to start writing.

    Returns:
        int, int, int: updated value of cursor, updated value of cursor_bytes, last last_processed_coverage_ids_idx
    """
    total_item_i = 0
    loss_i = 0
    for coverage_i in range(coverage_ids.shape[0]):
        coverage = coverages[coverage_ids[coverage_i]]
        tiv = coverage['tiv']  # coverages are indexed from 1
        Nitem_ids = coverage['cur_items']
        exposureValue = tiv / Nitem_ids

        items = items_data[coverage['start_items']: coverage['start_items'] + coverage['cur_items']]

        items_out = ndarray_prepare_add(items_out, total_item_i, coverage['cur_items'])
        losses_out = ndarray_prepare_add(losses_out, loss_i, coverage['cur_items'] * (sample_size + NUM_IDX))

        for item_i in range(coverage['cur_items']):
            item = items[item_i]
            damagecdf_i = item['damagecdf_i']
            rng_index = item['rng_index']

            rec = recs[rec_idx_ptr[damagecdf_i]:rec_idx_ptr[damagecdf_i + 1]]
            prob_to = rec['prob_to']
            bin_mean = rec['bin_mean']
            Nbins = len(prob_to)

            # compute mean values
            gul_mean, std_dev, chance_of_loss, max_loss = compute_mean_loss(
                tiv, prob_to, bin_mean, Nbins, damage_bins[Nbins - 1]['bin_to'],
            )

            losses[MAX_LOSS_IDX, item_i] = max_loss
            losses[CHANCE_OF_LOSS_IDX, item_i] = chance_of_loss
            losses[TIV_IDX, item_i] = exposureValue
            losses[STD_DEV_IDX, item_i] = std_dev
            losses[MEAN_IDX, item_i] = gul_mean

            if sample_size > 0:
                if debug:
                    for sample_idx in range(1, sample_size + 1):
                        rval = rndms[rng_index][sample_idx - 1]
                        losses[sample_idx, item_i] = rval
                else:
                    for sample_idx in range(1, sample_size + 1):
                        # cap `rval` to the maximum `prob_to` value (which should be 1.)
                        rval = rndms[rng_index][sample_idx - 1]

                        if rval >= prob_to[Nbins - 1]:
                            rval = prob_to[Nbins - 1] - 0.00000003
                            bin_idx = Nbins - 1
                        else:
                            # find the bin in which the random value `rval` falls into
                            # note that rec['bin_mean'] == damage_bins['interpolation'], therefore
                            # there's a 1:1 mapping between indices of rec and damage_bins
                            bin_idx = binary_search(rval, prob_to, Nbins)

                        # compute ground-up losses
                        gul = get_gul(
                            damage_bins['bin_from'][bin_idx],
                            damage_bins['bin_to'][bin_idx],
                            bin_mean[bin_idx],
                            prob_to[bin_idx - 1] * (bin_idx > 0),
                            prob_to[bin_idx],
                            rval,
                            tiv
                        )

                        if gul >= loss_threshold:
                            losses[sample_idx, item_i] = gul
                        else:
                            losses[sample_idx, item_i] = 0

        total_item_i, loss_i = write_losses(sample_size, loss_threshold, losses[:, :items.shape[0]], items['item_id'],
                                            alloc_rule, tiv, items_out, total_item_i, losses_out, loss_i)

    return items_out, total_item_i, losses_out, loss_i


@ray.remote
class GulPy:
    def __init__(self, cdf_queue, gul_queue, run_dir, ignore_file_type, sample_size, loss_threshold, alloc_rule, debug,
        random_generator):
        """Execute the main gulpy worklow.

        Args:
            run_dir: (str) the directory of where the process is running
            ignore_file_type set(str): file extension to ignore when loading
            sample_size (int): number of random samples to draw.
            loss_threshold (float): threshold above which losses are printed to the output stream.
            alloc_rule (int): back-allocation rule.
            debug (bool): if True, for each random sample, print to the output stream the random value
              instead of the loss.
            random_generator (int): random generator function id.
            file_in (str, optional): filename of input stream. Defaults to None.
            file_out (str, optional): filename of output stream. Defaults to None.

        Raises:
            ValueError: if alloc_rule is not 0, 1, or 2.

        Returns:
            int: 0 if no errors occurred.
        """
        logger.info("starting gulpy")

        self.cdf_queue = cdf_queue
        self.gul_queue = gul_queue

        static_path = os.path.join(run_dir, 'static')
        input_path = os.path.join(run_dir, 'input')
        ignore_file_type = set(ignore_file_type)

        static_path = 'static/'
        # TODO: store static_path in a paraparameters file
        self.damage_bins = get_damage_bins(static_path)

        input_path = 'input/'
        # TODO: store input_path in a paraparameters file

        # read coverages from file
        coverages_tiv = get_coverages(input_path)

        # init the structure for computation
        # coverages are numbered from 1, therefore we skip element 0 in `coverages`
        self.coverages = np.empty(coverages_tiv.shape[0] + 1, coverage_type)
        self.coverages[1:]['tiv'] = coverages_tiv
        del coverages_tiv

        items = gul_get_items(input_path)

        # in-place sort items in order to store them in item_map in the desired order
        # currently numba only supports a simple call to np.sort() with no `order` keyword,
        # so we do the sort here.
        items = np.sort(items, order=['areaperil_id', 'vulnerability_id'])
        self.item_map = generate_item_map(items, self.coverages)

        # init array to store the coverages to be computed
        # coverages are numebered from 1, therefore skip element 0.
        self.compute = np.zeros(self.coverages.shape[0] + 1, items.dtype['coverage_id'])

        self.sample_size = sample_size

        # set the random generator function
        self.generate_rndm = get_random_generator(random_generator)

        if alloc_rule not in [0, 1, 2]:
            raise ValueError(f"Expect alloc_rule to be 0 or 1 or 2, got {alloc_rule}")

        self.alloc_rule = alloc_rule
        self.loss_threshold = loss_threshold
        self.debug = debug

        # create the array to store the seeds
        self.seeds = np.zeros(len(np.unique(items['group_id'])), dtype=Item.dtype['group_id'])

        # create buffer to be reused to store all losses for one coverage
        self.losses_buffer = np.zeros((sample_size + NUM_IDX + 1, np.max(self.coverages['max_items'])), dtype=oasis_float)

    def run(self):
        try:
            group_id_rng_index = gen_structs()
            items_data = np.empty(2 ** NP_BASE_ARRAY_SIZE, dtype=items_data_type)
            items_out = np.empty(2 ** NP_BASE_ARRAY_SIZE, dtype=items_out_type)
            losses_out = np.empty(2 ** NP_BASE_ARRAY_SIZE, dtype=sidx_loss_type)
            while True:
                try:
                    cdf = self.cdf_queue.get(timeout=30)
                except ray.util.queue.Empty:
                    print('timed_out')
                else:
                    if cdf is None:
                        self.cdf_queue.put(None)
                        break
                    else:
                        event_id, cdf_info, recs = cdf
                        items_data, rec_idx_ptr, compute_i, rng_index = prepare(event_id, cdf_info, recs.shape[0], self.item_map,
                                                                                items_data, self.compute, self.seeds, group_id_rng_index, self.coverages)

                        rndms = self.generate_rndm(self.seeds[:rng_index], self.sample_size)

                        items_out, total_item_i, losses_out, loss_i = compute_event_losses(
                            self.coverages, self.compute[:compute_i], items_data,
                            self.sample_size, recs, rec_idx_ptr,
                            self.damage_bins, self.loss_threshold, self.losses_buffer, self.alloc_rule, rndms, self.debug,
                            items_out, losses_out
                        )
                        self.gul_queue.put([event_id, items_out[:total_item_i], losses_out[:loss_i]])


            print('finished')
        except Exception as e:
            print('Exception', e)
            raise