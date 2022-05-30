import ray
import os
import numpy as np
import numba as nb

from .manager import (logger, FootprintLayerClient, atexit, ExitStack, get_vulns, convert_vuln_id_to_index, Footprint,
                             get_mean_damage_bins, areaperil_int_relative_size, results_relative_size, buff_size, get_items,
                      oasis_float, areaperil_int, damage_bin_prob, oasis_int)

ProbMean = nb.from_dtype(np.dtype([('prob_to', oasis_float),
                                   ('bin_mean', oasis_float)
                                   ]))

cdf_info_type = nb.from_dtype(np.dtype([('area_peril_id', areaperil_int),
                                        ('vulnerability_id', oasis_int),
                                        ('bin_start', np.int64)
                                       ]))

@nb.jit(cache=True, fastmath=True)
def ndarray_prepare_add(_array, cur_size, elm_to_fit):
    while cur_size + elm_to_fit > _array.shape[0]:
        tmp = np.empty(shape=2*_array.shape[0], dtype = _array.dtype)
        tmp[:cur_size] = _array[:cur_size]
        _array = tmp
    return _array


@nb.jit(cache=True, fastmath=True)
def do_result(vulns_id, vuln_array, mean_damage_bins,
              num_damage_bins,
              intensities_min, intensities_max, intensities,
              areaperil_id, vuln_i,
              cdf_info, cdf_info_i,
              bins, bin_i):
    """
    Calculate the result concerning an event ID.

    Args:
        vulns_id: (List[int]) list of vulnerability IDs
        vuln_array: (List[List[list]]) list of vulnerabilities and their data
        mean_damage_bins: (List[float]) the mean of each damage bin (len(mean_damage_bins) == num_damage_bins)
        int32_mv: (List[int]) FILL IN LATER
        num_damage_bins: (int) number of damage bins in the data
        intensities_min: (int) intensity minimum
        intensities_max: (int) intensity maximum
        intensities: (List[float]) list of all the intensities
        event_id: (int) the event ID that concerns the result being calculated
        areaperil_id: (List[int]) the areaperil ID that concerns the result being calculated
        vuln_i: (int) the index concerning the vulnerability inside the vuln_array
        cursor: (int) PLEASE FILL IN

    Returns: (int) PLEASE FILL IN
    """
    cdf_info[cdf_info_i]['area_peril_id'] = areaperil_id[0]
    cdf_info[cdf_info_i]['vulnerability_id'] = vulns_id[vuln_i]
    cdf_info[cdf_info_i]['bin_start'] = bin_i

    cur_vuln_mat = vuln_array[vuln_i]
    p = 0
    damage_bin_i = 0

    while damage_bin_i < num_damage_bins:
        p = damage_bin_prob(p, intensities_min, intensities_max, cur_vuln_mat[damage_bin_i], intensities)
        bins[bin_i]['prob_to'] = p
        bins[bin_i]['bin_mean'] = mean_damage_bins[damage_bin_i]
        bin_i += 1
        damage_bin_i += 1
        if p > 0.999999940:
            break

    return bin_i



@nb.njit(cache=True)
def doCdf(num_intensity_bins, footprint,
          areaperil_to_vulns_idx_dict, areaperil_to_vulns_idx_array, areaperil_to_vulns,
          vuln_array, vulns_id, num_damage_bins, mean_damage_bins,
          cdf_info, bins
          ):
    """
    Calculates the cumulative distribution function (cdf) for an event ID.

    Args:
        event_id: (int) the event ID the the CDF is being calculated to.
        num_intensity_bins: (int) the number of intensity bins for the CDF
        footprint: (List[Tuple[int, int, float]]) information about the footprint with event_id, areaperil_id,
                                                  probability
        areaperil_to_vulns_idx_dict: (Dict[int, int]) maps the areaperil ID with the ENTER_HERE
        areaperil_to_vulns_idx_array: (List[Tuple[int, int]]) the index where the areaperil ID starts and finishes
        areaperil_to_vulns: (List[int]) maps the areaperil ID to the vulnerability ID
        vuln_array: (List[list]) FILL IN LATER
        vulns_id: (List[int]) list of vulnerability IDs
        num_damage_bins: (int) number of damage bins in the data
        mean_damage_bins: (List[float]) the mean of each damage bin (len(mean_damage_bins) == num_damage_bins)

    Returns: (int)
    """

    intensities_min = num_intensity_bins
    intensities_max = 0
    intensities = np.zeros(num_intensity_bins, dtype=oasis_float)

    areaperil_id = np.zeros(1, dtype=areaperil_int)
    has_vuln = False

    cdf_info_i = 0
    bin_i = 0

    for footprint_i in range(footprint.shape[0]):
        event_row = footprint[footprint_i]
        if areaperil_id[0] != event_row['areaperil_id']:
            if has_vuln and intensities_min <= intensities_max:
                areaperil_to_vulns_idx = areaperil_to_vulns_idx_array[areaperil_to_vulns_idx_dict[areaperil_id[0]]]
                intensities_max += 1
                cdf_info = ndarray_prepare_add(cdf_info, cdf_info_i, areaperil_to_vulns_idx['end'] - areaperil_to_vulns_idx['start'])
                bins = ndarray_prepare_add(bins, bin_i, num_damage_bins * (areaperil_to_vulns_idx['end'] - areaperil_to_vulns_idx['start']))

                for vuln_idx in range(areaperil_to_vulns_idx['start'], areaperil_to_vulns_idx['end']):
                    vuln_i = areaperil_to_vulns[vuln_idx]

                    bin_i = do_result(vulns_id, vuln_array, mean_damage_bins,
                                      num_damage_bins,
                                      intensities_min, intensities_max, intensities,
                                      areaperil_id, vuln_i,
                                      cdf_info, cdf_info_i,
                                      bins, bin_i)
                    cdf_info_i += 1

            areaperil_id[0] = event_row['areaperil_id']
            has_vuln = areaperil_id[0] in areaperil_to_vulns_idx_dict

            if has_vuln:
                intensities[intensities_min: intensities_max] = 0
                intensities_min = num_intensity_bins
                intensities_max = 0
        if has_vuln:
            if event_row['probability']>0:
                intensity_bin_i = event_row['intensity_bin_id'] - 1
                intensities[intensity_bin_i] = event_row['probability']
                if intensity_bin_i > intensities_max:
                    intensities_max = intensity_bin_i
                if intensity_bin_i < intensities_min:
                    intensities_min = intensity_bin_i

    if has_vuln and intensities_min <= intensities_max:
        areaperil_to_vulns_idx = areaperil_to_vulns_idx_array[areaperil_to_vulns_idx_dict[areaperil_id[0]]]
        intensities_max += 1
        for vuln_idx in range(areaperil_to_vulns_idx['start'], areaperil_to_vulns_idx['end']):
            vuln_i = areaperil_to_vulns[vuln_idx]
            cdf_info = ndarray_prepare_add(cdf_info, cdf_info_i, areaperil_to_vulns_idx['end'] - areaperil_to_vulns_idx['start'])
            bins = ndarray_prepare_add(bins, bin_i, num_damage_bins * (areaperil_to_vulns_idx['end'] - areaperil_to_vulns_idx['start']))

            bin_i = do_result(vulns_id, vuln_array, mean_damage_bins,
                                      num_damage_bins,
                                      intensities_min, intensities_max, intensities,
                                      areaperil_id, vuln_i,
                                      cdf_info, cdf_info_i,
                                      bins, bin_i)
            cdf_info_i += 1

    return cdf_info, cdf_info_i, bins, bin_i

@ray.remote
class ModelPy:
    def __init__(self, run_dir, eve_queue, cdf_queue, ignore_file_type, data_server):
        """
        Args:
            run_dir: (str) the directory of where the process is running
            file_in: (Optional[str]) the path to the input directory
            file_out: (Optional[str]) the path to the output directory
            ignore_file_type: set(str) file extension to ignore when loading
            data_server: (bool) if set to True runs the data server
        """
        self.eve_queue = eve_queue
        self.cdf_queue = cdf_queue
        static_path = os.path.join(run_dir, 'static')
        input_path = os.path.join(run_dir, 'input')
        ignore_file_type = set(ignore_file_type)
        self.data_server = data_server
        if self.data_server:
            print("data server active")
            FootprintLayerClient.register()
            print("registered with data server")
            atexit.register(FootprintLayerClient.unregister)
        else:
            print("data server not active")

        print('init items')
        vuln_dict, self.areaperil_to_vulns_idx_dict, self.areaperil_to_vulns_idx_array, self.areaperil_to_vulns = get_items(
            input_path, ignore_file_type)

        print('init footprint')
        self.stack = ExitStack()
        self.footprint_obj = self.stack.enter_context(Footprint.load(static_path, ignore_file_type))

        if self.data_server:
            self.num_intensity_bins: int = FootprintLayerClient.get_number_of_intensity_bins()
            logger.info(f"got {self.num_intensity_bins} intensity bins from server")
        else:
            self.num_intensity_bins: int = self.footprint_obj.num_intensity_bins

        print('init vulnerability')

        self.vuln_array, self.vulns_id, self.num_damage_bins = get_vulns(static_path, vuln_dict, self.num_intensity_bins,
                                                          ignore_file_type)
        convert_vuln_id_to_index(vuln_dict, self.areaperil_to_vulns)
        print('init mean_damage_bins')
        self.mean_damage_bins = get_mean_damage_bins(static_path, ignore_file_type)

    def run(self):
        cdf_info = np.empty(shape=64, dtype=cdf_info_type)
        bins = np.empty(shape=256, dtype=ProbMean)
        try:
            while True:
                try:
                    event_ids = self.eve_queue.get(timeout=30)
                except ray.util.queue.Empty:
                    print('timed_out')
                else:
                    if event_ids is None:
                        self.eve_queue.put(None)
                        break
                    else:
                        for i in range(len(event_ids)):
                            if self.data_server:
                                event_footprint = FootprintLayerClient.get_event(event_ids[i])
                            else:
                                event_footprint = self.footprint_obj.get_event(event_ids[i])
                            if event_footprint is not None and event_footprint.shape[0]:
                                cdf_info, cdf_info_i, bins, bin_i = doCdf(
                                              self.num_intensity_bins, event_footprint,
                                              self.areaperil_to_vulns_idx_dict, self.areaperil_to_vulns_idx_array, self.areaperil_to_vulns,
                                              self.vuln_array, self.vulns_id, self.num_damage_bins, self.mean_damage_bins,
                                              cdf_info, bins
                                              )
                                if cdf_info_i:
                                    self.cdf_queue.put([event_ids[i], cdf_info[:cdf_info_i], bins[:bin_i]])

        except Exception as e:
            print('Exception', e)
            raise
        self.stack.close()
        return 'finished'