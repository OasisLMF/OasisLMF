"""
This file is the entry point for the python get model command for the package

TODO: use selector and select for output

"""
import argparse
import logging
import numba as nb
import numpy as np
import pyarrow.parquet as pq
import os
import sys
from contextlib import ExitStack
from numba.typed import Dict
import pyarrow as pa

from .common import areaperil_int, oasis_float, Index_type
from .footprint import Footprint


logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

buff_size = 65536

oasis_int_dtype = np.dtype('i4')
oasis_int = np.int32
oasis_int_size = np.int32().itemsize
buff_int_size = buff_size // oasis_int_size

areaperil_int_relative_size = areaperil_int.itemsize // oasis_int_size
oasis_float_relative_size = oasis_float.itemsize // oasis_int_size
results_relative_size = 2 * oasis_float_relative_size


damagebindictionary =  nb.from_dtype(np.dtype([('bin_index', np.int32),
                                    ('bin_from', oasis_float),
                                    ('bin_to', oasis_float),
                                    ('interpolation', oasis_float),
                                    ('interval_type', np.int32),
                                  ]))

damagebindictionaryCsv =  nb.from_dtype(np.dtype([('bin_index', np.int32),
                                                  ('bin_from', oasis_float),
                                                  ('bin_to', oasis_float),
                                                  ('interpolation', oasis_float)]))

EventCSV =  nb.from_dtype(np.dtype([('event_id', np.int32),
                                    ('areaperil_id', areaperil_int),
                                    ('intensity_bin_id', np.int32),
                                    ('probability', oasis_float)
                                    ]))

Item = nb.from_dtype(np.dtype([('id', np.int32),
                               ('coverage_id', np.int32),
                               ('areaperil_id', areaperil_int),
                               ('vulnerability_id', np.int32),
                               ('group_id', np.int32)
                               ]))


Vulnerability = nb.from_dtype(np.dtype([('vulnerability_id', np.int32),
                                        ('intensity_bin_id', np.int32),
                                        ('damage_bin_id', np.int32),
                                        ('probability', oasis_float)
                                        ]))

VulnerabilityIndex = nb.from_dtype(np.dtype([('vulnerability_id', np.int32),
                                             ('offset', np.int64),
                                             ('size', np.int64),
                                             ('original_size', np.int64)
                                             ]))
VulnerabilityRow = nb.from_dtype(np.dtype([('intensity_bin_id', np.int32),
                                           ('damage_bin_id', np.int64),
                                           ('probability', oasis_float)
                                          ]))

vuln_offset = 4


@nb.jit(cache=True)
def load_areaperil_id_u4(int32_mv, cursor, areaperil_id):
    int32_mv[cursor] = areaperil_id.view('i4')
    return cursor + 1


@nb.jit(cache=True)
def load_areaperil_id_u8(int32_mv, cursor, areaperil_id):
    int32_mv[cursor: cursor+1] = areaperil_id.view('i4')
    return cursor + 2


if areaperil_int == 'u4':
    load_areaperil_id = load_areaperil_id_u4
elif areaperil_int == 'u8':
    load_areaperil_id = load_areaperil_id_u8
else:
    raise Exception(f"AREAPERIL_TYPE {areaperil_int} is not implemented chose u4 or u8")


@nb.jit(cache=True)
def load_items(items):
    """
    Processes the Items loaded from the file extracting meta data around the vulnerability data.

    Args:
        items: (List[Item]) Data loaded from the vulnerability file

    Returns: (Tuple[Dict[int, int], List[int], Dict[int, int], List[Tuple[int, int]], List[int]])
             vulnerability dictionary, vulnerability IDs, areaperil to vulnerability index dictionary,
             areaperil ID to vulnerability index array, areaperil ID to vulnerability array
    """
    areaperil_to_vulns_size = 0
    areaperil_dict = Dict()
    vuln_dict = Dict()
    vuln_idx = 0
    for i in range(items.shape[0]):
        item = items[i]

        # insert the vulnerability index if not in there
        if item['vulnerability_id'] not in vuln_dict:
            vuln_dict[item['vulnerability_id']] = vuln_idx
            vuln_idx += 1

        # insert an area dictionary into areaperil_dict under the key of areaperil ID
        if item['areaperil_id'] not in areaperil_dict:
            area_vuln = Dict()
            area_vuln[item['vulnerability_id']] = 0
            areaperil_dict[item['areaperil_id']] = area_vuln
            areaperil_to_vulns_size += 1
        else:
            if item['vulnerability_id'] not in areaperil_dict[item['areaperil_id']]:
                areaperil_to_vulns_size += 1
                areaperil_dict[item['areaperil_id']][item['vulnerability_id']] = 0

    areaperil_to_vulns_idx_dict = Dict()
    areaperil_to_vulns_idx_array = np.empty(len(areaperil_dict), dtype = Index_type)
    areaperil_to_vulns = np.empty(areaperil_to_vulns_size, dtype = np.int32)
    vulns_id = np.empty(len(vuln_dict), dtype = np.int32)

    for vuln_id, vuln_idx in vuln_dict.items():
        vulns_id[vuln_idx] = vuln_id

    areaperil_i = 0
    vulnerability_i = 0

    for areaperil_id, vulns in areaperil_dict.items():
        areaperil_to_vulns_idx_dict[areaperil_id] = areaperil_i
        areaperil_to_vulns_idx_array[areaperil_i]['start'] = vulnerability_i

        for vuln_id in sorted(vulns):  # sorted is not necessary but doesn't impede the perf and align with cpp getmodel
            areaperil_to_vulns[vulnerability_i] = vuln_dict[vuln_id]
            vulnerability_i +=1
        areaperil_to_vulns_idx_array[areaperil_i]['end'] = vulnerability_i
        areaperil_i+=1

    return vuln_dict, vulns_id, areaperil_to_vulns_idx_dict, areaperil_to_vulns_idx_array, areaperil_to_vulns


def get_items(input_path, file_type):
    """
    Loads the items from the items file.

    Args:
        input_path: (str) the path pointing to the file
        file_type: (str) the type of file being loaded

    Returns: (Tuple[Dict[int, int], List[int], Dict[int, int], List[Tuple[int, int]], List[int]])
             vulnerability dictionary, vulnerability IDs, areaperil to vulnerability index dictionary,
             areaperil ID to vulnerability index array, areaperil ID to vulnerability array
    """
    input_files = set(os.listdir(input_path))
    if "items.bin" in input_files and file_type == "bin":
        items = np.memmap(os.path.join(input_path, "items.bin"), dtype=Item, mode='r')
    elif "items.csv" in input_files and file_type == "csv":
        items = np.genfromtxt(os.path.join(input_path, "items.csv"), dtype=Item, delimiter=",")

    return load_items(items)


@nb.njit(cache=True)
def load_vulns_bin_idx(vulns_bin, vulns_idx_bin, vuln_dict,
                                 num_damage_bins, num_intensity_bins):
    """
    Not firing when testing so not able to inspect yet

    Args:
        vulns_bin:
        vulns_idx_bin:
        vuln_dict:
        num_damage_bins:
        num_intensity_bins:

    Returns:
    """
    vuln_array = np.zeros((len(vuln_dict), num_damage_bins, num_intensity_bins), dtype=oasis_float)
    for idx_i in range(vulns_idx_bin.shape[0]):
        vuln_idx = vulns_idx_bin[idx_i]
        if vuln_idx['vulnerability_id'] in vuln_dict:
            cur_vuln_array = vuln_array[vuln_dict[vuln_idx['vulnerability_id']]]
            start = (vuln_idx['offset'] - vuln_offset) // VulnerabilityRow.itemsize
            end = start + vuln_idx['size'] // VulnerabilityRow.itemsize
            for vuln_i in range(start, end):
                vuln = vulns_bin[vuln_i]
                cur_vuln_array[vuln['damage_bin_id'] -1, vuln['intensity_bin_id'] - 1] = vuln['probability']

    return vuln_array


@nb.njit(cache=True)
def load_vulns_bin(vulns_bin, vuln_dict, num_damage_bins, num_intensity_bins):
    """
    Loads the vulnerability data grouped by the intensity and damage bins.

    Args:
        vulns_bin: (List[Vulnerability]) vulnerability data from the vulnerability file
        vuln_dict: (Dict[int, int]) maps the vulnerability ID with the index in the vulnerability array
        num_damage_bins: (int) number of damage bins in the data
        num_intensity_bins: (int) the number of intensity bins

    Returns: (List[List[List[floats]]]) vulnerability data grouped by intensity bin and damage bin
    """
    vuln_array = np.zeros((len(vuln_dict), num_damage_bins, num_intensity_bins), dtype=oasis_float)
    cur_vulnerability_id = -1

    for vuln_i in range(vulns_bin.shape[0]):
        vuln = vulns_bin[vuln_i]
        if vuln['vulnerability_id'] != cur_vulnerability_id:
            if vuln['vulnerability_id'] in vuln_dict:
                cur_vulnerability_id = vuln['vulnerability_id']
                cur_vuln_array = vuln_array[vuln_dict[cur_vulnerability_id]]
            else:
                cur_vulnerability_id = -1
        if cur_vulnerability_id != -1:
            cur_vuln_array[vuln['damage_bin_id'] - 1, vuln['intensity_bin_id'] - 1] = vuln['probability']

    return vuln_array


@nb.jit(cache=True, fastmath=True)
def update_vulns_dictionary(vulns_dict, vulns_id_array):
    """
    Updates the indexes of the vulnerability IDs (usually used in loading vulnerability data from parquet file).

    Args:
        vulns_dict: (Dict[int, int]) vulnerability dict that maps the vulnerability IDs (key) with the index (value)
        vulns_id_array: (List[int]) list of vulnerability IDs loaded from the parquet file

    Returns: (Dict[int, int]) updated vulns_dict
    """
    index = 0
    max_index = len(vulns_id_array)

    while index < max_index:
        vuln_id = vulns_id_array[index]
        vulns_dict[vuln_id] = index
        index += 1
    return vulns_dict


def get_vulns(static_path, vuln_dict, num_intensity_bins, file_type):
    """
    Loads the vulnerabilities from the file.

    Args:
        static_path: (str) the path pointing to the static file where the data is
        vuln_dict: (Dict[int, int]) maps the vulnerability ID with the index in the vulnerability array
        num_intensity_bins: (int) the number of intensity bins 
        file_type: (str) the type of file being loaded

    Returns: (Tuple[List[List[float]], int]) vulnerability data, number of damage bins
    """
    input_files = set(os.listdir(static_path))
    if "vulnerability.parquet" in input_files and file_type == "parquet":
        parquet_handle = pq.ParquetDataset(os.path.join(static_path, "vulnerability.parquet"), use_legacy_dataset=False,
                                           filters=[("vulnerability_id", "in", list(vuln_dict.keys()))])
        vuln_table = parquet_handle.read()
        vuln_meta = vuln_table.schema.metadata
        number_of_vulnerability_ids = int(vuln_meta[b"num_vulnerability_id"].decode("utf-8"))
        num_damage_bins = int(vuln_meta[b"num_damage_bins"].decode("utf-8"))
        number_of_intensity_bins = int(vuln_meta[b"num_intensity_bins"].decode("utf-8"))
        vuln_array = vuln_table[1].to_numpy()
        vuln_array = np.vstack(vuln_array).reshape(number_of_vulnerability_ids,
                                                   num_damage_bins,
                                                   number_of_intensity_bins)
        vuln_dict = update_vulns_dictionary(vuln_dict, vuln_table[0].to_numpy())

    if "vulnerability.bin" in input_files and file_type == "bin":
        with open(os.path.join(static_path, "vulnerability.bin"), 'rb') as f:
            header = np.frombuffer(f.read(8), 'i4')
            num_damage_bins = header[0]
        if "vulnerability.idx" in static_path:
            vulns_bin = np.memmap(os.path.join(static_path, "vulnerability.bin"), dtype=VulnerabilityRow, offset=4, mode='r')
            vulns_idx_bin = np.memmap(os.path.join(static_path, "vulnerability.idx"), dtype=VulnerabilityIndex, mode='r')
            vuln_array = load_vulns_bin_idx(vulns_bin, vulns_idx_bin, vuln_dict,
                                                      num_damage_bins, num_intensity_bins)
        else:
            vulns_bin = np.memmap(os.path.join(static_path, "vulnerability.bin"), dtype=Vulnerability, offset=4, mode='r')
            vuln_array = load_vulns_bin(vulns_bin, vuln_dict, num_damage_bins, num_intensity_bins)

    elif "vulnerability.csv" in input_files and file_type == "csv":
        vuln_csv = np.genfromtxt(os.path.join(static_path, "vulnerability.csv"), dtype=Vulnerability, delimiter=",")
        num_damage_bins = max(vuln_csv['damage_bin_id'])
        vuln_array = load_vulns_bin(vuln_csv, vuln_dict, num_damage_bins, num_intensity_bins)

    return vuln_array, num_damage_bins


def get_mean_damage_bins(static_path, file_type):
    """
    Loads the mean damage bins from the damage_bin_dict file.

    Args:
        static_path: (str) the path pointing to the static file where the data is
        file_type: (str) the file extension and thus the file type to be loaded

    Returns: (List[Union[damagebindictionaryCsv, damagebindictionary]]) loaded data from the damage_bin_dict file
    """
    file_path = os.path.join(static_path, "damage_bin_dict.bin")
    if not os.path.isfile(str(file_path)) and file_type == "csv":
        return np.genfromtxt(os.path.join(static_path, "damage_bin_dict.csv"), dtype=damagebindictionaryCsv)['interpolation']
    return np.fromfile(os.path.join(static_path, "damage_bin_dict.bin"), dtype=damagebindictionary)['interpolation']


@nb.jit(cache=True, fastmath=True)
def damage_bin_prob(p, intensities_min, intensities_max, vulns, intensities):
    """
    Calculate the probability of an event happening and then causing damage.

    Args:
        p: (float) the probability to be updated
        intensities_min: (int) intensity minimum
        intensities_max: (int) intensity maximum
        vulns: (List[float]) PLEASE FILL IN
        intensities: (List[float]) list of all the intensities

    Returns: (float) the updated probability
    """
    i = intensities_min
    while i < intensities_max:
        p += vulns[i] * intensities[i]
        i += 1
    return p


@nb.jit(cache=True, fastmath=True)
def do_result(vulns_id, vuln_array, mean_damage_bins,
              int32_mv, num_damage_bins,
              intensities_min, intensities_max, intensities,
              event_id, areaperil_id, vuln_i, cursor):
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
    int32_mv[cursor], cursor = event_id, cursor + 1
    int32_mv[cursor:cursor + areaperil_int_relative_size] = areaperil_id.view(oasis_int_dtype)
    cursor += areaperil_int_relative_size
    int32_mv[cursor], cursor = vulns_id[vuln_i], cursor + 1

    cur_vuln_mat = vuln_array[vuln_i]
    p = 0
    cursor_start = cursor
    cursor += 1
    oasis_float_mv = int32_mv[cursor: cursor + num_damage_bins * results_relative_size].view(oasis_float)
    result_cursor = 0
    damage_bin_i = 0

    while damage_bin_i < num_damage_bins:
        p = damage_bin_prob(p, intensities_min, intensities_max, cur_vuln_mat[damage_bin_i], intensities)
        oasis_float_mv[result_cursor], result_cursor = p, result_cursor + 1
        oasis_float_mv[result_cursor], result_cursor = mean_damage_bins[damage_bin_i], result_cursor + 1
        damage_bin_i += 1
        if p > 0.999999940:
            break

    int32_mv[cursor_start] = damage_bin_i
    return cursor + (result_cursor * oasis_float_relative_size)


@nb.njit()
def doCdf(event_id,
          num_intensity_bins, footprint,
          areaperil_to_vulns_idx_dict, areaperil_to_vulns_idx_array, areaperil_to_vulns,
          vuln_array, vulns_id, num_damage_bins, mean_damage_bins,
          int32_mv, max_result_relative_size):
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
        int32_mv: (List[int]) FILL IN LATER
        max_result_relative_size: (int) the maximum result size

    Returns: (int)
    """
    if not footprint.shape[0]:
        return 0

    intensities_min = num_intensity_bins
    intensities_max = 0
    intensities = np.zeros(num_intensity_bins, dtype=oasis_float)

    areaperil_id = np.zeros(1, dtype=areaperil_int)
    has_vuln = False
    cursor = 0

    for footprint_i in range(footprint.shape[0]):
        event_row = footprint[footprint_i]
        if areaperil_id[0] != event_row['areaperil_id']:
            if has_vuln and intensities_min <= intensities_max:
                areaperil_to_vulns_idx = areaperil_to_vulns_idx_array[areaperil_to_vulns_idx_dict[areaperil_id[0]]]
                intensities_max += 1
                for vuln_idx in range(areaperil_to_vulns_idx['start'], areaperil_to_vulns_idx['end']):
                    vuln_i = areaperil_to_vulns[vuln_idx]
                    if cursor + max_result_relative_size > buff_int_size:
                        yield cursor * oasis_int_size
                        cursor = 0

                    cursor = do_result(vulns_id, vuln_array, mean_damage_bins,
                              int32_mv, num_damage_bins,
                              intensities_min, intensities_max, intensities,
                              event_id, areaperil_id, vuln_i, cursor)

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
            if cursor + max_result_relative_size > buff_int_size:
                yield cursor * oasis_int_size
                cursor = 0

            cursor = do_result(vulns_id, vuln_array, mean_damage_bins,
                      int32_mv, num_damage_bins,
                      intensities_min, intensities_max, intensities,
                      event_id, areaperil_id, vuln_i, cursor)

    yield cursor * oasis_int_size


def run(run_dir, file_in, file_out, file_type):
    """
    Runs the main process of the getmodel process.

    Args:
        run_dir: (str) the directory of where the process is running
        file_in: (Optional[str]) the path to the input directory
        file_out: (Optional[str]) the path to the output directory
        file_type: (str) the type of file being loaded

    Returns: None
    """
    with ExitStack() as stack:
        if file_in is None:
            streams_in = sys.stdin.buffer
        else:
            streams_in = stack.enter_context(open(file_in, 'rb'))

        if file_out is None:
            stream_out = sys.stdout.buffer
        else:
            stream_out = stack.enter_context(open(file_out, 'wb'))

        static_path = os.path.join(run_dir, 'static')
        input_path = os.path.join(run_dir, 'input')

        event_id_mv = memoryview(bytearray(4))
        event_ids = np.ndarray(1, buffer=event_id_mv, dtype='i4')

        logger.debug('init items')
        vuln_dict, vulns_id, areaperil_to_vulns_idx_dict, areaperil_to_vulns_idx_array, areaperil_to_vulns = get_items(input_path, file_type)

        logger.debug('init footprint')
        footprint_obj = stack.enter_context(Footprint.load(static_path))
        num_intensity_bins =  footprint_obj.num_intensity_bins

        logger.debug('init vulnerability')
        vuln_array, num_damage_bins = get_vulns(static_path, vuln_dict, num_intensity_bins, file_type)

        logger.debug('init mean_damage_bins')
        mean_damage_bins = get_mean_damage_bins(static_path, file_type)

        # even_id, areaperil_id, vulnerability_id, num_result, [oasis_float] * num_result
        max_result_relative_size = 1 + + areaperil_int_relative_size + 1 + 1 + num_damage_bins * results_relative_size

        mv = memoryview(bytearray(buff_size))

        int32_mv = np.ndarray(buff_size // np.int32().itemsize, buffer=mv, dtype=np.int32)

        # header
        stream_out.write(np.uint32(1).tobytes())

        while True:
            len_read = streams_in.readinto(event_id_mv)
            if len_read==0:
                break
            event_footprint = footprint_obj.get_event(event_ids[0])
            if event_footprint is not None:
                for cursor_bytes in doCdf(event_ids[0],
                      num_intensity_bins, event_footprint,
                      areaperil_to_vulns_idx_dict, areaperil_to_vulns_idx_array, areaperil_to_vulns,
                      vuln_array, vulns_id, num_damage_bins, mean_damage_bins,
                                          int32_mv, max_result_relative_size):

                    if cursor_bytes:
                        stream_out.write(mv[:cursor_bytes])
                    else:
                        break
        logger.debug('getmodel done')

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--file-in', help='names of the input file_path')
parser.add_argument('-o', '--file-out', help='names of the output file_path')
parser.add_argument('-r', '--run-dir', help='path to the run directory', default='.')
parser.add_argument('-f', '--file-type', help='the type of file to be loaded', default='bin')
parser.add_argument('-v', '--logging-level', help='logging level (debug:10, info:20, warning:30, error:40, critical:50)',
                    default=30, type=int)


def main():
    kwargs = vars(parser.parse_args())

    # add handler to fm logger
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logging_level = kwargs.pop('logging_level')
    logger.setLevel(logging_level)

    run(**kwargs)


if __name__ == "__main__":
    main()



