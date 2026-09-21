import numba as nb
import numpy as np

from oasislmf.pytools.common.data import areaperil_int, oasis_float, oasis_int, item_adjustment_dtype, item_id, NAME_DTYPE_SLICE

# gul stream type
# probably need to set this dynamically depending on the stream type
gul_header = np.int32(1 | 2 << 24).tobytes()

# define the damage_bin_dict damage_types
DAMAGE_TYPE_DEFAULT = 0
DAMAGE_TYPE_RELATIVE = 1
DAMAGE_TYPE_ABSOLUTE = 2
DAMAGE_TYPE_DURATION = 3
VALID_DAMAGE_TYPE = {DAMAGE_TYPE_DEFAULT, DAMAGE_TYPE_RELATIVE, DAMAGE_TYPE_ABSOLUTE, DAMAGE_TYPE_DURATION}

ItemAdjustment = nb.from_dtype(item_adjustment_dtype)

items_data_type = nb.from_dtype(np.dtype([item_id[NAME_DTYPE_SLICE],
                                          ('damagecdf_i', oasis_int),
                                          ('rng_index', oasis_int)
                                          ]))

items_MC_data_type = nb.from_dtype(np.dtype([item_id[NAME_DTYPE_SLICE],
                                             ('item_idx', oasis_int),
                                             ('haz_arr_i', oasis_int),
                                             ('rng_index', oasis_int),
                                             ('hazard_rng_index', oasis_int),
                                             ('intensity_adjustment', oasis_int),
                                             ('return_period', oasis_int),
                                             ('event_rp', oasis_int),
                                             ('eff_cdf_id', oasis_int),
                                             # signed: -1 means no source item, so this item is
                                             # independent even inside a dependent coverage
                                             ('source_item_j', np.int32),
                                             # signed: magnitude is the building count, negative
                                             # means the buildings stay separate
                                             ('packed_buildings', oasis_int),
                                             # the correlation actually applied to this item's
                                             # damage draws, so 0 when correlation is off -- the
                                             # writer needs the effective value, not the file's
                                             ('damage_correlation_value', oasis_float),
                                             ]))

VulnCdfLookup = nb.from_dtype(np.dtype([('start', oasis_int),
                                        ('length', oasis_int)]))

coverage_type = nb.from_dtype(np.dtype([('tiv', np.float64),
                                        ('max_items', np.int32),
                                        ('start_items', np.int32),
                                        ('cur_items', np.int32)
                                        ]))

NP_BASE_ARRAY_SIZE = 8

# Structured dtype for merged aggregate vulnerability sub-entries (vuln_idx + weight in one record).
agg_vuln_idx_weight_dtype = np.dtype([('vuln_idx', oasis_int), ('weight', oasis_float)])


AREAPERIL_TO_EFF_VULN_KEY_TYPE = nb.types.Tuple((nb.from_dtype(areaperil_int), nb.types.int64))
AREAPERIL_TO_EFF_VULN_VALUE_TYPE = nb.types.UniTuple(nb.types.int32, 2)

# compute the relative size of oasis_float and areaperil_int vs int32
oasis_float_to_int32_size = oasis_float.itemsize // np.int32().itemsize
areaperil_int_to_int32_size = areaperil_int.itemsize // np.int32().itemsize

haz_arr_type = nb.from_dtype(np.dtype([('probability', oasis_float),
                                       ('intensity_bin_id', np.int32),
                                       ('intensity', np.int32)]))

ProbMean = nb.from_dtype(np.dtype([('prob_to', oasis_float),
                                   ('bin_mean', oasis_float)
                                   ]))
ProbMean_size = ProbMean.size

damagecdfrec_stream = nb.from_dtype(np.dtype([('event_id', np.int32),
                                              ('areaperil_id', areaperil_int),
                                              ('vulnerability_id', np.int32)
                                              ]))

damagecdfrec = nb.from_dtype(np.dtype([('areaperil_id', areaperil_int),
                                       ('vulnerability_id', np.int32)
                                       ]))


gulSampleslevelHeader = nb.from_dtype(np.dtype([('event_id', 'i4'),
                                                ('item_id', 'i4'),
                                                ]))
gulSampleslevelHeader_size = gulSampleslevelHeader.size

gulSampleslevelRec = nb.from_dtype(np.dtype([('sidx', 'i4'),
                                             ('loss', oasis_float),
                                             ]))
gulSampleslevelRec_size = gulSampleslevelRec.size


Keys = {'LocID': np.int32,
        'PerilID': 'category',
        'CoverageTypeID': np.int32,
        'AreaPerilID': areaperil_int,
        'VulnerabilityID': np.int32}


NormInversionParameters = nb.from_dtype(np.dtype([('x_min', np.float64),
                                                  ('x_max', np.float64),
                                                  ('N', np.int32),
                                                  ('cdf_min', np.float64),
                                                  ('cdf_max', np.float64),
                                                  ('inv_factor', np.float64),
                                                  ('norm_factor', np.float64),
                                                  ]))

gulmc_compute_info_type = nb.from_dtype(np.dtype([
    ('event_id', oasis_int),
    # cursor and max_bytes_per_item count BYTES of the output buffer, not stream values. They are
    # int64 because nothing about the stream bounds them: a sidx is int32 because that is how it
    # is written, but the buffer is ordinary memory. Holding them in an int32 put a 2 GB ceiling
    # on the buffer for no reason other than the choice of field, and a kept-separate item with
    # enough buildings reaches it.
    ('cursor', np.int64),
    ('coverage_i', oasis_int),  # last_processed_coverage_ids_idx
    ('coverage_n', oasis_int),
    # Resume point WITHIN coverage_i, so a buffer flush need not fall on a coverage boundary:
    # item_j is the next item of that coverage to process, building_b how many of its buildings
    # have already been emitted (0 = none, so the item header is still to be written). Both are
    # cleared as each item and coverage completes, which is what makes the common path start at
    # the top. An item's records need only be contiguous in the STREAM, not in the buffer, so
    # flushing part-way through one is safe.
    ('item_j', oasis_int),
    ('building_b', oasis_int),
    ('cdf_cache_ctr', np.int64),
    ('max_bytes_per_item', np.int64),
    # bytes one building's block can take: its specials, its samples, and -- counted whether or
    # not this block carries them -- the item header and delimiter
    ('max_bytes_per_block', np.int64),
    ('Ndamage_bins_max', oasis_int),
    ('sample_size', oasis_int),
    ('loss_threshold', oasis_float),
    ('alloc_rule', np.int8),
    ('do_correlation', np.int8),
    ('do_haz_correlation', np.int8),
    ('effective_damageability', np.int8),
    ('debug', np.int8),
    ('do_coverage_dependency', np.int8),
]))
