import numpy as np
from numpy.testing import assert_allclose
from oasislmf.pytools.common.data import def_to_type_and_size, loss
from .common import EXTRA_SIDX_COUNT
from .financial_structure import load_static

loss_type, _ = def_to_type_and_size(loss)

# Define dtypes for reading binary stream format
event_agg_dtype = np.dtype([('event_id', 'i4'), ('item_id', 'i4')])
sidx_loss_dtype = np.dtype([('sidx', 'i4'), ('loss', loss_type)])
# sidx_loss_dtype = loss_pair_dtype


def stream_to_dict_array(stream_obj):
    stream_type = stream_obj.read(4)
    len_sample = np.frombuffer(stream_obj.read(4), dtype=np.int32)[0]

    event_agg_buf = bytearray(event_agg_dtype.itemsize)
    event_agg_mv = memoryview(event_agg_buf)

    sidx_loss_buf = bytearray(sidx_loss_dtype.itemsize)
    sidx_loss_mv = memoryview(sidx_loss_buf)

    event_agg = np.ndarray(1, buffer=event_agg_mv, dtype=event_agg_dtype)
    sidx_loss = np.ndarray(1, buffer=sidx_loss_mv, dtype=sidx_loss_dtype)

    event_id, agg_id = 0, 0
    dict_array = {}
    while True:
        if agg_id:
            if stream_obj.readinto(sidx_loss_mv) < sidx_loss_dtype.itemsize:
                break
            sidx, loss = sidx_loss[0]
            if sidx == -3:
                sidx = 0
            elif sidx == -2:
                continue
            elif sidx == 0:
                agg_id = 0
                continue
            cur_array[sidx] = 0 if np.isnan(loss) else loss
        else:
            if stream_obj.readinto(event_agg_mv) < event_agg_dtype.itemsize:
                break
            event_id, agg_id = event_agg[0]
            cur_array = np.zeros(len_sample + EXTRA_SIDX_COUNT, dtype=oasis_float)
            dict_array[(event_id, agg_id)] = cur_array

    return stream_type, len_sample, dict_array


def round_dict_array(dict_array, precision):
    for key, values in dict_array.items():
        values.round(decimals=precision, out=values)


def dict_array_to_np_array(dict_array, len_sample):
    res_dtype = np.dtype([('event_id', 'i4'), ('agg_id', 'i4'), ('loss', oasis_float, (len_sample + EXTRA_SIDX_COUNT))])
    res = np.empty(len(dict_array), dtype=res_dtype)
    for i, (event_id, agg_id) in enumerate(sorted(dict_array)):
        res[i] = event_id, agg_id, dict_array[(event_id, agg_id)]
    return res


def compare_streams(gul_stream, fm_stream_obj1, fm_stream_obj2, precision):
    fm_programme, fm_policytc, fm_profile, fm_xref, _, _ = load_static('./input')

    _, _, dict_array_gul = stream_to_dict_array(gul_stream)
    stream_type1, len_sample1, dict_array1 = stream_to_dict_array(fm_stream_obj1)
    stream_type2, len_sample2, dict_array2 = stream_to_dict_array(fm_stream_obj2)

    if stream_type1 != stream_type2:
        return f"stream have different type: {stream_type1}, {stream_type2}"

    if len_sample1 != len_sample2:
        return f"stream have different len_sample: {len_sample1}, {len_sample2}"

    keys1 = set(dict_array1)
    keys2 = set(dict_array2)

    missing_in_1 = keys2 - keys1
    missing_in_2 = keys1 - keys2

    if missing_in_1 or missing_in_2:
        msg = "some event_id, agg_id are not matching\n"
        for i, missing in enumerate([missing_in_1, missing_in_2]):
            if missing:
                msg += f"    {len(missing)} missing in {i + 1} : {sorted(missing)[:10]}" \
                    f"{'...' if len(missing) > 10 else ''}\n"
        return msg

    # round_dict_array(dict_array1, precision)
    # round_dict_array(dict_array2, precision)

    msg_list = []
    mismatch = 0
    for i, key in enumerate(dict_array1):
        try:
            assert_allclose(dict_array1[key], dict_array2[key], precision)
        except AssertionError:
            mismatch += 1
            msg_list.append(f"value mismatch for {key} index {i}:\n\t{dict_array_gul.get(key)}\n\t{dict_array1[key]}\n\t{dict_array2[key]}")
            output_id, agg_id, layer_id = np.extract(fm_xref['output_id'] == key[1], fm_xref)[0]
            cur_level, cur_agg_id = 1, agg_id
            while True:
                policytc = np.extract(np.logical_and(fm_policytc['level_id'] == cur_level,
                                                     fm_policytc['agg_id'] == agg_id,
                                                     np.logical_or(fm_policytc['layer_id'] == layer_id, fm_policytc['layer_id'] == 1)),
                                      fm_policytc)
                if policytc.shape[0] > 1:
                    profile_id = np.extract(policytc['layer_id'] == layer_id, policytc['profile_id'])
                    true_layer = layer_id
                else:
                    profile_id = np.extract(policytc['layer_id'] == 1, policytc['profile_id'])
                    true_layer = 1
                profile = np.extract(fm_profile['profile_id'] == profile_id, fm_profile)
                msg_list.append(str((cur_level, agg_id, true_layer, profile)))
                cur_level += 1
                parent = np.extract(np.logical_and(fm_programme['from_agg_id'] == agg_id,
                                                   fm_programme['level_id'] == cur_level),
                                    fm_programme)
                if not parent:
                    break
                else:
                    brothers = np.extract(np.logical_and(fm_programme['to_agg_id'] == parent['to_agg_id'],
                                                         fm_programme['level_id'] == cur_level),
                                          fm_programme)
                    msg_list.append(f"prothers {brothers}")

            if mismatch > 10:
                msg_list.append("...")
                break

    return "\n".join(msg_list)
