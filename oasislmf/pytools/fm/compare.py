import numpy as np
from numpy.testing import assert_allclose
from .stream import event_agg_dtype, sidx_loss_dtype, EXTRA_VALUES
from .common import np_oasis_float
from .financial_structure import load_static


def stream_to_dict_array(stream_obj):
    stream_type = stream_obj.read(4)
    len_sample = np.frombuffer(stream_obj.read(4), dtype=np.int32)[0]

    buf = bytearray(8)
    mv = memoryview(buf)

    event_agg = np.ndarray(1, buffer=mv, dtype=event_agg_dtype)
    sidx_loss = np.ndarray(1, buffer=mv, dtype=sidx_loss_dtype)

    event_id_last = 0
    event_id, agg_id = 0, 0
    dict_array = {}
    while stream_obj.readinto(mv):
        if agg_id:
            sidx, loss = sidx_loss[0]
            if sidx == -3:
                sidx = 0
            elif sidx == -2:
                continue
            elif sidx==0:
                agg_id = 0
                continue
            cur_array[sidx] = 0 if np.isnan(loss) else loss
        else:
            event_id, agg_id = event_agg[0]
            # if event_id_last != event_id:
            #     if event_id_last:
            #         break
            #     else:
            #         event_id_last = event_id
            cur_array = np.zeros(len_sample + EXTRA_VALUES, dtype=np_oasis_float)
            dict_array[(event_id, agg_id)] = cur_array

    return stream_type, len_sample, dict_array


def round_dict_array(dict_array, precision):
    for key, values in dict_array.items():
        values.round(decimals=precision, out=values)


def dict_array_to_np_array(dict_array, len_sample):
    res = np.empty(len(dict_array), dtype = np.dtype(f"i4, i4, ({len_sample + EXTRA_VALUES})f4"))
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
                msg += f"    {len(missing)} missing in {i+1} : {sorted(missing)[:10]}" \
                       f"{'...' if len(missing)>10 else ''}\n"
        return msg

    # round_dict_array(dict_array1, precision)
    # round_dict_array(dict_array2, precision)

    msg_list = []
    mismatch = 0
    for i, key in enumerate(dict_array1):
        try:
            assert_allclose(dict_array1[key], dict_array2[key], precision)
        except AssertionError:
            mismatch+=1
            msg_list.append(f"value mismatch for {key} index {i}:\n\t{dict_array_gul.get(key)}\n\t{dict_array1[key]}\n\t{dict_array2[key]}")
            output_id, agg_id, layer_id = np.extract(fm_xref['output_id'] == key[1], fm_xref)[0]
            cur_level, cur_agg_id = 1, agg_id
            while True:
                policytc = np.extract(np.logical_and(fm_policytc['level_id'] == cur_level,
                                                     fm_policytc['agg_id'] == agg_id,
                                                     np.logical_or(fm_policytc['layer_id'] == layer_id, fm_policytc['layer_id'] == 1)),
                                         fm_policytc)
                if policytc.shape[0] > 1:
                    policytc_id = np.extract(policytc['layer_id'] == layer_id, policytc['policytc_id'])
                    true_layer = layer_id
                else:
                    policytc_id = np.extract(policytc['layer_id'] == 1, policytc['policytc_id'])
                    true_layer = 1
                profile = np.extract(fm_profile['policytc_id'] == policytc_id, fm_profile)
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