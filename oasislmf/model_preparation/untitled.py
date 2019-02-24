def get_sub_layer_calcrule_id_for_dict_or_pandas_series(dict_or_series):
    ds = dict_or_series

    if (ds['deductible'] > 0 and ds['deductible_code'] == 0) and (ds['deductible_min'] == ds['deductible_max'] == 0) and (ds['limit'] > 0 and ds['limit_code'] == 0):
        return 1
    elif (ds['deductible'] > 0 and ds['deductible_code'] == 2) and (ds['deductible_min'] == ds['deductible_max'] == 0) and (ds['limit'] > 0 and ds['limit_code'] == 0):
        return 4
    elif (ds['deductible'] > 0 and ds['deductible_code'] == 1) and (ds['deductible_min'] == ds['deductible_max'] == 0) and (ds['limit'] > 0 and ds['limit_code'] == 1):
        return 5
    elif (ds['deductible'] > 0 and ds['deductible_code'] == 2) and (ds['deductible_min'] == ds['deductible_max'] == 0) and (ds['limit'] == ds['limit_code'] == 0):
        return 6
    elif (ds['deductible'] == ds['deductible_code'] == 0) and (ds['deductible_min'] == 0 and ds['deductible_max'] > 0) and (ds['limit'] > 0 and ds['limit_code'] == 0):
        return 7
    elif (ds['deductible'] == ds['deductible_code'] == 0) and (ds['deductible_min'] > 0 and ds['deductible_max'] == 0) and (ds['limit'] > 0 and ds['limit_code'] == 0):
        return 8
    elif (ds['deductible'] == ds['deductible_code'] == 0) and (ds['deductible_min'] == 0 and ds['deductible_max'] > 0) and (ds['limit'] == ds['limit_code'] == 0):
        return 10
    elif (ds['deductible'] == ds['deductible_code'] == 0) and (ds['deductible_min'] > 0 and ds['deductible_max'] == 0) and (ds['limit'] == ds['limit_code'] == 0):
        return 11
    elif (ds['deductible'] >= 0 and ds['deductible_code'] == 0) and (ds['deductible_min'] == ds['deductible_max'] == 0) and (ds['limit'] == ds['limit_code'] == 0):
        return 12
    elif (ds['deductible'] == ds['deductible_code'] == 0) and (ds['deductible_min'] > 0 and ds['deductible_max'] > 0) and (ds['limit'] == ds['limit_code'] == 0):
        return 13
    elif (ds['deductible'] == ds['deductible_code'] == 0) and (ds['deductible_min'] == ds['deductible_max'] == 0) and (ds['limit'] > 0 and ds['limit_code'] == 0):
        return 14
    elif (ds['deductible'] == ds['deductible_code'] == 0) and (ds['deductible_min'] == ds['deductible_max'] == 0) and (ds['limit'] > 0 and ds['limit_code'] == 1):
        return 15
    elif (ds['deductible'] > 0 and ds['deductible_code'] == 1) and (ds['deductible_min'] == ds['deductible_max'] == 0) and (ds['limit'] == ds['limit_code'] == 0):
        return 16
    elif (ds['deductible'] > 0 and ds['deductible_code'] == 1) and (ds['deductible_min'] > 0 and ds['deductible_max'] > 0) and (ds['limit'] == ds['limit_code'] == 0):
        return 19
    elif (ds['deductible'] > 0 and ds['deductible_code'] in [0, 2]) and (ds['deductible_min'] > 0 and ds['deductible_max'] > 0) and (ds['limit'] == ds['limit_code'] == 0):
        return 21