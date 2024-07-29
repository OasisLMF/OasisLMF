import logging
import os
import shutil

import numpy as np
import pandas as pd

from ods_tools.oed import fill_empty

from ..utils.exceptions import OasisException
from ..utils.log import oasis_log
from ..utils.data import get_dataframe
from . import oed


REINS_RISK_LEVEL_XREF_COLUMN_MAP = {
    oed.REINS_RISK_LEVEL_LOCATION_GROUP: ["LocGroup", "PortNumber", "AccNumber", "PolNumber", "LocNumber"],
    oed.REINS_RISK_LEVEL_LOCATION: ["PortNumber", "AccNumber", "LocNumber", "PolNumber"]
}
XREF_COLUMN_DEFAULT = ["PortNumber", "AccNumber", "PolNumber", "LocNumber"]

RISK_LEVEL_FIELD_MAP = {
    oed.REINS_RISK_LEVEL_PORTFOLIO: ['PortNumber'],
    oed.REINS_RISK_LEVEL_ACCOUNT: ['PortNumber', 'AccNumber'],
    oed.REINS_RISK_LEVEL_POLICY: ['PortNumber', 'AccNumber', 'PolNumber'],
    oed.REINS_RISK_LEVEL_LOCATION_GROUP: ['LocGroup'],
    oed.REINS_RISK_LEVEL_LOCATION: ['PortNumber', 'AccNumber', 'LocNumber']
}
RISK_LEVEL_ALL_FIELDS = ['PortNumber', 'AccNumber', 'PolNumber', 'LocGroup', 'LocNumber']
FILTER_LEVEL_EXTRA_FIELDS = ['CedantName', 'ProducerName', 'LOB', 'CountryCode', 'ReinsTag']

CHECK_RI_SCOPE_MAP = {
    oed.REINS_RISK_LEVEL_ACCOUNT: {'in': ['AccNumber'], 'out': ['PolNumber', 'LocNumber']},
    oed.REINS_RISK_LEVEL_POLICY: {'in': ['AccNumber', 'PolNumber'], 'out': ['LocNumber']},
    oed.REINS_RISK_LEVEL_LOCATION: {'in': ['AccNumber', 'LocNumber']},
    oed.REINS_RISK_LEVEL_LOCATION_GROUP: {'in': ['LocGroup']},
}

REINS_TYPE_EXACT_MATCH = [oed.REINS_TYPE_FAC, oed.REINS_TYPE_SURPLUS_SHARE]

NO_LOSS_PROFILE_ID = 1
PASSTHROUGH_PROFILE_ID = 2

NO_LOSS_PROFILE = dict(
    calcrule_id=oed.CALCRULE_ID_LIMIT_ONLY,
    deductible1=0.0,  # Not used
    deductible2=0.0,  # Not used
    deductible3=0.0,  # Not used
    attachment=0.0,   # Not used
    limit=0.0,
    share1=0.0,       # Not used
    share2=0.0,       # Not used
    share3=0.0        # Not used
)

PASSTHROUGH_PROFILE = dict(
    calcrule_id=oed.CALCRULE_ID_DEDUCTIBLE_ONLY,
    deductible1=0.0,
    deductible2=0.0,  # Not used
    deductible3=0.0,  # Not used
    attachment=0.0,   # Not used
    limit=0.0,        # Not used
    share1=0.0,       # Not used
    share2=0.0,       # Not used
    share3=0.0        # Not used
)


ITEM_LEVEL_ID = 1
FILTER_LEVEL_ID = 2
RISK_LEVEL_ID = 3
PROGRAM_LEVEL_ID = 4

FM_TERMS_PER_REINS_TYPE = {
    oed.REINS_TYPE_FAC: {
        RISK_LEVEL_ID: {
            'calcrule_id': {'default': oed.CALCRULE_ID_OCCURRENCE_CATASTROPHE_EXCESS_OF_LOSS},
            'deductible1': {'default': 0.},
            'deductible2': {'default': 0.},
            'deductible3': {'default': 0.},
            'attachment': {'oed_col': 'RiskAttachment', 'default': 0.},
            'limit': {'oed_col': 'RiskLimit', 'default': oed.LARGE_VALUE, 'to_default': [0.]},
            'share1': {'oed_col': 'CededPercent', 'default': 1.},
            'share2': {'oed_col': 'PlacedPercent', 'default': 1.},
            'share3': {'default': 1.},
        }
    },
    oed.REINS_TYPE_PER_RISK: {
        RISK_LEVEL_ID: {
            'calcrule_id': {'default': oed.CALCRULE_ID_OCCURRENCE_CATASTROPHE_EXCESS_OF_LOSS},
            'deductible1': {'default': 0.},
            'deductible2': {'default': 0.},
            'deductible3': {'default': 0.},
            'attachment': {'oed_col': 'RiskAttachment', 'default': 0.},
            'limit': {'oed_col': 'RiskLimit', 'default': oed.LARGE_VALUE, 'to_default': [0.]},
            'share1': {'oed_col': 'CededPercent', 'default': 1.},
            'share2': {'default': 1.},
            'share3': {'default': 1.},
        },
        PROGRAM_LEVEL_ID: {
            'calcrule_id': {'default': oed.CALCRULE_ID_OCCURRENCE_CATASTROPHE_EXCESS_OF_LOSS},
            'deductible1': {'default': 0.},
            'deductible2': {'default': 0.},
            'deductible3': {'default': 0.},
            'attachment': {'oed_col': 'OccAttachment', 'default': 0.},
            'limit': {'oed_col': 'OccLimit', 'default': oed.LARGE_VALUE, 'to_default': [0.]},
            'share1': {'default': 1.},
            'share2': {'oed_col': 'PlacedPercent', 'default': 1.},
            'share3': {'default': 1.},
        }
    },
    oed.REINS_TYPE_QUOTA_SHARE: {
        RISK_LEVEL_ID: {
            'calcrule_id': {'default': oed.CALCRULE_ID_OCCURRENCE_CATASTROPHE_EXCESS_OF_LOSS},
            'deductible1': {'default': 0.},
            'deductible2': {'default': 0.},
            'deductible3': {'default': 0.},
            'attachment': {'default': 0.},
            'limit': {'oed_col': 'RiskLimit', 'default': oed.LARGE_VALUE, 'to_default': [0.]},
            'share1': {'oed_col': 'CededPercent', 'default': 1.},
            'share2': {'default': 1.},
            'share3': {'default': 1.},
        },
        PROGRAM_LEVEL_ID: {
            'calcrule_id': {'default': oed.CALCRULE_ID_OCCURRENCE_LIMIT_AND_SHARE},
            'deductible1': {'default': 0.},
            'deductible2': {'default': 0.},
            'deductible3': {'default': 0.},
            'attachment': {'default': 0.},
            'limit': {'oed_col': 'OccLimit', 'default': oed.LARGE_VALUE, 'to_default': [0.]},
            'share1': {'default': 0.},
            'share2': {'oed_col': 'PlacedPercent', 'default': 1.},
            'share3': {'default': 1.},
        }
    },
    oed.REINS_TYPE_SURPLUS_SHARE: {
        RISK_LEVEL_ID: {
            'calcrule_id': {'default': oed.CALCRULE_ID_OCCURRENCE_CATASTROPHE_EXCESS_OF_LOSS},
            'deductible1': {'default': 0.},
            'deductible2': {'default': 0.},
            'deductible3': {'default': 0.},
            'attachment': {'oed_col': 'RiskAttachment', 'default': 0.},
            'limit': {'oed_col': 'RiskLimit', 'default': oed.LARGE_VALUE, 'to_default': [0.]},
            'share1': {'oed_col': 'CededPercent_scope', 'default': 1.},
            'share2': {'default': 1.},
            'share3': {'default': 1.},
        },
        PROGRAM_LEVEL_ID: {
            'calcrule_id': {'default': oed.CALCRULE_ID_OCCURRENCE_LIMIT_AND_SHARE},
            'deductible1': {'default': 0.},
            'deductible2': {'default': 0.},
            'deductible3': {'default': 0.},
            'attachment': {'default': 0.},
            'limit': {'oed_col': 'OccLimit', 'default': oed.LARGE_VALUE, 'to_default': [0.]},
            'share1': {'default': 0.},
            'share2': {'oed_col': 'PlacedPercent', 'default': 1.},
            'share3': {'default': 1.},
        }
    },
    oed.REINS_TYPE_CAT_XL: {
        PROGRAM_LEVEL_ID: {
            'calcrule_id': {'default': oed.CALCRULE_ID_OCCURRENCE_CATASTROPHE_EXCESS_OF_LOSS},
            'deductible1': {'default': 0.},
            'deductible2': {'default': 0.},
            'deductible3': {'default': 0.},
            'attachment': {'oed_col': 'OccAttachment', 'default': 0.},
            'limit': {'oed_col': 'OccLimit', 'default': oed.LARGE_VALUE, 'to_default': [0.]},
            'share1': {'oed_col': 'CededPercent', 'default': 1.},
            'share2': {'oed_col': 'PlacedPercent', 'default': 1.},
            'share3': {'default': 1.},
        }
    }
}


def create_risk_level_profile_id(ri_df, profile_map_df, fm_profile_df, reins_type, risk_level, fm_level_id):
    """
    Create new profile id from reinsurance in ri_df corresponding to reins_type.
    Add them to fm_profile_df and match the profile_ids in ri_df and profile_map_df
    Args:
        ri_df: ri info and scope
        profile_map_df: tree structure df representing each ri fm levels
        fm_profile_df: df containing all the profiles
        reins_type: type of reinsurance (one of oed.REINS_TYPES)
        risk_level: level of the reinsurance terms (one of oed.REINS_RISK_LEVELS)
        fm_level_id: fm level in profile_map_df

    Returns:
        fm_profile_df: updated version of fm_profile_df
    """
    reins_type_filter = ri_df['ReinsType'] == reins_type
    if not reins_type_filter.any():
        return fm_profile_df

    # create new fm profile from ri terms corresponding to the reins_type
    ri_term_map = {term_info['oed_col']: term for term, term_info in FM_TERMS_PER_REINS_TYPE[reins_type].get(fm_level_id, {}).items()
                   if 'oed_col' in term_info}

    if ri_term_map:
        # create a profile_id for each unique term combination
        ri_df.loc[reins_type_filter, 'profile_id'] = pd.factorize(pd._libs.lib.fast_zip([
            ri_df[reins_type_filter][col].values for col in ri_term_map
        ]))[0] + 1 + fm_profile_df['profile_id'].max()

        # create complete profile from terms and add it to fm_profile_df
        cur_fm_profiles = ri_df[reins_type_filter][['profile_id'] + list(ri_term_map)].rename(columns=ri_term_map)
        for term, term_info in FM_TERMS_PER_REINS_TYPE[reins_type][fm_level_id].items():
            if term not in cur_fm_profiles:
                cur_fm_profiles[term] = term_info['default']
            else:
                fill_empty(cur_fm_profiles, [term], term_info['default'])
                cur_fm_profiles.loc[cur_fm_profiles[term].isin(term_info.get('to_default', [])), term] = term_info['default']
        cur_fm_profiles.drop_duplicates(inplace=True)
        if not cur_fm_profiles.empty:
            fm_profile_df = pd.concat([fm_profile_df, cur_fm_profiles], ignore_index=True)

    else:  # No terms at risk level
        ri_df.loc[reins_type_filter, 'profile_id'] = PASSTHROUGH_PROFILE_ID

    # update profile_map profile_id for filter and risk level
    layer_id_set = set(ri_df[reins_type_filter]['layer_id'])
    these_profile_map_layers = profile_map_df[profile_map_df['layer_id'].isin(layer_id_set)]

    # filter level
    if fm_level_id == RISK_LEVEL_ID:
        ri_filter_fields = RISK_LEVEL_ALL_FIELDS + [field for field in FILTER_LEVEL_EXTRA_FIELDS if field in ri_df]
        ri_filter_valid_fields = [field + '_valid' for field in ri_filter_fields]
        merge_on = ['layer_id']
        if reins_type in REINS_TYPE_EXACT_MATCH:
            merge_on += RISK_LEVEL_FIELD_MAP[risk_level]
        filter_df = (these_profile_map_layers[these_profile_map_layers['level_id'] == FILTER_LEVEL_ID]
                     .reset_index()
                     .merge(ri_df[reins_type_filter][ri_filter_fields + ri_filter_valid_fields + ['layer_id']], how='inner', on=merge_on))

        def _match(row):
            for field in ri_filter_fields:
                if (field not in merge_on
                        and row[f'{field}_valid'] and row[f'{field}_x'] != row[f'{field}_y']):
                    return False
            return True
        profile_map_df.loc[np.unique(filter_df.loc[filter_df.apply(_match, axis=1), 'index']), 'profile_id'] = PASSTHROUGH_PROFILE_ID

    # Risk level
    layer_filter = reins_type_filter
    match_fields = list(RISK_LEVEL_FIELD_MAP[risk_level]) if fm_level_id == RISK_LEVEL_ID else []

    while layer_filter.any():
        ri_filter = layer_filter & (ri_df[match_fields] != '').all(axis=1)

        df_ = these_profile_map_layers[these_profile_map_layers['level_id'] == fm_level_id].reset_index().merge(
            ri_df[ri_filter][match_fields + ['layer_id', 'profile_id']].drop_duplicates(),
            how='inner', on=match_fields + ['layer_id'], suffixes=['', '_y']
        ).set_index('index')

        profile_map_df.loc[df_.index, 'profile_id'] = df_['profile_id_y']

        # match field are ordered by granularity, if an exact match is not needed we pop the most granular column and look for more general match
        if match_fields:
            if reins_type in REINS_TYPE_EXACT_MATCH:
                break
            else:
                layer_filter = layer_filter & (ri_df[match_fields.pop()] == '')
        else:
            break

    return fm_profile_df


def check_ri_scope_filter(ri_df, risk_level):
    # For some treaty types the scope filter must match exactly
    column_in = CHECK_RI_SCOPE_MAP.get(risk_level, {}).get('in', [])
    column_out = CHECK_RI_SCOPE_MAP.get(risk_level, {}).get('out', [])
    return (
        (~ri_df['ReinsType'].isin(REINS_TYPE_EXACT_MATCH))
        | (
            (column_in == [] or (ri_df[column_in] != "").all(axis=1))
            & (column_out == [] or (ri_df[column_out] == "").all(axis=1))
        )
    )


def get_xref_df(xref_descriptions_df, risk_level):
    """
    Build the cross-reference dataframe, which serves as a representation
    of the insurance programme depending on the reinsurance risk level.
    Dataframes for programme, risk, filter and items levels are created.
    The fields agg_id, level_id and to_agg_id (agg_id_to), which are used
    to construct the FM Programmes structure, are assigned. The
    aforementioned dataframes are concatenated to form a single dataframe
    called xref_df, which is returned. The returned dataframe features the
    fields necessary for the assignment of profile IDs.
    Args:
        xref_descriptions_df: Fm summary mapping enhanced by relevant information from Loc and Account
        risk_level: risk_level

    Returns:
        df_levels: list of dataframes, one per fm level
    """

    xref_descriptions = xref_descriptions_df.sort_values(by=REINS_RISK_LEVEL_XREF_COLUMN_MAP.get(risk_level, XREF_COLUMN_DEFAULT))
    risk_level_fields = RISK_LEVEL_FIELD_MAP[risk_level]

    df_levels = dict()
    # Programme level
    df_levels['programme_level'] = pd.DataFrame(
        {'agg_id': 1, 'level_id': PROGRAM_LEVEL_ID, 'agg_id_to': 0},
        index=[0]
    )

    # Risk level
    risk_level_df = xref_descriptions.drop_duplicates(
        subset=risk_level_fields, keep='first'
    ).reset_index(drop=True)
    risk_level_df['agg_id'] = risk_level_df.index + 1
    risk_level_df['level_id'] = RISK_LEVEL_ID
    risk_level_df['agg_id_to'] = 1
    df_levels['risk_level'] = risk_level_df

    # Filter level
    filter_level_df = xref_descriptions.drop_duplicates(
        subset=RISK_LEVEL_ALL_FIELDS
    ).reset_index(drop=True)
    filter_level_df['agg_id'] = filter_level_df.index + 1
    filter_level_df['level_id'] = FILTER_LEVEL_ID
    filter_level_df = filter_level_df.merge(
        risk_level_df[risk_level_fields + ['agg_id']], how='left', on=risk_level_fields,
        suffixes=['', '_to']
    )
    df_levels['filter_level'] = filter_level_df

    # Item level
    item_level_df = xref_descriptions.reset_index(drop=True)
    item_level_df['agg_id'] = item_level_df['output_id']
    item_level_df['level_id'] = ITEM_LEVEL_ID
    item_level_df = item_level_df.merge(
        filter_level_df[RISK_LEVEL_ALL_FIELDS + ['agg_id']], how='left',
        on=RISK_LEVEL_ALL_FIELDS, suffixes=['', '_to']
    )
    df_levels['items_level'] = item_level_df

    return df_levels


def _log_dataframe(logger, df_dict, ri_name):
    if logger.level >= logging.DEBUG:
        for df_name, df_ in df_dict.items():
            logger.debug(f'{df_name}: {ri_name}:')
            logger.debug(df_)


@oasis_log
def write_files_for_reinsurance(ri_info_df, ri_scope_df, xref_descriptions_df, output_dir, fm_xref_fp, logger):
    """
    Create the Oasis structures - FM Programmes, FM Profiles and FM Policy
    TCs - that represent the reinsurance structure.

    The cross-reference dataframe, which serves as a representation of the
    insurance programme depending on the reinsurance risk level, is built.
    Except facultative contracts, each contract is a
    separate layer. Profile IDs for the risk and filter levels are created
    using the merged reinsurance scope and info dataframes. These profile
    IDs are assigned according to some combination of the fields
    PortNumber, AccNumber, PolNumber, LocGroup and LocNumber, dependent on
    reinsurance risk level. Individual programme level profile IDs are
    assigned for each row of the reinsurance info dataframe. Finally, the
    Oasis structure is written out.
    """
    fm_xref_df = get_dataframe(fm_xref_fp)
    fm_xref_df['agg_id'] = range(1, 1 + len(fm_xref_df))
    fill_empty(ri_scope_df, RISK_LEVEL_ALL_FIELDS, '')

    reinsurance_index = 1
    inuring_metadata = {}
    for inuring_priority in range(1, ri_info_df['InuringPriority'].max() + 1):
        for risk_level in oed.REINS_RISK_LEVELS:
            cur_ri_info_df = ri_info_df[(ri_info_df['InuringPriority'] == inuring_priority) & (ri_info_df['RiskLevel'] == risk_level)]
            if cur_ri_info_df.empty:
                continue

            output_name = f"ri_{inuring_priority}_{risk_level}"
            df_levels = get_xref_df(xref_descriptions_df, risk_level)

            _log_dataframe(logger, df_levels, output_name)

            no_loss_profile = {'profile_id': NO_LOSS_PROFILE_ID, **NO_LOSS_PROFILE}
            pass_through_profile = {'profile_id': PASSTHROUGH_PROFILE_ID, **PASSTHROUGH_PROFILE}

            fm_profile_df = pd.DataFrame([no_loss_profile, pass_through_profile])

            xref_df = pd.concat(df_levels.values(), ignore_index=True)

            # Assign default profile IDs
            xref_df['profile_id'] = NO_LOSS_PROFILE_ID
            xref_df['profile_id'] = xref_df['profile_id'].where(
                xref_df['level_id'].isin([FILTER_LEVEL_ID, RISK_LEVEL_ID]),
                PASSTHROUGH_PROFILE_ID
            )

            # Merge RI info and scope dataframes, and assign layers
            # Use as few layers as possible for FAC
            # Otherwise separate layers for each contract
            logger.debug(
                'Merging RI info and scope dataframes and assigning layers'
            )
            risk_level_fields = RISK_LEVEL_FIELD_MAP[risk_level]
            ri_df = cur_ri_info_df.merge(ri_scope_df, on='ReinsNumber', suffixes=['', '_scope'])
            valid_rows = check_ri_scope_filter(ri_df, risk_level)
            if not valid_rows.all():
                raise OasisException(
                    f'Invalid combination of Risk Level and Reinsurance Type. Please check scope file:\n{ri_df[~valid_rows]}'
                )

            ri_df['layer_id'] = 0
            ri_df.loc[ri_df['ReinsType'] == oed.REINS_TYPE_FAC, 'layer_id'] = (ri_df.loc[ri_df['ReinsType'] == oed.REINS_TYPE_FAC]
                                                                               .groupby(risk_level_fields, observed=True).cumcount() + 1)
            ri_info_no_fac = cur_ri_info_df[cur_ri_info_df['ReinsType'] != oed.REINS_TYPE_FAC].reset_index(drop=True)
            ri_info_no_fac['layer_id'] = ri_info_no_fac.index + 1 + ri_df['layer_id'].max()
            ri_df = ri_df.merge(ri_info_no_fac, how='left', on=ri_info_no_fac.columns.to_list()[:-1], suffixes=['', '_y'])
            ri_df['layer_id'] = ri_df['layer_id'].where(ri_df['layer_id_y'].isna(), ri_df['layer_id_y'])
            ri_df = ri_df.drop('layer_id_y', axis=1)

            for field in RISK_LEVEL_ALL_FIELDS:
                ri_df[field + '_valid'] = (ri_df[field] != '')
            for field in FILTER_LEVEL_EXTRA_FIELDS:
                if field in ri_df.columns:
                    fill_empty(ri_df, [field], '')
                    ri_df[field + '_valid'] = (ri_df[field] != '')

            del ri_info_no_fac

            profile_maps = [xref_df.copy() for i in range(ri_df['layer_id'].max())]
            for i, df in enumerate(profile_maps):
                df['layer_id'] = i + 1
            profile_map_df = pd.concat(profile_maps, ignore_index=True)

            logger.debug('Creating risk level and filter level profile IDs:')
            for fm_level_id in [RISK_LEVEL_ID, PROGRAM_LEVEL_ID]:
                for reins_type in FM_TERMS_PER_REINS_TYPE:
                    logger.debug(f'level_id {fm_level_id}, {reins_type} profiles...')
                    fm_profile_df = create_risk_level_profile_id(ri_df, profile_map_df, fm_profile_df, reins_type, risk_level, fm_level_id)

            ri_df['profile_id'] = ri_df['profile_id'].astype('int64')
            profile_map_df['profile_id'] = profile_map_df['profile_id'].astype('int64')
            fm_profile_df['profile_id'] = fm_profile_df['profile_id'].astype('int64')

            _log_dataframe(logger, {'ri_df': ri_df, 'profile_map_df': profile_map_df, 'fm_profile_df': fm_profile_df}, output_name)

            # create fm df
            fm_programme_df = xref_df[xref_df['agg_id_to'] != 0][['agg_id', 'level_id', 'agg_id_to']].reset_index(drop=True)
            fm_programme_df.columns = ['from_agg_id', 'level_id', 'to_agg_id']
            fm_profile_df = fm_profile_df.sort_values(by='profile_id').reset_index(drop=True)

            fm_policytc_df = profile_map_df[profile_map_df['level_id'] > 1][
                ['layer_id', 'level_id', 'agg_id', 'profile_id']
            ].reset_index(drop=True)
            fm_policytc_df['level_id'] = fm_policytc_df['level_id'] - 1
            # Net losses across all layers is associated to the max layer ID.
            fm_xref_df['layer_id'] = fm_policytc_df['layer_id'].max()

            _log_dataframe(logger, {'fm_programme_df': fm_programme_df, 'fm_profile_df': fm_profile_df,
                                    'fm_policytc_df': fm_policytc_df}, output_name)

            # Write out Oasis structure
            ri_output_dir = os.path.join(output_dir, "RI_{}".format(reinsurance_index))
            if os.path.exists(ri_output_dir):
                shutil.rmtree(ri_output_dir)
            os.makedirs(ri_output_dir)

            fm_programme_df.to_csv(
                os.path.join(ri_output_dir, "fm_programme.csv"), index=False)
            fm_profile_df.to_csv(
                os.path.join(ri_output_dir, "fm_profile.csv"), index=False)
            fm_policytc_df.to_csv(
                os.path.join(ri_output_dir, "fm_policytc.csv"), index=False)
            fm_xref_df.to_csv(
                os.path.join(ri_output_dir, "fm_xref.csv"), index=False)

            inuring_metadata[reinsurance_index] = {
                'inuring_priority': inuring_priority,
                'risk_level': risk_level,
                'directory': ri_output_dir
            }

            reinsurance_index = reinsurance_index + 1

    return inuring_metadata
