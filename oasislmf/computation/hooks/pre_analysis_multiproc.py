__all__ = [
    'run_pre_analysis_multiproc',
]

import copy

import numpy as np
import pandas as pd

from ...utils.exceptions import OasisException
from ...utils.multiproc import run_multiproc


def _split_chunks(loc_df, acc_df, part_count, group_cols):
    """Yield (loc_part, acc_part) chunks split from loc_df/acc_df.

    If group_cols is given, chunks are formed from unique combinations of those columns
    (e.g. ['PortNumber', 'AccNumber']) so that every location/account row belonging to the
    same group ends up in the same chunk. Otherwise (no account grouping possible), chunks
    are formed from unique 'loc_id' values, matching oasislmf.lookup.factory's location split.
    """
    if group_cols:
        if acc_df is not None:
            # union with acc_df's groups too - an account with no matching location rows (e.g.
            # a genuinely locationless account, or leftover after filtering location/
            # location_numbers without also filtering account/account_numbers) still needs a
            # group of its own, otherwise it's never assigned to any chunk and silently dropped.
            group_keys = pd.concat(
                [loc_df[group_cols], acc_df[group_cols]], ignore_index=True
            ).drop_duplicates().reset_index(drop=True)
        else:
            group_keys = loc_df[group_cols].drop_duplicates().reset_index(drop=True)
        key_parts = [group_keys.iloc[idx] for idx in np.array_split(np.arange(len(group_keys)), part_count)]
        loc_parts = (loc_df.merge(key_part, on=group_cols) for key_part in key_parts)
        acc_parts = (
            (acc_df.merge(key_part, on=group_cols) if acc_df is not None else None)
            for key_part in key_parts
        )
    else:
        loc_id_parts = np.array_split(np.unique(loc_df['loc_id']), part_count)
        loc_parts = (loc_df[loc_df['loc_id'].isin(loc_id_parts[i])] for i in range(part_count))
        acc_parts = (None for _ in range(part_count))

    return zip(loc_parts, acc_parts)


def _make_chunk_processor(exposure_data, hook_cls, hook_kwargs):
    """Build the per-chunk function run in each worker process: instantiate hook_cls on a
    chunk of exposure_data's location/account dataframes and call .run(), returning the
    (possibly hook-modified) location/account dataframes plus the hook's return value.

    Raises OasisException if the hook mutates exposure_data.ri_info/ri_scope - these aren't
    chunked or merged back (only the main process's copy is kept), so silently allowing such a
    mutation would discard it without warning.
    """
    has_account = exposure_data.account is not None
    unchunked_ri_info = exposure_data.ri_info.dataframe.copy() if exposure_data.ri_info is not None else None
    unchunked_ri_scope = exposure_data.ri_scope.dataframe.copy() if exposure_data.ri_scope is not None else None

    def process_chunk(chunk):
        loc_part, acc_part = chunk

        chunk_exposure_data = copy.copy(exposure_data)
        chunk_exposure_data.location = copy.copy(exposure_data.location)
        chunk_exposure_data.location.dataframe = loc_part
        if has_account:
            chunk_exposure_data.account = copy.copy(exposure_data.account)
            chunk_exposure_data.account.dataframe = acc_part

        chunk_kwargs = dict(hook_kwargs)
        chunk_kwargs['exposure_data'] = chunk_exposure_data
        class_return = hook_cls(**chunk_kwargs).run()

        if unchunked_ri_info is not None and not chunk_exposure_data.ri_info.dataframe.equals(unchunked_ri_info):
            raise OasisException(
                'ExposurePreAnalysis hook modified exposure_data.ri_info, which is not supported '
                'when pre-analysis multiprocessing is enabled - ri_info is not chunked or merged '
                'back, so the change would otherwise be silently discarded. Set '
                'lookup_multiprocessing=False to run this hook single-process.'
            )
        if unchunked_ri_scope is not None and not chunk_exposure_data.ri_scope.dataframe.equals(unchunked_ri_scope):
            raise OasisException(
                'ExposurePreAnalysis hook modified exposure_data.ri_scope, which is not supported '
                'when pre-analysis multiprocessing is enabled - ri_scope is not chunked or merged '
                'back, so the change would otherwise be silently discarded. Set '
                'lookup_multiprocessing=False to run this hook single-process.'
            )

        return (
            chunk_exposure_data.location.dataframe,
            chunk_exposure_data.account.dataframe if has_account else None,
            class_return,
        )

    return process_chunk


def run_pre_analysis_multiproc(exposure_data, hook_cls, hook_kwargs, pool_count, part_count, group_cols):
    """Run hook_cls(**hook_kwargs, exposure_data=<chunk>).run() across pool_count worker
    processes, splitting exposure_data's location/account dataframes into part_count chunks.

    Returns (location_df, account_df, [chunk_return, ...]) with the per-chunk results merged
    back together; account_df is None if exposure_data has no account data.
    """
    loc_df = exposure_data.location.dataframe
    acc_df = exposure_data.account.dataframe if exposure_data.account is not None else None
    chunks = _split_chunks(loc_df, acc_df, part_count, group_cols)
    # process_chunk doesn't depend on which worker runs it, so every worker gets the same one
    process_chunk = _make_chunk_processor(exposure_data, hook_cls, hook_kwargs)

    def on_results(results):
        loc_results, acc_results, class_returns = [], [], []
        for loc_part, acc_part, class_return in results:
            loc_results.append(loc_part)
            if acc_part is not None:
                acc_results.append(acc_part)
            class_returns.append(class_return)

        return (
            pd.concat(loc_results, ignore_index=True),
            pd.concat(acc_results, ignore_index=True) if acc_results else None,
            class_returns,
        )

    return run_multiproc(chunks, lambda worker_id: process_chunk, pool_count, on_results)
