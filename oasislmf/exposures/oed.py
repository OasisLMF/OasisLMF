import pandas as pd
import os
import common


# TODO - add validator 

def load_oed_dfs(oed_dir, show_all=False):
    """
    Load OED data files.
    """

    do_reinsurance = True
    if oed_dir is not None:
        if not os.path.exists(oed_dir):
            print("Path does not exist: {}".format(oed_dir))
            exit(1)
        # Account file
        oed_account_file = os.path.join(oed_dir, "account.csv")
        if not os.path.exists(oed_account_file):
            print("Path does not exist: {}".format(oed_account_file))
            exit(1)
        account_df = pd.read_csv(oed_account_file)

        # Location file
        oed_location_file = os.path.join(oed_dir, "location.csv")
        if not os.path.exists(oed_location_file):
            print("Path does not exist: {}".format(oed_location_file))
            exit(1)
        location_df = pd.read_csv(oed_location_file)

        # RI files
        oed_ri_info_file = os.path.join(oed_dir, "ri_info.csv")
        oed_ri_scope_file = os.path.join(oed_dir, "ri_scope.csv")
        oed_ri_info_file_exists = os.path.exists(oed_ri_info_file)
        oed_ri_scope_file_exists = os.path.exists(oed_ri_scope_file)

        if not oed_ri_info_file_exists and not oed_ri_scope_file_exists:
            ri_info_df = None
            ri_scope_df = None
            do_reinsurance = False
        elif oed_ri_info_file_exists and oed_ri_scope_file_exists:
            ri_info_df = pd.read_csv(oed_ri_info_file)
            ri_scope_df = pd.read_csv(oed_ri_scope_file)
        else:
            print("Both reinsurance files must exist: {} {}".format(
                oed_ri_info_file, oed_ri_scope_file))
        if not show_all:
            account_df = account_df[common.OED_ACCOUNT_FIELDS].copy()
            location_df = location_df[common.OED_LOCATION_FIELDS].copy()
            if do_reinsurance:
                ri_info_df = ri_info_df[common.OED_REINS_INFO_FIELDS].copy()
                ri_scope_df = ri_scope_df[common.OED_REINS_SCOPE_FIELDS].copy()
    return (account_df, location_df, ri_info_df, ri_scope_df, do_reinsurance)


