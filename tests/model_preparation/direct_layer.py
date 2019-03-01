import os
import subprocess32 as subprocess
import shutil

import pandas as pd

from oasislmf.model_preparation import oed


class DirectLayer(object):
    """
    Set of direct policiies.
    Generates ktools inputs and runs financial module.

    Does not handle multiple policies on same set of risks i.e. multiple layers.

    NB. This is a simple implication to support the reinsurance Oasis file generation.
        A full implementation supporting OED account and location files will be developed
        and included in the OasisLmf package separately allowing this code to be deprecated.
    """

    def __init__(self, accounts, locations):
        self.accounts = accounts
        self.locations = locations

        self.item_ids = list()
        self.item_tivs = list()
        self.coverages = pd.DataFrame()
        self.items = pd.DataFrame()
        self.fmprogrammes = pd.DataFrame()
        self.fmprofiles = pd.DataFrame()
        self.fm_policytcs = pd.DataFrame()
        self.fm_xrefs = pd.DataFrame()
        self.xref_descriptions = pd.DataFrame()

        self.item_id_dict = dict()

    def _get_location_tiv(self, location, coverage_type_id):
        switcher = {
            oed.BUILDING_COVERAGE_TYPE_ID: location.BuildingTIV,
            oed.OTHER_BUILDING_COVERAGE_TYPE_ID: location.OtherTIV,
            oed.CONTENTS_COVERAGE_TYPE_ID: location.ContentsTIV,
            oed.TIME_COVERAGE_TYPE_ID: location.BITIV
        }
        return switcher.get(coverage_type_id, 0)

    def generate_oasis_structures(self):

        coverage_id = 0
        item_id = 0
        group_id = 0
        policy_agg_id = 0
        profile_id = 0

        coverages_list = list()
        items_list = list()
        fmprogrammes_list = list()
        fmprofiles_list = list()
        fm_policytcs_list = list()
        fm_xrefs_list = list()
        xref_descriptions_list = list()

        site_agg_id = 0
        for policy_index, policy in self.accounts.iterrows():
            policy_agg_id = policy_agg_id + 1
            profile_id = profile_id + 1
            fmprofiles_list.append(
                oed.get_profile(
                    profile_id,
                    deductible=policy.AccDed6All,
                    limit=policy.AccLimit6All))
            fm_policytcs_list.append(oed.FmPolicyTc(
                layer_id=1,
                level_id=2,
                agg_id=policy_agg_id,
                profile_id=profile_id
            ))
            for location_index, location in self.locations.loc[self.locations["AccNumber"] == policy.AccNumber].iterrows():
                group_id = group_id + 1
                site_agg_id = site_agg_id + 1
                profile_id = profile_id + 1

                fmprofiles_list.append(
                    oed.get_profile(
                        profile_id=profile_id,
                        deductible=location.LocDed6All,
                        limit=location.LocLimit6All))
                fm_policytcs_list.append(oed.FmPolicyTc(
                    layer_id=1,
                    level_id=1,
                    agg_id=site_agg_id,
                    profile_id=profile_id
                ))
                fmprogrammes_list.append(
                    oed.FmProgramme(
                        from_agg_id=site_agg_id,
                        level_id=2,
                        to_agg_id=policy_agg_id
                    )
                )

                for coverage_type_id in oed.COVERAGE_TYPES:
                    tiv = self._get_location_tiv(location, coverage_type_id)
                    if tiv > 0:
                        coverage_id = coverage_id + 1
                        self.item_id_dict[coverage_id] = location

                        coverages_list.append(
                            oed.Coverage(
                                coverage_id=coverage_id,
                                tiv=tiv,
                            ))
                        for peril in oed.PERILS:
                            item_id = item_id + 1
                            self.item_ids.append(item_id)
                            self.item_tivs.append(tiv)
                            items_list.append(
                                oed.Item(
                                    item_id=item_id,
                                    coverage_id=coverage_id,
                                    areaperil_id=-1,
                                    vulnerability_id=-1,
                                    group_id=group_id
                                ))
                            fmprogrammes_list.append(
                                oed.FmProgramme(
                                    from_agg_id=item_id,
                                    level_id=1,
                                    to_agg_id=site_agg_id
                                )
                            )
                            fm_xrefs_list.append(
                                oed.FmXref(
                                    output_id=item_id,
                                    agg_id=item_id,
                                    layer_id=1
                                ))
                            xref_descriptions_list.append(
                                oed.XrefDescription(
                                    xref_id=item_id,
                                    account_number=location.AccNumber,
                                    location_number=location.LocNumber,
                                    location_group=location.LocGroup,
                                    cedant_name = policy.CedantName,
                                    producer_name = policy.ProducerName,
                                    lob = policy.LOB,
                                    country_code = location.CountryCode,
                                    reins_tag = location.ReinsTag,
                                    coverage_type_id=coverage_type_id,
                                    peril_id=peril,
                                    policy_number=policy.PolNumber,    
                                    portfolio_number=policy.PortNumber,
                                    tiv=tiv
                                )
                            )

        self.coverages = pd.DataFrame(coverages_list)
        self.items = pd.DataFrame(items_list)
        self.fmprogrammes = pd.DataFrame(fmprogrammes_list)
        self.fmprofiles = pd.DataFrame(fmprofiles_list)
        self.fm_policytcs = pd.DataFrame(fm_policytcs_list)
        self.fm_xrefs = pd.DataFrame(fm_xrefs_list)
        self.xref_descriptions = pd.DataFrame(xref_descriptions_list)
        pass

    def write_oasis_files(self, directory=None):
        self.coverages.to_csv("coverages.csv", index=False)
        self.items.to_csv("items.csv", index=False)
        self.fmprogrammes.to_csv("fm_programme.csv", index=False)
        self.fmprofiles.to_csv("fm_profile.csv", index=False)
        self.fm_policytcs.to_csv("fm_policytc.csv", index=False)
        self.fm_xrefs.to_csv("fm_xref.csv", index=False)

        if directory is None:
            directory = "direct"
        else:
            directory = os.path.join(directory, "direct")

        if os.path.exists(directory):
            shutil.rmtree(directory)
        os.makedirs(directory)

        input_files = oed.GUL_INPUTS_FILES + oed.IL_INPUTS_FILES

        for input_file in input_files:
            conversion_tool = oed.CONVERSION_TOOLS[input_file]
            input_file_path = input_file + ".csv"
            if not os.path.exists(input_file_path):
                continue

            output_file_path = os.path.join(directory, input_file + ".bin")
            command = "{} < {} > {}".format(
                conversion_tool, input_file_path, output_file_path)
            proc = subprocess.Popen(command, shell=True)
            proc.wait()
            if proc.returncode != 0:
                raise Exception(
                    "Failed to convert {}: {}".format(input_file_path, command))


    def report_item_ids(self):
        """
        return a dataframe showing the relationship between item_id's and Locations
        """
        locations_list  = [self.item_id_dict[ID].LocNumber for ID in self.item_ids] 
        from_agg_ids = self.fmprogrammes[self.fmprogrammes['level_id'] == 1].from_agg_id.tolist()
        item_map_df = pd.concat([
            self.items[['item_id','coverage_id']],
            self.coverages['tiv'],
            pd.DataFrame({'LocNumber': locations_list})
        ],axis=1)  
        # filter 'item_id' that exisit in 'from_agg_id'
        return item_map_df[item_map_df['item_id'].isin(from_agg_ids)]

    def apply_fm(self, loss_percentage_of_tiv=1.0, net=False):
        guls_list = list()
        for item_id, tiv in zip(self.item_ids, self.item_tivs):
            event_loss = loss_percentage_of_tiv * tiv
            guls_list.append(
                oed.GulRecord(event_id=1, item_id=item_id, sidx=-1, loss=event_loss))
            guls_list.append(
                oed.GulRecord(event_id=1, item_id=item_id, sidx=-2, loss=0))
            guls_list.append(
                oed.GulRecord(event_id=1, item_id=item_id, sidx=1, loss=event_loss))
        guls_df = pd.DataFrame(guls_list)
        guls_df.to_csv("guls.csv", index=False)
        net_flag = ""
        if net:
            net_flag = "-n"
        command = "gultobin -S 1 < guls.csv | fmcalc -p direct {} -a {} | tee ils.bin | fmtocsv > ils.csv".format(
            net_flag, oed.ALLOCATE_TO_ITEMS_BY_PREVIOUS_LEVEL_ALLOC_ID)
        proc = subprocess.Popen(command, shell=True)
        proc.wait()
        if proc.returncode != 0:
            raise Exception("Failed to run fm")
        losses_df = pd.read_csv("ils.csv")
        losses_df.drop(losses_df[losses_df.sidx != 1].index, inplace=True)
        del losses_df['sidx']
        guls_df.drop(guls_df[guls_df.sidx != 1].index, inplace=True)
        del guls_df['event_id']
        del guls_df['sidx']
        guls_df = pd.merge(
            self.xref_descriptions,
            guls_df, left_on=['xref_id'], right_on=['item_id'])
        losses_df = pd.merge(
            guls_df,
            losses_df, left_on='xref_id', right_on='output_id',
            suffixes=["_gul", "_il"])
        del losses_df['event_id']
        del losses_df['output_id']
        del losses_df['xref_id']
        del losses_df['item_id']

        return losses_df
