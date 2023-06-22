class ExposurePreAnalysis:
    """
    Fake exposure pre-analysis module.
    """

    def __init__(self, exposure_data, exposure_pre_analysis_setting, **kwargs):
        self.exposure_data = exposure_data
        self.exposure_pre_analysis_setting = exposure_pre_analysis_setting

    def run(self):
        loc_df = self.exposure_data.location.dataframe
        acc_df = self.exposure_data.account.dataframe

        loc_df['LocNumber'] = self.exposure_pre_analysis_setting['override_loc_num']
        acc_df['AccNumber'] = self.exposure_pre_analysis_setting['override_acc_num']
