import itertools
import logging
import json

from oasislmf.utils import (
    coverages,
    peril,
)
from oasislmf.utils.status import (
    OASIS_KEYS_SC,
    OASIS_KEYS_FL,
    OASIS_KEYS_NM,
    OASIS_KEYS_STATUS
)
from oasislmf.preparation.lookup import OasisBaseKeysLookup


class FakeComplexModelKeysLookup(OasisBaseKeysLookup):

    def __init__(self,
                 keys_data_directory=None,
                 supplier=None,
                 model_name=None,
                 model_version=None,
                 **kwargs):

        self._peril_ids = [
            peril.PERILS['tropical cyclone']['id'],
            peril.PERILS['storm surge']['id']
        ]

        self._coverage_types = [
            coverages.COVERAGE_TYPES['buildings']['id'],
            coverages.COVERAGE_TYPES['contents']['id']
        ]

    def process_location(self, loc, peril_id, coverage_type):

        status = OASIS_KEYS_SC
        message = "OK"

        if (
            peril_id == peril.PERILS['tropical cyclone']['id'] and
            coverage_type == coverages.COVERAGE_TYPES['buildings']['id']
        ):
            data = {
                "area_peril_id": 54,
                "vulnerability_id": 2
            }

        elif (
            peril_id == peril.PERILS['tropical cyclone']['id'] and
            coverage_type == coverages.COVERAGE_TYPES['contents']['id']
        ):
            data = {
                "area_peril_id": 54,
                "vulnerability_id": 5
            }

        elif (
            peril_id == peril.PERILS['storm surge']['id'] and
            coverage_type == coverages.COVERAGE_TYPES['buildings']['id']
        ):
            data = {
                "area_peril_id": 154,
                "vulnerability_id": 8
            }

        elif (
            peril_id == peril.PERILS['storm surge']['id'] and
            coverage_type == coverages.COVERAGE_TYPES['contents']['id']
        ):
            data = {
                "area_peril_id": 154,
                "vulnerability_id": 11
            }

        return {
            'loc_id': loc['loc_id'],
            'locnumber': loc['locnumber'],
            'peril_id': peril_id,
            'coverage_type': coverage_type,
            'areaperil_id': 154,
            'vulnerability_id': 11,
            'model_data': json.dumps(data),
            'status': status,
            'message': message
        }

    def process_locations(self, loc_df):
        logger = logging.getLogger(__name__)
        logger.info("Log Message From FakeComplexModelKeysLookup")

        loc_df = loc_df.rename(columns=str.lower)

        locs_seq = (loc for _, loc in loc_df.iterrows())
        for loc, peril_id, coverage_type in \
                itertools.product(locs_seq, self._peril_ids, self._coverage_types):
            yield self.process_location(loc, peril_id, coverage_type)
