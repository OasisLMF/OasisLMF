"""Script to check default profiles are valid according to OEDSpec"""
import warnings
import pytest

from ods_tools.oed import OedSchema
from oasislmf.utils.data import DEFAULT_ADDITIONAL_FIELDS
from oasislmf.utils.defaults import get_default_accounts_profile, get_default_exposure_profile


def get_oed_fields(oed_type, oed_version="latest version"):
    oed_schema = OedSchema.from_oed_schema_info(oed_version)
    schema_fields = oed_schema.schema['input_fields'][oed_type]
    return schema_fields


def get_cyber_tiv_fields(oed_version="latest version"):
    oed_coverages = OedSchema.from_oed_schema_info(oed_version).schema['CoverageValues']
    return [f'Cyber{k}TIV' for k, v in oed_coverages.items() if v['Type'] == 'Cyber']


def get_additional_fields(oed_type):
    additional_fields = []

    # AnnualRevenue for Cyber in Loc
    if oed_type == 'Loc':
        additional_fields += ['AnnualRevenue']
    return additional_fields + list(DEFAULT_ADDITIONAL_FIELDS[oed_type].keys())


@pytest.fixture
def acc_profile():
    return get_default_accounts_profile()


@pytest.fixture
def exposure_profile():
    return get_default_exposure_profile()


@pytest.mark.parametrize("profile,message_prefix",
                         [("acc_profile", "accounts_profile"),
                          ("exposure_profile", "exposure_profile")])
def test_PorfileElementName_same_as_key(profile, message_prefix, request):
    profile = request.getfixturevalue(profile)

    # Ignore the CyberTIV fields
    cyber_tiv_fields = get_cyber_tiv_fields()
    keys = [k for k in profile.keys() if k not in cyber_tiv_fields]

    profile_element_values = [profile[k]['ProfileElementName'] for k in keys]
    assert keys == profile_element_values


@pytest.mark.parametrize("profile,oed_type,profile_name",
                         [("acc_profile", 'Acc', 'accounts_profile'),
                          ("exposure_profile", 'Loc', 'exposure_profile')])
def test_profile_fields_in_spec(profile, oed_type, profile_name, request):
    profile = request.getfixturevalue(profile)
    profile_cols = [v['ProfileElementName'] for v in profile.values()]
    schema_fields = get_oed_fields(oed_type)

    mapped_schema_fields = OedSchema.column_to_field(profile_cols, schema_fields)

    # LMF additional fields
    additional_fields = get_additional_fields(oed_type)

    valid_schema_fields = set(mapped_schema_fields.keys()).union(additional_fields)
    fields_only_in_profile = set(profile_cols).difference(valid_schema_fields)

    assert len(fields_only_in_profile) == 0, f'Fields in profile are not valid OED Fields: {fields_only_in_profile}'
