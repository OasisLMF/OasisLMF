"""Script to check default profiles are valid according to OEDSpec"""
from copy import copy
import argparse

from ods_tools.oed import OedSchema
from oasislmf.utils.data import DEFAULT_ADDITIONAL_FIELDS
from oasislmf.utils.defaults import get_default_accounts_profile, get_default_exposure_profile

error_t_string = '%(level)s - %(test_name)s: %(message_prefix)s%(error_message)s'


def get_oed_fields(oed_type, oed_version="latest version"):
    """Get the relevant OED fields from the spec for a given profile.
    """
    oed_schema = OedSchema.from_oed_schema_info(oed_version)
    schema_fields = oed_schema.schema['input_fields'][oed_type]
    return schema_fields


def print_errors(errors, ignore_warnings):
    error_errors = [e for e in errors if e['level'] == 'ERROR']

    for e in error_errors:
        print(error_t_string % e)

    if ignore_warnings:
        return

    warning_errors = [e for e in errors if e['level'] == 'WARNING']
    for e in warning_errors:
        print(error_t_string % e)

# Checks


def check_profile_fields_in_spec(profile, schema_fields, additional_fields=[], message_prefix=''):
    '''Check that the fields in `default_profile` are part of the OEDSpec.
    Throw an error if not as they will not pass the OED validation.
    '''
    mapped_schema_fields = OedSchema.column_to_field(list(profile.keys()), schema_fields)
    fields_only_in_profile = set(profile.keys()).difference(mapped_schema_fields.keys())

    if message_prefix:
        message_prefix = message_prefix + ' - '

    errors = []
    for field in fields_only_in_profile:
        if field not in additional_fields:
            errors.append({'level': 'ERROR', 'test_name': 'check_profile_fields_in_spec',
                           'message_prefix': message_prefix,
                           'error_message': f'{field} not found in OEDSpec.'})

    return errors


def check_spec_fields_in_profile(profile, all_schema_fields, message_prefix=''):
    '''Check that the fields in OEDSpec are in `default_profile`.
    Throw warning as a potential field to be included.
    '''
    schema_fields_in_profile = OedSchema.column_to_field(list(profile.keys()), all_schema_fields)

    remaining_schema_fields = copy(all_schema_fields)
    for profile_field in schema_fields_in_profile.values():
        remaining_schema_fields.pop(OedSchema.to_universal_field_name(profile_field['Input Field Name']))

    errors = []
    if message_prefix:
        message_prefix = message_prefix + ' - '
    for k, v in remaining_schema_fields.items():
        errors.append({
            'level': 'WARNING', 'test_name': 'check_spec_fields_in_profile',
            'message_prefix': message_prefix,
            'error_message': f'OEDSpec key {k} : {v["Input Field Name"]} is not in profile.'
        })
    return errors


def check_element_name_same_as_key(profile, message_prefix=''):
    errors = []
    if message_prefix:
        message_prefix = message_prefix + ' - '

    for k, v in profile.items():
        if k != v['ProfileElementName']:
            errors.append({
                'level': 'WARNING', 'test_name': 'check_element_name_same_as_key',
                'message_prefix': message_prefix,
                'error_message': f'Profile key {k} does not match ProfileElementName {v["ProfileElementName"]}.'
            })
    return errors


def main(oed_version="latest version", ignore_warnings=False,
         spec_in_profile=True,
         profile_in_spec=True,
         element_name_check=True):
    # Prepare inputs
    accounts_profile = get_default_accounts_profile()
    acc_schema_fields = get_oed_fields('Acc', oed_version=oed_version)
    acc_additional_fields = DEFAULT_ADDITIONAL_FIELDS['Acc'].keys()

    exposure_profile = get_default_exposure_profile()
    loc_schema_fields = get_oed_fields('Loc', oed_version=oed_version)
    loc_additional_fields = DEFAULT_ADDITIONAL_FIELDS['Loc'].keys()

    errors = []

    # Run checks
    if spec_in_profile:
        errors += check_spec_fields_in_profile(accounts_profile, acc_schema_fields, 'accounts_profile')
        errors += check_spec_fields_in_profile(exposure_profile, loc_schema_fields, 'exposure_profile')

    if profile_in_spec:
        errors += check_profile_fields_in_spec(accounts_profile,
                                               acc_schema_fields,
                                               acc_additional_fields,
                                               'accounts_profile')
        errors += check_profile_fields_in_spec(exposure_profile,
                                               loc_schema_fields,
                                               loc_additional_fields,
                                               'exposure_profile')

    if element_name_check:
        errors += check_element_name_same_as_key(accounts_profile, 'accounts_profile')
        errors += check_element_name_same_as_key(exposure_profile, 'exposure_profile')

    print_errors(errors, ignore_warnings)
    return errors


parser = argparse.ArgumentParser(prog='check_default_profile',
                                 description='Validate default profiles against OEDSpec')
parser.add_argument('-s', '--oed-version', default="latest version", help="Version of OEDSpec to use.")
parser.add_argument('-w', '--ignore-warnings', action="store_false", help="Do not show WARNING messages")
parser.add_argument('--profile-in-spec', action="store_false", help="Check if all profile fields in spec, error level ERROR.")
parser.add_argument('--spec-in-profile', action="store_false", help="Check if all spec fields are in profile, error level WARNING.")
parser.add_argument('--element-name-check', action="store_false", help="Check profile key matches ProfileElementName, error level WARNING.")

args = vars(parser.parse_args())
print(args)
main(**args)
