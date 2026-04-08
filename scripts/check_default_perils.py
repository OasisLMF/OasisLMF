"""Script to check default profiles are valid according to OEDSpec"""
from copy import copy
import argparse
from ods_tools.oed import OedSchema, oed_schema
from oasislmf.utils.defaults import get_default_peril_groups, get_default_perils


def get_oed_peril_info(oed_type, oed_version="latest version"):
    """Get the relevant OED peril info.
    """
    oed_schema = OedSchema.from_oed_schema_info(oed_version)


error_t_string = '%(level)s - %(test_name)s: %(error_message)s'


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


def check_all_perils_included(peril_dct, all_perils_dct, all_perils_name='peril codes'):
    '''
    Check `peril` dict keys are in  all_perils.
    '''
    all_perils = set(all_perils_dct.keys())
    errors = []
    for p_code, p in peril_dct.items():
        if p_code not in all_perils:
            errors.append({
                'level': 'ERROR',
                'test_name': 'check_all_perils_included',
                'error_message': f'{p_code} not found in {all_perils_name}.'
            })

    return errors


def check_all_default_peril_groups_correct(default_peril_groups, spec_peril_covered):
    errors = []
    for peril_group in default_peril_groups.values():
        perils_covered = spec_peril_covered.get(peril_group['peril_code'], None)
        if perils_covered is None:
            errors.append({
                'level': 'ERROR',
                'test_name': 'check_all_default_peril_groups_correct',
                'error_message': f'{peril_group["peril_code"]} peril group not found in OEDSpec perils covered.'
            })
            continue
        if set(perils_covered) != set(peril_group['peril_ids']):
            perils_in_default = set(perils_covered).difference(peril_group['peril_ids'])
            perils_in_spec = set(peril_group['peril_ids']).difference(perils_covered)

            error_template = {
                'level': 'ERROR',
                'test_name': 'check_all_default_peril_groups_correct',
            }

            errors.append(error_template | {'error_message': f'{peril_group["peril_code"]} peril group does not match OEDSpec perils covered.'})

            if perils_in_default:
                errors.append(error_template | {'error_message': f'Missing perils: {perils_in_default}'})
            if perils_in_spec:
                errors.append(error_template | {'error_message': f'Extra perils: {perils_in_spec}'})
    return errors


def main(oed_version="latest version", ignore_warnings=False):
    # Prepare inputs
    perils = get_default_perils()
    peril_groups = get_default_peril_groups()
    combined_perils_default = {p['peril_code']: p for p in perils.values()}
    combined_perils_default |= {p['peril_code']: p for p in peril_groups.values()}

    oed_schema = OedSchema.from_oed_schema_info(oed_version)
    oed_peril_info = oed_schema.schema['perils']['info']
    oed_peril_groups = oed_schema.schema['perils']['covered']
    errors = []

    # Run checks
    errors += check_all_perils_included(combined_perils_default,
                                        oed_peril_info, 'schema peril codes')
    errors += check_all_perils_included(oed_peril_info,
                                        combined_perils_default, 'LMF peril codes')
    errors += check_all_default_peril_groups_correct(peril_groups, oed_peril_groups)

    print_errors(errors, ignore_warnings)
    return errors


parser = argparse.ArgumentParser(prog='check_default_profile',
                                 description='Validate default profiles against OEDSpec')
parser.add_argument('-s', '--oed-version', default="latest version", help="Version of OEDSpec to use.")
parser.add_argument('-w', '--ignore-warnings', action="store_false", help="Do not show WARNING messages")

args = vars(parser.parse_args())
main(**args)
