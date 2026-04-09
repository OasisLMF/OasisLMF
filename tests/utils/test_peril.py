from mock import patch

from oasislmf.utils.peril import get_peril_info_from_schema


@patch("oasislmf.utils.peril.OedSchema")
def test_get_peril_info_from_schema(mock_oed_schema):
    schema = {
        'perils': {
            'info': {
                'ALL': {'id': 'ALL',
                        'Peril Description': 'All Perils',
                        'Grouped PerilCode': 'Yes'
                        },
                'PERIL1': {'id': 'PERIL1',
                           'Peril Description': 'Peril 1',
                           'Grouped PerilCode': 'No'
                           },
                'PERIL2': {'id': 'PERIL2',
                           'Peril Description': 'Peril 2',
                           'Grouped PerilCode': 'No'
                           }
            },
            'covered': {
                'ALL': ['PERIL1', 'PERIL2']
            }
        }
    }

    mock_oed_schema.from_oed_schema_info.return_value.schema = schema
    perils, peril_groups = get_peril_info_from_schema()

    expected_perils = {'PERIL1': {'id': 'PERIL1', 'desc': 'Peril 1'},
                       'PERIL2': {'id': 'PERIL2', 'desc': 'Peril 2'}}
    expected_group_perils = {'ALL': {'id': 'ALL', 'desc': 'All Perils',
                                     'peril_ids': ['PERIL1', 'PERIL2']}}

    assert perils == expected_perils
    assert expected_group_perils == expected_group_perils
