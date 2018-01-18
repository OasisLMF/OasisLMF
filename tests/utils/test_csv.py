from tempfile import NamedTemporaryFile
from unittest import TestCase

from oasislmf.utils.csv import get_csv_rows_as_dicts


class GetCsvRowsAsDicts(TestCase):
    def test_file_has_no_rows___result_is_empty(self):
        with NamedTemporaryFile('w') as f:
            f.writelines([
                'first,second\n',
            ])
            f.flush()

            self.assertEqual(0, len(list(get_csv_rows_as_dicts(f.name))))

    def test_no_meta_is_given___base_csv_data_is_given(self):
        with NamedTemporaryFile('w') as f:
            f.writelines([
                'first,second\n',
                'a,b\n',
                'c,d\n',
            ])
            f.flush()

            data = list(get_csv_rows_as_dicts(f.name))

            self.assertEqual(
                data, [
                    {
                        'first': 'a',
                        'second': 'b',
                    },
                    {
                        'first': 'c',
                        'second': 'd',
                    }
                ]
            )

    def test_no_meta_is_given___data_is_transformed_by_the_meta_functions(self):
        with NamedTemporaryFile('w') as f:
            f.writelines([
                'first,second\n',
                'a,1\n',
                'c,2\n',
            ])
            f.flush()

            data = list(get_csv_rows_as_dicts(f.name, csv_field_meta={
                'FIRST': {'validator': str.upper, 'csv_header': 'first'},
                'SECOND': {'validator': int, 'csv_header': 'second'},
            }))

            self.assertEqual(
                data, [
                    {
                        'FIRST': 'A',
                        'SECOND': 1,
                    },
                    {
                        'FIRST': 'C',
                        'SECOND': 2,
                    }
                ]
            )
