import os
from tempfile import NamedTemporaryFile
from unittest import TestCase

from mock import patch, Mock

from oasislmf.utils.diff import unified_diff
from oasislmf.utils.exceptions import OasisException


class UnifiedDiff(TestCase):
    def test_opening_file_raises_os_error___oasis_exception_is_raised(self):
        def raising_function(*args, **kwargs):
            raise OSError()

        with patch('io.open', Mock(side_effect=raising_function)), self.assertRaises(OasisException):
            unified_diff('first', 'second')

    def test_opening_file_raises_io_error___oasis_exception_is_raised(self):
        def raising_function(*args, **kwargs):
            raise IOError()

        with patch('io.open', Mock(side_effect=raising_function)), self.assertRaises(OasisException):
            unified_diff('first', 'second')

    def test_two_files_are_the_same___result_is_empty(self):
        f = NamedTemporaryFile(mode='w', delete=False)
        try:
            f.write('content')
            f.close()

            diff = list(unified_diff(f.name, f.name))

            self.assertEqual(0, len(diff))
        finally:
            os.remove(f.name)

    def test_two_files_are_different___result_is_list_of_differences(self):
        first = NamedTemporaryFile(mode='w', delete=False)
        second = NamedTemporaryFile(mode='w', delete=False)
        try:
            first.writelines([
                'HEADING\n',
                'first\n',
                'same\n',
                'second\n',
                'FOOTER\n'
            ])
            first.close()

            second.writelines([
                'HEADING\n',
                'third\n',
                'same\n',
                'fourth\n',
                'FOOTER\n'
            ])
            second.close()

            diff = list(unified_diff(first.name, second.name))

            self.assertEqual(diff, [
                '--- {}\n'.format(first.name),
                '+++ {}\n'.format(second.name),
                '@@ -1,5 +1,5 @@\n',
                ' HEADING\n',
                '-first\n',
                '+third\n',
                ' same\n',
                '-second\n',
                '+fourth\n',
                ' FOOTER\n',
            ])
        finally:
            os.remove(first.name)
            os.remove(second.name)

    def test_two_files_are_different_as_string_is_true___result_is_concatenated_list_of_differences(self):
        first = NamedTemporaryFile(mode='w', delete=False)
        second = NamedTemporaryFile(mode='w', delete=False)
        try:
            first.writelines([
                'HEADING\n',
                'first\n',
                'same\n',
                'second\n',
                'FOOTER\n'
            ])
            first.close()

            second.writelines([
                'HEADING\n',
                'third\n',
                'same\n',
                'fourth\n',
                'FOOTER\n'
            ])
            second.close()

            diff = unified_diff(first.name, second.name, as_string=True)

            self.assertEqual(diff, ''.join([
                '--- {}\n'.format(first.name),
                '+++ {}\n'.format(second.name),
                '@@ -1,5 +1,5 @@\n',
                ' HEADING\n',
                '-first\n',
                '+third\n',
                ' same\n',
                '-second\n',
                '+fourth\n',
                ' FOOTER\n',
            ]))
        finally:
            os.remove(first.name)
            os.remove(second.name)
