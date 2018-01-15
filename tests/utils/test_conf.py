import socket
from unittest import TestCase

from tempfile import NamedTemporaryFile

from oasislmf.utils.conf import load_ini_file


class LoadInIFile(TestCase):
    def test_values_are_bool___values_are_correctly_converted_to_bool_value(self):
        with NamedTemporaryFile(mode='w') as f:
            f.writelines([
                '[section]\n',
                'a = True\n',
                'b = False\n',
            ])
            f.flush()

            conf = load_ini_file(f.name)

            self.assertTrue(conf['a'])
            self.assertFalse(conf['b'])

    def test_values_are_int___values_are_correctly_converted_to_int_value(self):
        with NamedTemporaryFile(mode='w') as f:
            f.writelines([
                '[section]\n',
                'a = 1\n',
                'b = 2\n',
            ])
            f.flush()

            conf = load_ini_file(f.name)

            self.assertEqual(1, conf['a'])
            self.assertEqual(2, conf['b'])

    def test_values_are_float___value_are_correctly_converted_to_int_value(self):
        with NamedTemporaryFile(mode='w') as f:
            f.writelines([
                '[section]\n',
                'a = 1.1\n',
                'b = 2.2\n',
            ])
            f.flush()

            conf = load_ini_file(f.name)

            self.assertEqual(1.1, conf['a'])
            self.assertEqual(2.2, conf['b'])

    def test_values_are_ip_addresses___values_are_converted_into_byte_format(self):
        with NamedTemporaryFile(mode='w') as f:
            f.writelines([
                '[section]\n',
                'a = 127.0.0.1\n',
                'b = 127.127.127.127\n',
            ])
            f.flush()

            conf = load_ini_file(f.name)

            self.assertEqual(socket.inet_aton('127.0.0.1'), conf['a'])
            self.assertEqual(socket.inet_aton('127.127.127.127'), conf['b'])

    def test_values_are_string_values___values_are_unchanged(self):
        with NamedTemporaryFile(mode='w') as f:
            f.writelines([
                '[section]\n',
                'a = first.value\n',
                'b = another value\n',
            ])
            f.flush()

            conf = load_ini_file(f.name)

            self.assertEqual('first.value', conf['a'])
            self.assertEqual('another value', conf['b'])
