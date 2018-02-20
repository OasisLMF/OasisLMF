import json
import string
from argparse import Namespace
from tempfile import NamedTemporaryFile
from unittest import TestCase

import os
from hypothesis import given
from hypothesis.strategies import text

from oasislmf.cmd.base import InputValues
from oasislmf.utils.exceptions import OasisException


class InputValuesGet(TestCase):
    @given(text(min_size=1, max_size=10, alphabet=string.ascii_letters))
    def test_variable_is_on_the_command_line_but_not_in_config___command_line_is_used(self, cmd_var):
        with NamedTemporaryFile('w') as conf_file:
            json.dump({'bar': 'boo'}, conf_file)
            conf_file.flush()

            args = Namespace(foo=cmd_var, config=conf_file.name)

            inputs = InputValues(args)

            self.assertEqual(cmd_var, inputs.get('foo'))

    @given(
        text(min_size=1, max_size=10, alphabet=string.ascii_letters),
        text(min_size=1, max_size=10, alphabet=string.ascii_letters),
    )
    def test_variable_is_on_the_command_line_and_in_config___command_line_is_used(self, cmd_var, conf_var):
        with NamedTemporaryFile('w') as conf_file:
            json.dump({'foo': conf_var}, conf_file)
            conf_file.flush()

            args = Namespace(foo=cmd_var, config=conf_file.name)

            inputs = InputValues(args)

            self.assertEqual(cmd_var, inputs.get('foo'))

    @given(text(min_size=1, max_size=10, alphabet=string.ascii_letters))
    def test_variable_is_not_on_the_command_line_but_is_in_config___config_is_used(self, conf_var):
        with NamedTemporaryFile('w') as conf_file:
            json.dump({'foo': conf_var}, conf_file)
            conf_file.flush()

            args = Namespace(config=conf_file.name)

            inputs = InputValues(args)

            self.assertEqual(conf_var, inputs.get('foo'))

    def test_variable_is_not_on_the_command_line_or_config_var_is_not_required___none_is_returned(self):
        with NamedTemporaryFile('w') as conf_file:
            json.dump({'bar': 'boo'}, conf_file)
            conf_file.flush()

            args = Namespace(config=conf_file.name)

            inputs = InputValues(args)

            self.assertIsNone(inputs.get('foo'))

    @given(text(min_size=1, max_size=10, alphabet=string.ascii_letters))
    def test_variable_is_not_on_the_command_line_or_config_var_is_not_required_default_is_supplied___default_is_returned(self, default):
        with NamedTemporaryFile('w') as conf_file:
            json.dump({'bar': 'boo'}, conf_file)
            conf_file.flush()

            args = Namespace(config=conf_file.name)

            inputs = InputValues(args)

            self.assertEqual(default, inputs.get('foo', default=default))

    def test_variable_is_not_on_the_command_line_or_config_var_is_required___error_is_raised(self):
        with NamedTemporaryFile('w') as conf_file:
            json.dump({'bar': 'boo'}, conf_file)
            conf_file.flush()

            args = Namespace(config=conf_file.name)

            inputs = InputValues(args)

            with self.assertRaises(OasisException):
                inputs.get('foo', required=True)

    def test_variable_is_a_path___path_is_relative_to_config_file(self):
        with NamedTemporaryFile('w') as conf_file:
            json.dump({'foo': './some/path'}, conf_file)
            conf_file.flush()

            expected_result = os.path.abspath(
                os.path.join(os.path.dirname(conf_file.name), 'some', 'path')
            )

            args = Namespace(config=conf_file.name)

            inputs = InputValues(args)

            result = inputs.get('foo', required=True, is_path=True)

            self.assertEqual(expected_result, result)
