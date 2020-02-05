import json
import string
from argparse import Namespace
from tempfile import NamedTemporaryFile
from unittest import TestCase

import os
from hypothesis import given
from hypothesis.strategies import (
    floats,
    text
)

from oasislmf.cli.inputs import InputValues
from oasislmf.utils.exceptions import OasisException


class InputValuesGet(TestCase):
    @given(text(min_size=1, max_size=10, alphabet=string.ascii_letters))
    def test_variable_is_on_the_command_line_but_not_in_config___command_line_is_used(self, cmd_var):
        conf_file = NamedTemporaryFile('w', delete=False)
        try:
            json.dump({'bar': 'boo'}, conf_file)
            conf_file.close()

            args = Namespace(foo=cmd_var, config=conf_file.name)

            inputs = InputValues(args)

            self.assertEqual(cmd_var, inputs.get('foo'))
        finally:
            os.remove(conf_file.name)

    def test_boolean_variable_with_command_default_null_is_on_the_command_line_as_false_but_not_in_config___command_line_false_value_is_picked_up(self):
        conf_file = NamedTemporaryFile('w', delete=False)
        try:
            json.dump({'bar': 'boo'}, conf_file)
            conf_file.close()

            args = Namespace(foo=False, config=conf_file.name)

            inputs = InputValues(args)

            self.assertEqual(False, inputs.get('foo', type=bool))
        finally:
            os.remove(conf_file.name)

    def test_boolean_variable_with_command_default_false_is_not_on_the_command_line_or_in_config___command_default_false_value_is_picked_up(self):
        conf_file = NamedTemporaryFile('w', delete=False)
        try:
            json.dump({'bar': 'boo'}, conf_file)
            conf_file.close()

            args = Namespace(foo=False, config=conf_file.name)

            inputs = InputValues(args)

            self.assertEqual(False, inputs.get('foo', type=bool, default=False))
        finally:
            os.remove(conf_file.name)

    def test_boolean_variable_with_command_default_true_is_not_on_the_command_line_or_in_config___command_default_true_value_is_picked_up(self):
        conf_file = NamedTemporaryFile('w', delete=False)
        try:
            json.dump({'bar': 'boo'}, conf_file)
            conf_file.close()

            args = Namespace(foo=True, config=conf_file.name)

            inputs = InputValues(args)

            self.assertEqual(True, inputs.get('foo', type=bool, default=True))
        finally:
            os.remove(conf_file.name)

    def test_boolean_variable_with_command_default_true_is_on_the_command_line_as_false_but_not_in_config___command_line_false_value_is_picked_up(self):
        conf_file = NamedTemporaryFile('w', delete=False)
        try:
            json.dump({'bar': 'boo'}, conf_file)
            conf_file.close()

            args = Namespace(foo=False, config=conf_file.name)

            inputs = InputValues(args)

            self.assertEqual(False, inputs.get('foo', type=bool, default=True))
        finally:
            os.remove(conf_file.name)

    def test_boolean_variable_with_command_default_null_is_on_the_command_line_as_true_but_not_in_config___command_line_true_value_is_picked_up(self):
        conf_file = NamedTemporaryFile('w', delete=False)
        try:
            json.dump({'bar': 'boo'}, conf_file)
            conf_file.close()

            args = Namespace(foo=False, config=conf_file.name)

            inputs = InputValues(args)

            self.assertEqual(False, inputs.get('foo', type=bool))
        finally:
            os.remove(conf_file.name)

    def test_boolean_variable_with_command_default_false_is_on_the_command_line_as_true_but_not_in_config___command_line_true_value_is_picked_up(self):
        conf_file = NamedTemporaryFile('w', delete=False)
        try:
            json.dump({'bar': 'boo'}, conf_file)
            conf_file.close()

            args = Namespace(foo=True, config=conf_file.name)

            inputs = InputValues(args)

            self.assertEqual(True, inputs.get('foo', type=bool, default=False))
        finally:
            os.remove(conf_file.name)

    @given(
        text(min_size=1, max_size=10, alphabet=string.ascii_letters),
        text(min_size=1, max_size=10, alphabet=string.ascii_letters),
    )
    def test_variable_is_on_the_command_line_and_in_config___command_line_is_used(self, cmd_var, conf_var):
        conf_file = NamedTemporaryFile('w', delete=False)
        try:
            json.dump({'foo': conf_var}, conf_file)
            conf_file.close()

            args = Namespace(foo=cmd_var, config=conf_file.name)

            inputs = InputValues(args)

            self.assertEqual(cmd_var, inputs.get('foo'))
        finally:
            os.remove(conf_file.name)

    @given(text(min_size=1, max_size=10, alphabet=string.ascii_letters))
    def test_variable_is_not_on_the_command_line_but_is_in_config___config_is_used(self, conf_var):
        conf_file = NamedTemporaryFile('w', delete=False)
        try:
            json.dump({'foo': conf_var}, conf_file)
            conf_file.close()

            args = Namespace(config=conf_file.name)

            inputs = InputValues(args)

            self.assertEqual(conf_var, inputs.get('foo'))
        finally:
            os.remove(conf_file.name)

    def test_boolean_variable_with_command_default_null_is_not_on_the_command_line_but_is_in_config_as_true___config_true_value_is_picked_up(self):
        conf_file = NamedTemporaryFile('w', delete=False)
        try:
            json.dump({'foo': True}, conf_file)
            conf_file.close()

            args = Namespace(config=conf_file.name)

            inputs = InputValues(args)

            self.assertEqual(True, inputs.get('foo', type=bool))
        finally:
            os.remove(conf_file.name)

    def test_boolean_variable_with_command_default_false_is_not_on_the_command_line_but_is_in_config_as_true___config_false_value_is_picked_up(self):
        conf_file = NamedTemporaryFile('w', delete=False)
        try:
            json.dump({'foo': True}, conf_file)
            conf_file.close()

            args = Namespace(config=conf_file.name)

            inputs = InputValues(args)

            self.assertEqual(True, inputs.get('foo', type=bool, default=False))
        finally:
            os.remove(conf_file.name)

    def test_boolean_variable_with_command_default_true_is_not_on_the_command_line_but_is_in_config_as_false___config_false_value_is_picked_up(self):
        conf_file = NamedTemporaryFile('w', delete=False)
        try:
            json.dump({'foo': False}, conf_file)
            conf_file.close()

            args = Namespace(config=conf_file.name)

            inputs = InputValues(args)

            self.assertEqual(False, inputs.get('foo', type=bool, default=True))
        finally:
            os.remove(conf_file.name)

    def test_boolean_variable_with_command_default_null_is_not_on_the_command_line_but_is_in_config_as_false___config_false_value_is_picked_up(self):
        conf_file = NamedTemporaryFile('w', delete=False)
        try:
            json.dump({'foo': False}, conf_file)
            conf_file.close()

            args = Namespace(config=conf_file.name)

            inputs = InputValues(args)

            self.assertEqual(False, inputs.get('foo', type=bool))
        finally:
            os.remove(conf_file.name)

    def test_variable_is_not_on_the_command_line_or_config_var_is_not_required___none_is_returned(self):
        conf_file = NamedTemporaryFile('w', delete=False)
        try:
            json.dump({'bar': 'boo'}, conf_file)
            conf_file.close()

            args = Namespace(config=conf_file.name)

            inputs = InputValues(args)

            self.assertIsNone(inputs.get('foo'))
        finally:
            os.remove(conf_file.name)

    @given(text(min_size=1, max_size=10, alphabet=string.ascii_letters))
    def test_variable_is_not_on_the_command_line_or_config_var_is_not_required_default_is_supplied___default_is_returned(self, default):
        conf_file = NamedTemporaryFile('w', delete=False)
        try:
            json.dump({'bar': 'boo'}, conf_file)
            conf_file.close()

            args = Namespace(config=conf_file.name)

            inputs = InputValues(args)

            self.assertEqual(default, inputs.get('foo', default=default))
        finally:
            os.remove(conf_file.name)

    def test_variable_is_not_on_the_command_line_or_config_var_is_required___error_is_raised(self):
        conf_file = NamedTemporaryFile('w', delete=False)
        try:
            json.dump({'bar': 'boo'}, conf_file)
            conf_file.close()

            args = Namespace(config=conf_file.name)

            inputs = InputValues(args)

            with self.assertRaises(OasisException):
                inputs.get('foo', required=True)
        finally:
            os.remove(conf_file.name)

    @given(floats(0.0, 1.0))
    def test_float_variable_is_not_on_the_command_line_or_config_var_is_not_required_default_is_supplied___default_is_returned(self, default):
        conf_file = NamedTemporaryFile('w', delete=False)
        try:
            json.dump({'bar': 'boo'}, conf_file)
            conf_file.close()

            args = Namespace(config=conf_file.name)

            inputs = InputValues(args)

            self.assertEqual(
                default, inputs.get('foo', default=default, type=float)
            )
        finally:
            os.remove(conf_file.name)

    @given(floats(0.0, 1.0), floats(0.0, 1.0))
    def test_float_variable_is_on_the_command_line_but_not_in_config_var_is_not_required_default_is_supplied___command_line_is_used(self, cmd_var, default):
        conf_file = NamedTemporaryFile('w', delete=False)
        try:
            json.dump({'bar': 'boo'}, conf_file)
            conf_file.close()

            args = Namespace(foo=cmd_var, config=conf_file.name)

            inputs = InputValues(args)

            self.assertEqual(
                cmd_var, inputs.get('foo', default=default, type=float)
            )
        finally:
            os.remove(conf_file.name)

    @given(floats(0.0, 1.0), floats(0.0, 1.0), floats(0.0, 1.0))
    def test_float_variable_is_on_the_command_line_and_in_config_var_is_not_required_default_is_supplied___command_line_is_used(self, cmd_var, conf_var, default):
        conf_file = NamedTemporaryFile('w', delete=False)
        try:
            json.dump({'foo': conf_var}, conf_file)
            conf_file.close()

            args = Namespace(foo=cmd_var, config=conf_file.name)

            inputs = InputValues(args)

            self.assertEqual(
                cmd_var, inputs.get('foo', default=default, type=float)
            )
        finally:
            os.remove(conf_file.name)

    @given(floats(0.0, 1.0), floats(0.0, 1.0))
    def test_float_variable_is_not_on_command_line_but_is_in_config_var_is_not_required_default_is_supplied___config_is_used(self, conf_var, default):
        conf_file = NamedTemporaryFile('w', delete=False)
        try:
            json.dump({'foo': conf_var}, conf_file)
            conf_file.close()

            args = Namespace(config=conf_file.name)

            inputs = InputValues(args)

            self.assertEqual(
                conf_var, inputs.get('foo', default=default, type=float)
            )
        finally:
            os.remove(conf_file.name)

    def test_variable_is_a_path___path_is_relative_to_config_file(self):
        conf_file = NamedTemporaryFile('w', delete=False)
        try:
            test_path = os.path.join("some", "path")
            json.dump({'foo': test_path}, conf_file)
            conf_file.close()

            expected_result = os.path.abspath(
                os.path.join(os.path.dirname(conf_file.name), test_path)
            )

            args = Namespace(config=conf_file.name)

            inputs = InputValues(args)

            result = inputs.get('foo', required=True, is_path=True)

            self.assertEqual(expected_result, result)
        finally:
            os.remove(conf_file.name)
