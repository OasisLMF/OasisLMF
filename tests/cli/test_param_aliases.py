"""`--ri-info` and `--ri-scope` match `ods_tools check` (#2228)."""
import argparse

import pytest

from oasislmf.cli.root import RootCmd


def _leaf_parsers(parser, path=()):
    sub_actions = [a for a in parser._actions if isinstance(a, argparse._SubParsersAction)]
    if not sub_actions:
        yield path, parser
    for action in sub_actions:
        for name, sub_parser in action.choices.items():
            yield from _leaf_parsers(sub_parser, path + (name,))


@pytest.fixture(scope="module")
def root_parser():
    return RootCmd().arg_parser


@pytest.mark.parametrize("command", [["exposure", "run"], ["model", "run"], ["model", "generate-oasis-files"]])
def test_ri_aliases_set_the_oed_ri_file_params(root_parser, command):
    args = root_parser.parse_args(command + ["--ri-info", "info.csv", "--ri-scope", "scope.csv"])
    assert (args.oed_info_csv, args.oed_scope_csv) == ("info.csv", "scope.csv")


def test_every_command_with_oed_ri_files_accepts_the_aliases(root_parser):
    checked = 0
    for path, parser in _leaf_parsers(root_parser):
        options = {o for a in parser._actions for o in a.option_strings}
        if "--oed-info-csv" in options:
            assert "--ri-info" in options, path
            checked += 1
        if "--oed-scope-csv" in options:
            assert "--ri-scope" in options, path
    assert checked > 0


def test_filter_help_says_what_the_values_are(root_parser):
    parser = dict(_leaf_parsers(root_parser))[("model", "generate-oasis-files")]
    help_by_dest = {a.dest: a.help for a in parser._actions}
    assert help_by_dest["account"] == "Only include these accounts (AccNumber values)"
    assert help_by_dest["location"] == "Only include these locations (LocNumber values)"
