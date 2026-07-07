import sys
import os
from pathlib import Path
import shutil
import subprocess
from textwrap import dedent
from tempfile import TemporaryDirectory
import numpy as np
import json
from unittest import TestCase

from oasislmf.pytools.converters.data import TOOL_INFO
from tests.pytools.converters.test_converters import compare_conversion_outputs, TESTS_ASSETS_DIR

_DTYPE_EXT = "dtype.json"

def copy_working_files(source_dir, work_dir, file_in, kwarg_file=None):
    source_dir = Path(source_dir)
    work_dir = Path(work_dir)
    shutil.copyfile(source_dir / file_in, work_dir / file_in)
    if kwarg_file is not None:
        shutil.copyfile(source_dir / kwarg_file, work_dir / kwarg_file)


def generate_conversion_fragment(work_dir, file_in, file_out, file_type, converter='csvtobin',
                                 kwarg_file=None):

    output = dedent(f"""\
            work_dir = Path(\"{work_dir}\")
            kwarg_file = \"{kwarg_file}\"
            if kwarg_file != \"None\":
                with open(work_dir / kwarg_file, "r") as f:
                    kwargs = json.load(f)
            else:
                kwargs = {{}}

            {converter}(
                file_in = work_dir / \"{file_in}\",
                file_out = work_dir / \"{file_out}\",
                file_type = \"{file_type}\",
                **kwargs
            )

            # Serialise dtype
            with open(work_dir / \"{file_type}_{_DTYPE_EXT}\", "w") as f:
                json.dump(TOOL_INFO[\"{file_type}\"][\"dtype\"].descr, f)

            """)
    return output

def generate_header_fragment():
    out_string = dedent("""\
            from pathlib import Path
            import json

            from oasislmf.pytools.converters.csvtobin.manager import csvtobin
            from oasislmf.pytools.converters.bintocsv.manager import bintocsv
            from oasislmf.pytools.converters.data import TOOL_INFO

            """)
    return out_string


def converter_to_ext(converter):
    if converter == "bintocsv":
        in_ext = ".bin"
        out_ext = ".csv"
    elif converter == "csvtobin":
        in_ext = ".csv"
        out_ext = ".bin"
    else:
        raise RuntimeError(f"Unknown test type {converter}, {file_type}")
    return in_ext, out_ext


def cases_runner(case_args, tmp_dir, env_vars=None):
    '''
    Run multiple cases with single set of adjusted environment variables.
    '''
    for case_arg in case_args:
        case_arg['in_ext'], case_arg['out_ext'] = converter_to_ext(case_arg['converter'])

        if case_arg.get('filename', None) is None:
            case_arg['filename'] = case_arg['file_type']

        case_arg['file_in'] = f"{case_arg['filename']}{case_arg['in_ext']}"
        case_arg['file_out'] = f"{case_arg['filename']}_out{case_arg['out_ext']}"
        case_arg['expected_file_out'] = f"{case_arg['filename']}{case_arg['out_ext']}"

    valid_env_vars = ['OASIS_FLOAT', 'OASIS_INT', 'OASIS_AREAPERIL_TYPE']

    # Copy all necessary input files
    for case_arg in case_args:
        file_in = case_arg['file_in']
        sub_dir = case_arg['sub_dir']
        kwarg_file = case_arg.get('kwarg_file', None)
        copy_working_files(Path(TESTS_ASSETS_DIR, sub_dir), tmp_dir, file_in,
                           kwarg_file=kwarg_file)

    # Write combined script
    script = generate_header_fragment()
    for case_arg in case_args:
        script += generate_conversion_fragment(
                            work_dir=tmp_dir,
                            file_in=case_arg['file_in'],
                            file_out=case_arg['file_out'],
                            file_type=case_arg['file_type'],
                            converter=case_arg['converter'],
                            kwarg_file=case_arg.get('kwarg_file', None)
                )

    script_path = Path(tmp_dir) / "script.py"
    with open(script_path, 'w') as f:
        f.write(script)

    # Setup environment and run script
    env = {**os.environ}
    for env_key, env_value in env_vars.items():
        if env_key not in valid_env_vars:
            continue

        if env_value is None:
            env.pop(env_key, None)
        else:
            env[env_key] = env_value

    result = subprocess.run(
            [sys.executable, str(script_path)],
            env = env,
            capture_output=True,
            text=True,
            timeout=300,
            )

    assert result.returncode == 0, (
        f"conversion subprocess failed ({result.returncode}):\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )

class MultiConversionTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.tmp_dir = self.enterContext(TemporaryDirectory())

        self.case_args = [
            dict(converter="csvtobin", file_type="coverages",
                 sub_dir="envdtype",),
            dict(converter="bintocsv", file_type="coverages",
                 sub_dir="envdtype",),

            dict(converter="csvtobin", file_type="damagebin",
                 sub_dir="envdtype", kwarg_file="damagebin_args.json"),
            dict(converter="bintocsv", file_type="damagebin",
                 sub_dir="envdtype",),

            dict(converter="csvtobin", file_type="items", sub_dir="envdtype",),
            dict(converter="bintocsv", file_type="items", sub_dir="envdtype",),

            # only test vuln noidx route
            dict(converter="csvtobin", file_type="vulnerability",
                 sub_dir="envdtype",
                 kwarg_file="vulnerability_csvtobin_args.json",),
            dict(converter="bintocsv", file_type="vulnerability",
                 sub_dir="envdtype",
                 kwarg_file="vulnerability_bintocsv_args.json",
                ),

            dict(converter="csvtobin",
                file_type="weights",
                sub_dir="envdtype",
                ),
            dict(converter="bintocsv",
                file_type="weights",
                sub_dir="envdtype",
                ),

            dict(converter="csvtobin",
                 file_type="fm_profile",
                 sub_dir="envdtype"),
            dict(converter="bintocsv",
                 file_type="fm_profile",
                 sub_dir="envdtype"),

            dict(converter="csvtobin",
                 file_type="fm_profile_step",
                 sub_dir="envdtype"),
            dict(converter="bintocsv",
                 file_type="fm_profile_step",
                 sub_dir="envdtype")
            ]

        env_args = { "OASIS_FLOAT": "f8", "OASIS_INT": "i8",
                    "OASIS_AREAPERIL_TYPE": "u8" }

        cases_runner(self.case_args, tmp_dir=self.tmp_dir, env_vars=env_args)


    def _run_general_case(self, file_type, abnormal_dtype=False):
        case_args = [ca for ca in self.case_args if ca['file_type'] == file_type]

        for args in case_args:
            file_out = args['file_out']
            expected_file_out = args['expected_file_out']
            out_ext = args['out_ext']
            sub_dir = args['sub_dir']
            assert file_out in os.listdir(self.tmp_dir), f"Output file {file_out} not generated."

            with open(Path(self.tmp_dir, f"{file_type}_{_DTYPE_EXT}"), "r") as f:
                dtype = np.dtype([tuple(_d) for _d in json.load(f)])

            expected_outfile = Path(TESTS_ASSETS_DIR, sub_dir, expected_file_out)
            actual_outfile = Path(self.tmp_dir, file_out)

            compare_conversion_outputs(expected_outfile, actual_outfile, file_type, out_ext,
                                       dtype=dtype, abnormal_dtype=abnormal_dtype)


    def test_coverages(self):
        self._run_general_case(file_type="coverages")

    def test_damagebin(self):
        self._run_general_case(file_type="damagebin")

    def test_items(self):
        self._run_general_case(file_type="items")

    def test_vulnerability(self):
        self._run_general_case(file_type="vulnerability")

    def test_weights(self):
        self._run_general_case(file_type="weights")

    def test_fm_profile(self):
        self._run_general_case(file_type="fm_profile")

    def test_fm_profile_step(self):
        self._run_general_case(file_type="fm_profile_step")
