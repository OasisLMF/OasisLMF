import sys
import os
from pathlib import Path
import shutil
import subprocess
from textwrap import dedent
from tempfile import TemporaryDirectory
import numpy as np
import json

from oasislmf.pytools.converters.data import TOOL_INFO
from tests.pytools.converters.test_converters import compare_conversion_outputs, TESTS_ASSETS_DIR

_DTYPE_FILE = "dtype.json"

def copy_working_files(source_dir, work_dir, file_in, kwarg_file=None):
    source_dir = Path(source_dir)
    work_dir = Path(work_dir)
    shutil.copyfile(source_dir / file_in, work_dir / file_in)
    if kwarg_file is not None:
        shutil.copyfile(source_dir / kwarg_file, work_dir / kwarg_file)


def generate_conversion_script(work_dir, file_in, file_out, file_type,
                               converter="csvtobin",
                               kwarg_file=None):
    # Generate script string
    out_string = dedent(f"""\
            from pathlib import Path
            import json

            from oasislmf.pytools.converters.csvtobin.manager import csvtobin
            from oasislmf.pytools.converters.bintocsv.manager import bintocsv
            from oasislmf.pytools.converters.data import TOOL_INFO

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
            with open(work_dir / \"{_DTYPE_FILE}\", "w") as f:
                json.dump(TOOL_INFO[\"{file_type}\"][\"dtype\"].descr, f)
            """)

    return out_string

def case_runner(converter, file_type, sub_dir, env_vars, filename=None, kwarg_file=None):
    if converter == "bintocsv":
        in_ext = ".bin"
        out_ext = ".csv"
    elif converter == "csvtobin":
        in_ext = ".csv"
        out_ext = ".bin"
    else:
        raise RuntimeError(f"Unknown test type {converter}, {file_type}")

    if filename is None:
        filename = file_type

    file_in = f"{filename}{in_ext}"
    file_out = f"{filename}{out_ext}"

    valid_env_vars = ['OASIS_FLOAT', 'OASIS_INT', 'OASIS_AREAPERIL_TYPE']

    with TemporaryDirectory() as tmp_dir:
        copy_working_files(Path(TESTS_ASSETS_DIR, sub_dir), tmp_dir, file_in,
                           kwarg_file=kwarg_file)
        script = generate_conversion_script(tmp_dir, file_in, file_out,
                                            file_type, kwarg_file=kwarg_file,
                                            converter=converter)

        tmp_path = Path(tmp_dir)
        script_path = tmp_path / "script.py"
        with open(script_path, "w") as f:
            f.write(script)

        # set script env variables
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

        assert file_out in os.listdir(tmp_dir), f"Output file {file_out} not generated."

        with open(tmp_path / _DTYPE_FILE, "r") as f:
            dtype = np.dtype([tuple(_d) for _d in json.load(f)])

        expected_outfile = Path(TESTS_ASSETS_DIR, sub_dir, file_out)
        actual_outfile = Path(tmp_dir, file_out)

        compare_conversion_outputs(expected_outfile, actual_outfile, file_type, out_ext,
                                   dtype=dtype)

def test_coverages():
    case_runner(converter="csvtobin",
                file_type="coverages",
                sub_dir="envdtype",
                env_vars={
                    "OASIS_FLOAT": "f8"
                    }
                )

    case_runner(converter="bintocsv",
                file_type="coverages",
                sub_dir="envdtype",
                env_vars={
                    "OASIS_FLOAT": "f8"
                    },
                )

def test_damagebin():
    case_runner(converter="csvtobin",
                file_type="damagebin",
                sub_dir="envdtype",
                env_vars={
                    "OASIS_FLOAT": "f8"
                    },
                kwarg_file="damagebin_args.json"
                )

    case_runner(converter="bintocsv",
                file_type="damagebin",
                sub_dir="envdtype",
                env_vars={
                    "OASIS_FLOAT": "f8"
                    },
                )

def test_items():
    case_runner(converter="csvtobin",
                file_type="items",
                sub_dir="envdtype",
                env_vars={
                    "OASIS_AREAPERIL_TYPE": "u8"
                    }
                )

    case_runner(converter="bintocsv",
                file_type="items",
                sub_dir="envdtype",
                env_vars={
                    "OASIS_AREAPERIL_TYPE": "u8"
                    },
                )

def test_vulnerability():
    # only testing noidx route
    case_runner(converter="csvtobin",
                file_type="vulnerability",
                sub_dir="envdtype",
                env_vars={
                    "OASIS_FLOAT": "f8"
                    },
                kwarg_file="vulnerability_csvtobin_args.json",
                )

    case_runner(converter="bintocsv",
                file_type="vulnerability",
                sub_dir="envdtype",
                env_vars={
                    "OASIS_FLOAT": "f8"
                    },
                kwarg_file="vulnerability_bintocsv_args.json",
                )

def test_weights():
    case_runner(converter="csvtobin",
                file_type="weights",
                sub_dir="envdtype",
                env_vars={
                    "OASIS_AREAPERIL_TYPE": "u8",
                    "OASIS_FLOAT": "f8"
                    },
                )

    case_runner(converter="bintocsv",
                file_type="weights",
                sub_dir="envdtype",
                env_vars={
                    "OASIS_AREAPERIL_TYPE": "u8",
                    "OASIS_FLOAT": "f8"
                    },
                )
