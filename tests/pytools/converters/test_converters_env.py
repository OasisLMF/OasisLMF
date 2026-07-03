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

def copy_working_files(source_dir, work_dir, file_in, kwarg_file=None):
    source_dir = Path(source_dir)
    work_dir = Path(work_dir)
    shutil.copyfile(source_dir / file_in, work_dir / file_in)
    if kwarg_file is not None:
        shutil.copyfile(source_dir / kwarg_file, work_dir / kwarg_file)


def generate_conversion_script(work_dir, file_in, file_out, file_type,
                               kwarg_file=None):
    # Generate script string
    out_string = dedent(f"""\
            from pathlib import Path
            import json

            from oasislmf.pytools.converters.csvtobin.manager import csvtobin
            from oasislmf.pytools.converters.data import TOOL_INFO

            work_dir = Path(\"{work_dir}\")
            kwarg_file = \"{kwarg_file}\"
            if kwarg_file != \"None\":
                with open(work_dir / kwarg_file, "r") as f:
                    kwargs = json.load(f)
            else:
                kwargs = {{}}

            csvtobin(
                file_in = work_dir / \"{file_in}\",
                file_out = work_dir / \"{file_out}\",
                file_type = \"{file_type}\",
                **kwargs
            )

            # Serialise dtype
            with open(work_dir / "dt.json", "w") as f:
                json.dump(TOOL_INFO[\"{file_type}\"][\"dtype\"].descr, f)
            """)

    return out_string

def test_coverages_conversion():
    file_in = "coverages.csv"
    file_out = "coverages.bin"
    file_type = "coverages"
    sub_dir = "envdtype"
    oasis_float = "f8"
    with TemporaryDirectory() as tmp_dir:
        print("Copying working files...")
        copy_working_files(Path(TESTS_ASSETS_DIR, sub_dir), tmp_dir, file_in)
        print("Generating conversion script...")
        script = generate_conversion_script(tmp_dir, file_in, file_out,
                                            file_type)

        tmp_path = Path(tmp_dir)
        script_path = tmp_path / "script.py"
        with open(script_path, "w") as f:
            f.write(script)

        print("Running script")
        env = {**os.environ} # set script environment variables
        if oasis_float is None:
            env.pop("OASIS_FLOAT", None)
        else:
            env["OASIS_FLOAT"] = oasis_float

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

        print("Ran process...")

        assert file_out in os.listdir(tmp_dir), f"Output file {file_out} not generated."

        with open(tmp_path / "dt.json", "r") as f:
            dtype = np.dtype([tuple(_d) for _d in json.load(f)])

        print(f"Loaded dtype: {dtype}")

        expected_outfile = Path(TESTS_ASSETS_DIR, sub_dir, file_out)
        actual_outfile = Path(tmp_dir, file_out)

        compare_conversion_outputs(expected_outfile, actual_outfile, file_type, ".bin",
                                   dtype=dtype)

        print("Compared outfiles and passed")

    print("Done...")
