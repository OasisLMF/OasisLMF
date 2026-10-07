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

from tests.pytools.converters.helpers import compare_conversion_outputs, TESTS_ASSETS_DIR

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
        raise RuntimeError(f"Unknown test type {converter}")
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
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert result.returncode == 0, (
        f"conversion subprocess failed ({result.returncode}):\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )


class MultiConversionTest(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp_dir = TemporaryDirectory()
        cls.addClassCleanup(cls.tmp_dir.cleanup)

        cls.case_args = [
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
                 sub_dir="envdtype"),

            dict(converter="csvtobin",
                 file_type="fm",
                 filename="raw_ils",
                 sub_dir="envdtype",
                 kwarg_file="gul_fm_args.json",
                 ),
            dict(converter="bintocsv",
                 file_type="fm",
                 filename="raw_ils",
                 sub_dir="envdtype",
                 ),

            dict(converter="csvtobin",
                 file_type="gul",
                 filename="raw_guls",
                 sub_dir="envdtype",
                 kwarg_file="gul_fm_args.json",
                 ),
            dict(converter="bintocsv",
                 filename="raw_guls",
                 file_type="gul",
                 sub_dir="envdtype",
                 ),
        ]

        env_args = {"OASIS_FLOAT": "f8", "OASIS_INT": "i8",
                    "OASIS_AREAPERIL_TYPE": "u8"}

        cases_runner(cls.case_args, tmp_dir=cls.tmp_dir.name, env_vars=env_args)

        super().setUpClass()

    @staticmethod
    def _run_general_case(case_args, tmp_dir, file_type, abnormal_dtype=False):
        case_args = [ca for ca in case_args if ca['file_type'] == file_type]

        for args in case_args:
            file_out = args['file_out']
            expected_file_out = args['expected_file_out']
            out_ext = args['out_ext']
            sub_dir = args['sub_dir']
            assert file_out in os.listdir(tmp_dir), f"Output file {file_out} not generated."

            with open(Path(tmp_dir, f"{file_type}_{_DTYPE_EXT}"), "r") as f:
                dtype = np.dtype([tuple(_d) for _d in json.load(f)])

            expected_outfile = Path(TESTS_ASSETS_DIR, sub_dir, expected_file_out)
            actual_outfile = Path(tmp_dir, file_out)

            compare_conversion_outputs(expected_outfile, actual_outfile, file_type, out_ext,
                                       dtype=dtype, abnormal_dtype=abnormal_dtype)

    def test_coverages(self):
        self._run_general_case(self.case_args, self.tmp_dir.name, file_type="coverages")

    def test_damagebin(self):
        self._run_general_case(self.case_args, self.tmp_dir.name, file_type="damagebin")

    def test_items(self):
        self._run_general_case(self.case_args, self.tmp_dir.name, file_type="items")

    def test_vulnerability(self):
        self._run_general_case(self.case_args, self.tmp_dir.name, file_type="vulnerability")

    def test_weights(self):
        self._run_general_case(self.case_args, self.tmp_dir.name, file_type="weights")

    def test_fm_profile(self):
        self._run_general_case(self.case_args, self.tmp_dir.name, file_type="fm_profile")

    def test_fm_profile_step(self):
        self._run_general_case(self.case_args, self.tmp_dir.name, file_type="fm_profile_step")

    def test_gul(self):
        self._run_general_case(self.case_args, self.tmp_dir.name, file_type="gul", abnormal_dtype=True)

    def test_fm(self):
        self._run_general_case(self.case_args, self.tmp_dir.name, file_type="fm", abnormal_dtype=True)


def test_summarycalc_oasis_float_f8_round_trips_through_eltpy():
    # Loss/ImpactedExposure were hardcoded to float32 regardless of OASIS_FLOAT, so building
    # with OASIS_FLOAT=f8 made csvtobin write an 8-byte-per-sample stream that eltpy (which reads
    # samples using the real oasis_float width) parses as a 4-byte stream -- misaligning every
    # read after the first sample. Verified end-to-end through the real eltpy reader (elt.manager
    # .run), not just a unit check, since the bug is specifically about the two sides of the
    # stream disagreeing on record width.
    value = 1234567.891234567  # loses precision as float32; distinguishes f4 from f8 handling
    csv = f"EventId,SummaryId,SampleId,Loss,ImpactedExposure\n1,1,1,{value},{value}\n1,1,2,{value},{value}\n"

    with TemporaryDirectory() as tmp:
        Path(tmp, "summarycalc.csv").write_text(csv)

        script = dedent(f"""\
                from pathlib import Path
                from oasislmf.pytools.converters.csvtobin.manager import csvtobin
                from oasislmf.pytools.elt.manager import run as elt_run

                work_dir = Path(r"{tmp}")
                csvtobin(work_dir / "summarycalc.csv", work_dir / "summarycalc.bin", "summarycalc",
                         summary_set_id=1, max_sample_index=10)
                elt_run(str(work_dir), [str(work_dir / "summarycalc.bin")],
                        selt_output_file=str(work_dir / "selt_out.csv"))
                """)
        script_path = Path(tmp, "script.py")
        script_path.write_text(script)

        env = {**os.environ, "OASIS_FLOAT": "f8"}
        result = subprocess.run([sys.executable, str(script_path)], env=env,
                                capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, (
            f"subprocess failed ({result.returncode}):\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

        rows = Path(tmp, "selt_out.csv").read_text().strip().splitlines()[1:]

    # SELT's CSV format is "%.2f", which still distinguishes correct f8 handling (1234567.89)
    # from the old float32 corruption (which would round-trip to 1234567.90)
    assert len(rows) == 2
    for row in rows:
        _, _, _, loss, impacted_exposure = row.split(",")
        assert loss == "1234567.89"
        assert impacted_exposure == "1234567.89"


def test_eve_oasis_int_i8_round_trips_through_evepy():
    # event_id in events.bin is a fixed-width 4-byte int regardless of OASIS_INT, matching
    # ktools' eve.cpp (plain C "int", never configurable) and every other id field (event_id,
    # item_id, sidx, ...) in pytools' own binary streams. read_events used oasis_int instead, so
    # with OASIS_INT=i8 it read the file at double its real record width, corrupting every id.
    csv = "event_id\n1\n2\n3\n4\n"

    with TemporaryDirectory() as tmp:
        Path(tmp, "events.csv").write_text(csv)

        script = dedent(f"""\
                from pathlib import Path
                from oasislmf.pytools.converters.csvtobin.manager import csvtobin
                from oasislmf.pytools.eve.manager import main as eve_main

                work_dir = Path(r"{tmp}")
                csvtobin(work_dir / "events.csv", work_dir / "events.bin", "eve")
                eve_main(input_file=str(work_dir / "events.bin"), process_number=1, total_processes=1,
                         no_shuffle=True, output_file=str(work_dir / "out.bin"))
                """)
        script_path = Path(tmp, "script.py")
        script_path.write_text(script)

        env = {**os.environ, "OASIS_INT": "i8"}
        result = subprocess.run([sys.executable, str(script_path)], env=env,
                                capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, (
            f"subprocess failed ({result.returncode}):\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

        # evepy's own output stream is always a fixed int32, regardless of OASIS_INT
        out_events = np.fromfile(Path(tmp, "out.bin"), dtype=np.int32)

    assert list(out_events) == [1, 2, 3, 4]


def test_generate_losses_events_total_matches_real_event_count_under_oasis_int_i8():
    # GenerateLosses reported progress by dividing events.bin's byte size by oasis_int_size
    # (os.path.getsize("input/events.bin") // oasis_int_size), which under/overcounts
    # events.bin's actual (always-4-byte) records whenever OASIS_INT != i4. event_id_size is
    # the module-level constant the fix introduced for this computation, always resolving to 4.
    csv = "event_id\n1\n2\n3\n4\n"

    with TemporaryDirectory() as tmp:
        Path(tmp, "events.csv").write_text(csv)

        script = dedent(f"""\
                from pathlib import Path
                import os
                from oasislmf.pytools.converters.csvtobin.manager import csvtobin
                from oasislmf.pytools.common.data import oasis_int_size
                from oasislmf.computation.generate.losses import event_id_size

                work_dir = Path(r"{tmp}")
                csvtobin(work_dir / "events.csv", work_dir / "events.bin", "eve")
                size = os.path.getsize(work_dir / "events.bin")

                # the old formula: wrong under OASIS_INT=i8 (demonstrates the bug directly)
                assert size // oasis_int_size != 4, "oasis_int_size-based count unexpectedly correct"

                # the fixed formula: always matches the real (fixed-width) event count
                events_total = size // event_id_size
                assert events_total == 4, f"expected 4 events, got {{events_total}}"
                """)
        script_path = Path(tmp, "script.py")
        script_path.write_text(script)

        env = {**os.environ, "OASIS_INT": "i8"}
        result = subprocess.run([sys.executable, str(script_path)], env=env,
                                capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, (
            f"subprocess failed ({result.returncode}):\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
