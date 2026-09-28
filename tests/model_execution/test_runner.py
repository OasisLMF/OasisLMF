import json
import os
from tempfile import TemporaryDirectory
from unittest import TestCase

from mock import patch

from oasislmf.execution.runner import rerun


class RerunTestCase(TestCase):
    """rerun() replays a single failing event outside numba. It parses the
    gul command straight out of run_kernel.sh, which still carries that
    run's own output redirect (e.g. to a fifo whose reader has already
    exited). That stale redirect must be stripped before rerun() appends
    its own `-o` output flag, otherwise the replayed command blocks
    forever trying to write to an orphaned fifo.
    """

    def setUp(self):
        self.tmp_dir = TemporaryDirectory()
        self.addCleanup(self.tmp_dir.cleanup)
        self.cwd = os.getcwd()
        os.chdir(self.tmp_dir.name)
        self.addCleanup(os.chdir, self.cwd)

        with open("event_error.json", "w") as f:
            json.dump({"event_id": "1"}, f)

        with open("run_kernel.sh", "w") as f:
            f.write(
                "( ( eve 1 2 | getmodel | gulcalc -S100 -L100 -r -a1 -i - "
                "> /tmp/xyz123/fifo/gul_P1 ) | fmcalc -a2 -o /tmp/il_P1 ) "
                "2>> log/stderror.err &\n"
            )

    def test_stale_fifo_redirect_is_stripped_from_gul_cmd(self):
        with patch("oasislmf.execution.runner.subprocess.run") as subprocess_run:
            rerun()

        gul_pipe = subprocess_run.call_args_list[0].args[0]

        self.assertNotIn("/tmp/xyz123/fifo/gul_P1", gul_pipe)
        self.assertIn("-o 1_gul.bin", gul_pipe)
        self.assertTrue(gul_pipe.startswith("printf"))

    def test_no_event_error_file_returns_without_running(self):
        os.remove("event_error.json")

        with patch("oasislmf.execution.runner.subprocess.run") as subprocess_run:
            rerun()

        subprocess_run.assert_not_called()

    def test_non_trailing_redirect_is_not_mangled_in_gul_cmd(self):
        # a non-trailing redirect (e.g. the stderr guard appended around the whole
        # pipeline) must survive untouched: only the genuinely trailing output
        # redirect on the gul segment itself should be stripped.
        with open("run_kernel.sh", "w") as f:
            f.write(
                "( ( eve 1 2 | getmodel | gulcalc -S100 -L100 -r -a1 -i - "
                "2>> log/gul_stderror.err > /tmp/xyz123/fifo/gul_P1 ) | fmcalc -a2 > /tmp/il_P1 ) "
                "2>> log/stderror.err &\n"
            )

        with patch("oasislmf.execution.runner.subprocess.run") as subprocess_run:
            rerun()

        gul_pipe = subprocess_run.call_args_list[0].args[0]

        self.assertNotIn("/tmp/xyz123/fifo/gul_P1", gul_pipe)
        self.assertIn("2>> log/gul_stderror.err", gul_pipe)
        self.assertIn("-o 1_gul.bin", gul_pipe)

    def test_stale_redirect_is_stripped_from_fm_cmd(self):
        # the fm segment carries the same kind of stale output redirect (e.g. to a
        # fifo consumed by the original run) and must be stripped before rerun()
        # points it at its own output file, otherwise it hangs the same way the
        # gul segment used to.
        with open("run_kernel.sh", "w") as f:
            f.write(
                "( ( eve 1 2 | getmodel | gulcalc -S100 -L100 -r -a1 -i - "
                "> /tmp/xyz123/fifo/gul_P1 ) | fmcalc -a2 > /tmp/il_P1 ) "
                "2>> log/stderror.err &\n"
            )

        with patch("oasislmf.execution.runner.subprocess.run") as subprocess_run:
            rerun()

        fm_pipe = subprocess_run.call_args_list[1].args[0]

        self.assertNotIn("/tmp/il_P1", fm_pipe)
        self.assertIn("-o 1_fm1.bin", fm_pipe)
        self.assertTrue(fm_pipe.startswith("fmcalc"))
