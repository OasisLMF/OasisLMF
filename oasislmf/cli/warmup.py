from .command import OasisBaseCommand
import os
from pathlib import Path


class WarmupCmd(OasisBaseCommand):
    """
    Pre-compiles all Numba JIT functions to eliminate cold-start overhead.
    """

    def action(self, args):
        """
        Runs the JIT warmup in parallel across all pytools modules.

        :param args: The arguments from the command line
        :type args: Namespace
        """
        from oasislmf.warmup import warmup, _DATA_DIR

        if os.environ.get("NUMBA_DISABLE_JIT") == "1":
            self.logger.info("NUMBA_DISABLE_JIT=1 — skipping JIT warmup.")
            return

        from oasislmf.warmup import ALL_TASKS
        self.logger.info(
            f"Warming Numba JIT cache ({len(ALL_TASKS)} tasks, "
            f"max_workers={os.cpu_count()}) ..."
        )
        errors = warmup()
        if errors:
            for name, err in errors.items():
                self.logger.error(f"  {name}: {err}")
            self.logger.error(f"FAILED — {len(errors)} task(s)")
            return 1
        else:
            pkg_root = Path(_DATA_DIR).resolve().parents[1]
            cache_count = sum(1 for _ in pkg_root.rglob("*.nbi"))
            self.logger.info(f"Done — {cache_count} Numba cache files written.")
