# evepy/manager.py

import logging
from pathlib import Path

from oasislmf.pytools.utils import redirect_logging

logger = logging.getLogger(__name__)

DEFAULT_EVENTS_FILE = Path('input/events.bin')


def run(input_file, process_number, total_processes, no_shuffle=False,
        randomise=False,
        ):
    """Generate event ID partitions as a binary data stream with shuffling. By
    default the events are shuffled by assiging to processes one by one
    cyclically.

    Args:
        input_file (str | os.PathLike): Path to binary events file. If None
        then defaults to DEFAULT_EVENTS_FILE.
        process_number (int): The process number to receive a partition of events.
        total_processes (int): Total number of partitions of events to distribute to processes.
        no_shuffle (bool, optional): Disable shuffling events. Events are split
            and distributed into blocks in the order they are input. Takes priority over `randomise`.
        randomise (bool, optional): Shuffle events randomly in the blocks. If
            `no_shuffle` is `True` then it takes priority.
    """
    if input_file is None:
        input_file = DEFAULT_EVENTS_FILE

    # Check input file is valid
    input_file = Path(input_file)
    if not input_file.exists():
        raise FileNotFoundError(f"ERROR: File \'{input_file}\' does not exist.")
    if not input_file.is_file():
        raise ValueError(f"ERROR: \'{input_file}\' is not a file.")


@redirect_logging(exec_name='evepy')
def main(input_file=None, process_number=None, total_processes=None,
         no_shuffle=False, randomise=False, **kwargs):

    run(input_file=input_file, process_number=process_number,
        total_processes=total_processes, no_shuffle=no_shuffle,
        randomise=randomise)
