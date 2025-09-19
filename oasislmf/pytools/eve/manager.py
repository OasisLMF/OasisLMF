# evepy/manager.py

import logging
from pathlib import Path
import numpy as np

from oasislmf.pytools.utils import redirect_logging
from oasislmf.pytools.common.data import oasis_int

logger = logging.getLogger(__name__)

DEFAULT_EVENTS_FILE = Path('input/events.bin')


def read_events(input_file):
    """Read the event IDs from the binary events file.

    Args:
        input_file (str | os.PathLike): Path to binary events file.
    """
    return np.fromfile(input_file, dtype=oasis_int)


def ceil_int(numerator, divisor):
    """Perform ceil on integers without converting to floar (like builtin
    `ceil`).
    """
    quotient, remainder = divmod(numerator, divisor)
    return quotient + bool(remainder)  # add 1 if remainder


def partition_events__no_shuffle(events, process_number, total_processes):
    """Assign events in the order they are loaded to each process in turn. Only
    output the event IDs allocated to the given `process_number`.

    Args:
        events (np.array): Array of ordered event IDs.
        process_number (int): The process number to receive a partition of events.
        total_processes (int): Total number of partitions of events to distribute to processes.
    """
    # TODO - check #-events < total_processes
    events_per_partition = ceil_int(len(events), total_processes)
    return events[(process_number - 1) * events_per_partition:
                  process_number * events_per_partition]


def partition_events__random(events, process_number, total_processes):
    """Shuffle the events randomly and allocate to each process. Only output
    the event IDs to the given `process_number`.
    """
    pass


def partition_events__sequential(events, process_number, total_processes):
    """Partition the events sequentially in a round robin style per proccess.
    Only output the events allocated to the given `process_number`.
    """
    pass


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

    # Check shuffle and randomise settings
    if no_shuffle and randomise:
        logger.warning("Warning: `no_shuffle` and `randomise` options are incompatible. Ignoring `randomise`.")
        randomise = False

    events = read_events(input_file)

    if no_shuffle:
        event_partitions = partition_events__no_shuffle(events,
                                                        process_number,
                                                        total_processes)
    elif randomise:
        event_partitions = partition_events__random(events,
                                                    process_number,
                                                    total_processes)
    else:
        event_partitions = partition_events__sequential(events,
                                                        process_number,
                                                        total_processes)

    # output event partitions


@redirect_logging(exec_name='evepy')
def main(input_file=None, process_number=None, total_processes=None,
         no_shuffle=False, randomise=False, **kwargs):

    run(input_file=input_file, process_number=process_number,
        total_processes=total_processes, no_shuffle=no_shuffle,
        randomise=randomise)
