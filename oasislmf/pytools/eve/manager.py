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


def stream_events(events):
    """Stream the output events.

    Args:
        events (Iterable): Iterable containing the events to stream.
    """
    pass


def calculate_events_per_process(n_events, total_processes):
    """Calculate number of events per process.
    """
    events_per_process, remainder = divmod(n_events, total_processes)
    return events_per_process + bool(remainder)  # add 1 if remainder


def partition_events__no_shuffle(events, process_number, total_processes):
    """Assign events in the order they are loaded to each process in turn. Only
    output the event IDs allocated to the given `process_number`.

    Args:
        events (np.array): Array of ordered event IDs.
        process_number (int): The process number to receive a partition of events.
        total_processes (int): Total number of processes to distribute the events over.
    """
    # TODO - check #-events < total_processes
    events_per_process = calculate_events_per_process(len(events), total_processes)
    return events[(process_number - 1) * events_per_process:
                  process_number * events_per_process]


def partition_events__random(events, process_number, total_processes):
    """Shuffle the events randomly and allocate to each process. Only output
    the event IDs to the given `process_number`.

    Args:
        events (np.array): Array of ordered event IDs.
        process_number (int): The process number to receive a partition of events.
        total_processes (int): Total number of processes to distribute the events over.
    """
    pass


def partition_events__round_robin(events, process_number, total_processes):
    """Partition the events sequentially in a round robin style per process.
    Only output the events allocated to the given `process_number`.

    Args:
        events (np.array): Array of ordered event IDs.
        process_number (int): The process number to receive a partition of events.
        total_processes (int): Total number of processes to distribute the events over.
    """
    return events[np.arange(process_number - 1, len(events), total_processes)]


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
        total_processes (int): Total number of processes to distribute the events over.
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
        event_partitions = partition_events__round_robin(events,
                                                         process_number,
                                                         total_processes)

    stream_events(event_partitions)


@redirect_logging(exec_name='evepy')
def main(input_file=None, process_number=None, total_processes=None,
         no_shuffle=False, randomise=False, **kwargs):

    run(input_file=input_file, process_number=process_number,
        total_processes=total_processes, no_shuffle=no_shuffle,
        randomise=randomise)
