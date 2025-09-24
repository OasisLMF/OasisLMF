# evepy/manager.py

from contextlib import ExitStack
import logging
from pathlib import Path
import numpy as np

from oasislmf.pytools.utils import redirect_logging
from oasislmf.pytools.common.data import oasis_int, resolve_file

logger = logging.getLogger(__name__)

DEFAULT_EVENTS_FILE = Path('input/events.bin')
NUMPY_RANDOM_SEED = 723706


def read_events(input_file):
    """Read the event IDs from the binary events file.

    Args:
        input_file (str | os.PathLike): Path to binary events file.
    """
    return np.fromfile(input_file, dtype=oasis_int)


def stream_events(events, stream_out):
    """Stream the output events.

    Args:
        events (Iterable): Iterable containing the events to stream.
        stream_out (File object): File object with `write` method for handling output.
    """
    for e in events:
        stream_out.write(np.int32(e).tobytes())


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
    events_per_process = calculate_events_per_process(len(events), total_processes)
    return events[(process_number - 1) * events_per_process:
                  process_number * events_per_process]


def partition_events__random_builtin(events, process_number, total_processes):
    """Shuffle the events randomly and allocate to each process using builtin
    shuffle. Only output the event IDs to the given `process_number`.

    Note that this can be memory intensive. For `len(events) > 10**5` recommend
    using `partition_events__random`.

    Args:
        events (np.array): Array of ordered event IDs.
        process_number (int): The process number to receive a partition of events.
        total_processes (int): Total number of processes to distribute the events over.
    """
    rng = np.random.default_rng(NUMPY_RANDOM_SEED)
    rng.shuffle(events)
    return partition_events__no_shuffle(events, process_number, total_processes)


def partition_events__random(events, process_number, total_processes):
    """Shuffle the events randomly and allocate to each process. Only output
    the event IDs to the given `process_number`. Generates an iterator.

    Randomisation is implemented using the Fisher-Yates algorithm.

    Args:
        events (np.array): Array of ordered event IDs.
        process_number (int): The process number to receive a partition of events.
        total_processes (int): Total number of processes to distribute the events over.
    """
    rng = np.random.default_rng(NUMPY_RANDOM_SEED)

    for i in range(len(events) - 1, 0, -1):
        j = rng.integers(0, i + 1)

        events[i], events[j] = events[j], events[i]

        if (i - process_number) % total_processes == 0:
            yield events[i]

    if process_number % total_processes == 0:  # Event at 0 index
        yield events[0]


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
        randomise=False, randomise_builtin=False, output_file='-',
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
            and distributed into blocks in the order they are input. Takes
            priority over `randomise(_builtin)`.
        randomise (bool, optional): Shuffle events randomly in the blocks and
            stream events on the fly. If `no_shuffle` is `True` then it takes
            priority.
        randomise_builtin (bool, optional): Shuffle events randomly in the blocks using
            builtin shuffle. If `no_shuffle` or `randomise` is `True` then it
            they take priority.
        output_file (str | os.PathLike): Path to output file. If '-' then outputs to stdout.
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
    if no_shuffle and (randomise or randomise_builtin):
        logger.warning("Warning: `no_shuffle` and `randomise(_builtin)` options are incompatible. Ignoring `randomise(_builtin)`.")
        randomise = False
        randomise_builtin = False

    if randomise and randomise_builtin:
        logger.warning("Warning: `randomise` and `randomise_builtin` options are incompatible. Ignoring `randomise_builtin`.")
        randomise_builtin = False

    events = read_events(input_file)

    if no_shuffle:
        event_partitions = partition_events__no_shuffle(events,
                                                        process_number,
                                                        total_processes)
    elif randomise:
        event_partitions = partition_events__random(events,
                                                    process_number,
                                                    total_processes)
    elif randomise_builtin:
        event_partitions = partition_events__random_builtin(events,
                                                            process_number,
                                                            total_processes)
    else:
        event_partitions = partition_events__round_robin(events,
                                                         process_number,
                                                         total_processes)

    with ExitStack() as stack:
        stream_out = resolve_file(output_file, 'wb', stack)

        stream_events(event_partitions, stream_out)


@redirect_logging(exec_name='evepy')
def main(input_file=None, process_number=None, total_processes=None,
         no_shuffle=False, randomise=False, randomise_builtin=False, output_file='-', **kwargs):

    run(input_file=input_file, process_number=process_number,
        total_processes=total_processes, no_shuffle=no_shuffle,
        randomise=randomise, randomise_builtin=randomise_builtin, output_file=output_file)
