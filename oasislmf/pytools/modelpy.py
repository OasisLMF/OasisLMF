import argparse
import logging

from .getmodel import manager, logger


parser = argparse.ArgumentParser()
parser.add_argument('-i', '--file-in', help='names of the input file_path')
parser.add_argument('-o', '--file-out', help='names of the output file_path')
parser.add_argument('-r', '--run-dir', help='path to the run directory', default='.')
parser.add_argument('--ignore-file-type', nargs='*', help='the type of file to be loaded', default=set())
parser.add_argument('--data-server', help='=Use tcp/sockets for IPC data sharing', action='store_true')
parser.add_argument('-v', '--logging-level', help='logging level (debug:10, info:20, warning:30, error:40, critical:50)',
                    default=30, type=int)
# parser.add_argument('-s', '--data-server', help='If True run the data server', default=False, type=bool)


def main() -> None:
    """
    Is the entry point for the modelpy command which loads data and constructs a model.

    Returns: None
    """
    kwargs = vars(parser.parse_args())

    # add handler to fm logger
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logging_level = kwargs.pop('logging_level')
    logger.setLevel(logging_level)

    manager.run(**kwargs)


if __name__ == "__main__":
    main()
