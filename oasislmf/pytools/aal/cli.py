#!/usr/bin/env python

import argparse
import logging

from oasislmf.utils.ping import oasis_ping_progress

from . import manager, logger
from .data import VALID_EXT


def main():
    parser = argparse.ArgumentParser(description='Process average annual loss and standard deviation')
    parser.add_argument('--run_dir', type=str, default='.', help='path to the run directory')
    parser.add_argument('-K', '--subfolder', type=str, default=None, help='workspace sub folder name, inside <run_dir>/work/<sub folder name>')
    parser.add_argument('-a', '--aal', type=str, default=None, help='Output Average Annual Loss Table (AAL)')
    parser.add_argument('-c', '--alct', type=str, default=None, help='Output Average Loss Convergence Table (ALCT)')
    parser.add_argument('-M', '--meanonly', action='store_true', help='Output AAL with mean only')
    parser.add_argument('-l', '--confidence', type=float, default=0.95, help='0 <= confidence level <= 1, default 0.95')
    parser.add_argument('-v', '--logging-level', type=int, default=30,
                        help='logging level (debug:10, info:20, warning:30, error:40, critical:50)')
    parser.add_argument('-H', '--noheader', action='store_true', help='Suppress header in output files')
    parser.add_argument('-E', '--ext', type=str, default='csv', choices=VALID_EXT, help='Output data format')

    parser.add_argument('--socket-server', default='False',
                        help='Report completion to a progress server: a port for the local socket server, or True for the default target')
    parser.add_argument('--analysis-pk', default=None, help='Only used by platform to link run to an analysis')

    args = parser.parse_args()
    kwargs = vars(args)

    # Set up logging
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logging_level = kwargs.pop('logging_level')
    logger.setLevel(logging_level)
    socket_server = kwargs.pop('socket_server')
    analysis_pk = kwargs.pop('analysis_pk')

    manager.main(**kwargs)
    oasis_ping_progress(socket_server, analysis_pk, outputs_complete=1)


if __name__ == '__main__':
    main()
