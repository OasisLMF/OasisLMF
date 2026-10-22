#!/usr/bin/env python

import argparse
import logging

from oasislmf.utils.ping import oasis_ping_progress

from . import manager, logger


def main():
    parser = argparse.ArgumentParser(description='Join Summary ID info to data file and output new file')
    parser.add_argument('-s', '--summaryinfo', type=str, required=True, default=None, help='Summary Info File')
    parser.add_argument('-d', '--data', type=str, required=True, default=None, help='Ouput Data File')
    parser.add_argument('-o', '--output', type=str, required=True, default=None, help='Joined Output File')
    parser.add_argument('-v', '--logging-level', type=int, default=30,
                        help='logging level (debug:10, info:20, warning:30, error:40, critical:50)')
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
