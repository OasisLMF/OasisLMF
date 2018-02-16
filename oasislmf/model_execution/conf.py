from __future__ import unicode_literals

import csv
import json
import logging
import os
import io
from collections import defaultdict

import six

from ..utils.exceptions import OasisException
from ..utils.log import oasis_log
from .files import GENERAL_SETTINGS_FILE, GUL_SUMMARIES_FILE, IL_SUMMARIES_FILE, MODEL_SETTINGS_FILE


def _get_summaries(summary_file):
    """
    Get a list representation of a summary file.
    """
    summaries_dict = defaultdict(lambda: {'leccalc': {}})

    with io.open(summary_file, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            id = int(row[0])

            if row[1].startswith('leccalc'):
                summaries_dict[id]['leccalc'][row[1]] = row[2].lower() == 'true'
            else:
                summaries_dict[id][row[1]] = row[2].lower() == 'true'

    summaries = list()
    for id in sorted(six.iterkeys(summaries_dict)):
        summaries_dict[id]['id'] = id
        summaries.append(summaries_dict[id])

    return summaries


@oasis_log
def create_analysis_settings_json(directory):
    """
    Generate an analysis settings JSON from a set of
    CSV files in a specified directory.
    Args:
        ``directory`` (string): the directory containing the CSV files.
    Returns:
        The analysis settings JSON.
    """
    if not os.path.exists(directory):
        error_message = "Directory does not exist: {}".format(directory)
        logging.getLogger().error(error_message)
        raise OasisException(error_message)

    general_settings_file = os.path.join(directory, GENERAL_SETTINGS_FILE)
    model_settings_file = os.path.join(directory, MODEL_SETTINGS_FILE)
    gul_summaries_file = os.path.join(directory, GUL_SUMMARIES_FILE)
    il_summaries_file = os.path.join(directory, IL_SUMMARIES_FILE)

    for file in [general_settings_file, model_settings_file, gul_summaries_file, il_summaries_file]:
        if not os.path.exists(file):
            error_message = "File does not exist: {}".format(directory)
            logging.getLogger().error(error_message)
            raise OasisException(error_message)

    general_settings = dict()
    with io.open(general_settings_file, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            general_settings[row[0]] = eval("{}('{}')".format(row[2], row[1]))

    model_settings = dict()
    with io.open(model_settings_file, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            model_settings[row[0]] = eval("{}('{}')".format(row[2], row[1]))

    gul_summaries = _get_summaries(gul_summaries_file)
    il_summaries = _get_summaries(il_summaries_file)

    analysis_settings = general_settings
    analysis_settings['model_settings'] = model_settings
    analysis_settings['gul_summaries'] = gul_summaries
    analysis_settings['il_summaries'] = il_summaries
    output_json = json.dumps(analysis_settings)
    logging.getLogger().info("Analysis settings json: {}".format(output_json))

    return output_json
