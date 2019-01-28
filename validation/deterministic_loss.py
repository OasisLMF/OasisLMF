#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Deterministic loss validation
"""
# Python standard library imports
import argparse
import os
import subprocess

from shutil import copyfile

# Custom library imports
from tabulate import tabulate

# MDK imports
from oasislmf.exposures import oed
from oasislmf.utils.deterministic_loss import (
    generate_oasis_files,
    generate_losses,
)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Validate deterministic losses')
    parser.add_argument(
        '-o', '--output-dir', type=str, required=True, default=os.path.join(os.getcwd(), 'output'),
        help='Output dir')
    parser.add_argument(
        '-i', '--input-dir', type=str, default=os.getcwd(), required=True,
        help='Input dir - should contain the OED exposure + accounts data + other assets required for Oasis files generation via MDK')
    parser.add_argument(
        '-l', '--loss-factor', type=float, default=1.0,
        help='Loss factor to apply to TIVs.')

    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir) if not os.path.isabs(args.output_dir) else args.output_dir
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    input_dir = os.path.abspath(args.input_dir) if not os.path.isabs(args.input_dir) else args.input_dir

    loss_factor = args.loss_factor

    # Create file paths for the source exposure + accounts files + transformation files
    # and all these are assumed to be already present in the specified input directory
    srcexp_fp = os.path.join(input_dir, 'location.csv')
    srcexptocan_trans_fp = os.path.join(input_dir, 'MappingMapToOED_CanLocA.xslt')

    srcacc_fp = os.path.join(input_dir, 'account.csv')
    srcacctocan_trans_fp = os.path.join(input_dir, 'MappingMapToOED_CanAccA.xslt')

    # Start Oasis files generation
    generate_oasis_files(
        input_dir,
        srcexp_fp,
        srcexptocan_trans_fp,
        srcacc_fp,
        srcacctocan_trans_fp
    )

    # copy the Oasis files to the output directory and convert to binary
    input_files = oed.GUL_INPUTS_FILES + oed.IL_INPUTS_FILES

    for input_file in input_files:
        conversion_tool = oed.CONVERSION_TOOLS[input_file]
        input_file_path = input_file + ".csv"

        if not os.path.exists(os.path.join(input_dir, input_file_path)):
            continue

        copyfile(
            os.path.join(input_dir, input_file_path),
            os.path.join(output_dir, input_file_path)
        )

        input_file_path = os.path.join(output_dir, input_file_path)

        output_file_path = os.path.join(output_dir, input_file + ".bin")
        command = "{} < {} > {}".format(
            conversion_tool, input_file_path, output_file_path)
        proc = subprocess.Popen(command, shell=True)
        proc.wait()
        if proc.returncode != 0:
            raise Exception(
                "Failed to convert {}: {}".format(input_file_path, command))

    losses_df = generate_losses(input_dir, output_dir, loss_percentage_of_tiv=loss_factor, print_losses=False)
    losses_df['event_id'] = losses_df['event_id'].astype(object)
    losses_df['output_id'] = losses_df['output_id'].astype(object)
    print(tabulate(losses_df, headers='keys', tablefmt='psql', floatfmt=".2f"))
