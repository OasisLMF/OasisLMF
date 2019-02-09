#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deterministic loss validation
"""
# Python standard library imports
import argparse
import os

# Custom library imports
from tabulate import tabulate

# MDK imports
from oasislmf.model_preparation import oed
from oasislmf.utils.deterministic_loss import (
    generate_oasis_files,
    generate_binary_inputs,
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

    # Create file paths for the source exposure + accounts files -
    # all these are assumed to be already present in the specified input directory
    srcexp_fp = [os.path.join(input_dir, p) for p in os.listdir(input_dir) if p.startswith('location') or p.startswith('srcexp')][0]
    srcacc_fp = [os.path.join(input_dir, p) for p in os.listdir(input_dir) if p.startswith('account') or p.startswith('srcacc')][0]

    # Start Oasis files generation
    oasis_files = generate_oasis_files(
        input_dir,
        srcexp_fp,
        srcacc_fp
    )

    # Generate the binary inputs from the Oasis files
    generate_binary_inputs(input_dir, output_dir)

    losses_df = generate_losses(input_dir, output_dir, loss_percentage_of_tiv=loss_factor, print_losses=False)
    losses_df['event_id'] = losses_df['event_id'].astype(object)
    losses_df['output_id'] = losses_df['output_id'].astype(object)
    print(tabulate(losses_df, headers='keys', tablefmt='psql', floatfmt=".2f"))
