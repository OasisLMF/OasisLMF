#!/usr/bin/env python
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

    srcexptocan_trans_fp = os.path.join(input_dir, 'MappingMapToOED_CanLocA.xslt')
    srcacctocan_trans_fp = os.path.join(input_dir, 'MappingMapToOED_CanAccA.xslt')


    (ri_layers, xref_descriptions) = generate_oasis_files(
        input_dir, output_dir, 
        srcexptocan_trans_fp, srcacctocan_trans_fp)
    net_losses = generate_losses(
        output_dir, xref_descriptions, loss_percentage_of_tiv=loss_factor, ri_layers=ri_layers)

    for (description, net_loss) in net_losses.items():
        print(description)
        print(tabulate(net_loss, headers='keys', tablefmt='psql', floatfmt=".2f"))
        print("")
