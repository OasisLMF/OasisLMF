#!/usr/bin/env python3

import argparse
import os
import io
import subprocess
import sys
from tabulate import tabulate


def parse_args():
    desc = (
        'Performance testing script for OasisLMF input file generation'
        'This script expects a set of nested sub directories each containing'
        'acc.csv, loc.csv, keys.csv'
    )
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-t', '--time-threshold', default=300, type=int, help='Maximum time for each file generation test')
    parser.add_argument('-d', '--test-directory', default='.', type=str, help='File path of the test data directory')
    parser.add_argument('-o', '--output-directory', default='/tmp/oasis-files', type=str, help='Filepath to generate oasisfiles in')
    parser.add_argument('-l', '--log-output', default='/var/report/oasisfiles_benchmark.log', type=str, help='Log file path')
    parser.add_argument('-a', '--extra-oasislmf-args', default='', type=str, help='Addtional Aguments to run Oasislmf with')
    return vars(parser.parse_args())


def run_command(cmd_str):
    resp = subprocess.run(cmd_str.split(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    stdout = resp.stdout.decode('utf-8').strip()
    print(stdout)
    resp.check_returncode()
    return stdout


def pasrse_gen_output(stdout_string):
    runtime_list = [l for l in stdout_string.split('\n') if 'COMPLETED' in l]
    t_breakdown = dict()
    total = runtime_list.pop().rsplit(' ').pop()
    t_breakdown['total'] = float(total[:-1])

    for l in runtime_list:
        line = l.split(' ')
        func = line[-3]
        time = line[-1]
        _ ,func ,_ ,time = l.split(' ')
        t_breakdown[func] = float(time[:-1])
    return t_breakdown


def tabulate_data(test_results, output_fp=None):
    input_sizes = sorted(list(test_results.keys()))
    time_values = dict()
    func_names = test_results[input_sizes[0]].keys()

    for f in func_names:
        name = f.split('.')[-1:].pop()
        time_values[name] = list()

    for n in input_sizes:
        for f in func_names:
            name = f.split('.')[-1:].pop()
            time_values[name].append(test_results[n][f])

    timing_tbl = tabulate(
        [[k] + time_values[k] for k in time_values],
        headers=['portfolio size'] + input_sizes,
        tablefmt="rst")

    # if set write to test summary table to log file
    if output_fp:
        log_path = os.path.abspath(output_fp)
        log_dir = os.path.dirname(log_path)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        with io.open(log_path, 'w') as log:
            log.write(timing_tbl)

    print(timing_tbl)


def run_tests(test_dir, run_dir, log_fp, oasis_args, threshold=None):
    '''
    Output of each run entry in `results`

    In [3]: example_run
    Out[3]:
    {'total': 88.63,
     'oasislmf.manager.__init__': 0.0,
     'oasislmf.model_preparation.gul_inputs.get_gul_input_items': 16.05,
     'oasislmf.model_preparation.gul_inputs.write_items_file': 3.84,
     'oasislmf.model_preparation.gul_inputs.write_coverages_file': 1.88,
     'oasislmf.model_preparation.gul_inputs.write_gul_input_files': 5.94,
     'oasislmf.model_preparation.summaries.get_summary_mapping': 0.8,
     'oasislmf.model_preparation.summaries.write_mapping_file': 6.77,
     'oasislmf.model_preparation.il_inputs.get_il_input_items': 30.42,
     'oasislmf.model_preparation.il_inputs.write_fm_policytc_file': 8.49,
     'oasislmf.model_preparation.il_inputs.write_fm_profile_file': 1.59,
     'oasislmf.model_preparation.il_inputs.write_fm_programme_file': 7.52,
     'oasislmf.model_preparation.il_inputs.write_fm_xref_file': 2.98,
     'oasislmf.model_preparation.il_inputs.write_il_input_files': 21.44}
    '''
    sub_dirs = next(os.walk(test_dir))[1]
    test_data = dict()
    results= dict()

    for d in sub_dirs:
        loc_fp = os.path.join(test_dir, d, 'loc.csv')
        acc_fp = os.path.join(test_dir, d, 'acc.csv')
        keys_fp = os.path.join(test_dir, d, 'keys.csv')

        n_sample = sum(1 for line in open(loc_fp)) -1
        cmd_str = f'oasislmf model generate-oasis-files -x {loc_fp} -y {acc_fp} -z {keys_fp} --oasis-files-dir {run_dir} {oasis_args} --verbose'
        test_data[n_sample] = cmd_str

    for t in sorted(test_data.keys()):
        print('Running: ')
        print(f"cmd = {test_data[t]}")
        print(f'size = {t}')
        print(f't_max = {threshold}')
        stdout = run_command(test_data[t])
        run = pasrse_gen_output(stdout)
        results[t] = run
        print(f"t_total = {run['total']}\n")

        # If given check that threshold isn't exceeded
        if threshold:
            if run['total'] > threshold:
                print('FAILED\n')
                tabulate_data(results, log_fp)
                sys.exit(1)
            else:
                print('PASSED\n')

    tabulate_data(results, log_fp)
    return results


if __name__ == "__main__":
    args = parse_args()
    run_tests(args['test_directory'],
              args['output_directory'],
              args['log_output'],
              args['extra_oasislmf_args'],
              args['time_threshold'])
