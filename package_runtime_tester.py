#!/usr/bin/env python3

"""
Installs a version of the MDK package from a given branch in the OasisLMF
GitHub repository, and doe an end-to-end runtime test against a small
PiWind sample dataset.
"""
import argparse
import os
import shutil
import subprocess
import sys

from pkg_resources import DistributionNotFound

from subprocess import (
    CalledProcessError,
    run,
)


class MDKRuntimeTesterException(Exception):
    pass


def get_default_pip_path():
    default_pip_path = ''

    cmd_str = 'which pip'
    try:
        resp = run(cmd_str.split(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True).stdout
    except CalledProcessError as e:
        return
    else:
        if not resp:
            raise MDKRuntimeTesterException('No default pip path found!')
        default_pip_path = resp.decode('utf-8').strip()

    return default_pip_path


def parse_args():
    parser = argparse.ArgumentParser(
        description='Installs a version of the MDK package '
                    'from a given branch in the OasisLMF '
                    'GitHub repository, and does an '
                    'end-to-end runtime test against a '
                    'small PiWind sample dataset.'
    )

    parser.add_argument('-b', '--package-repo-branch', default='develop', help='Target branch in the package GitHub repository to build the package from')

    parser.add_argument('-w', '--piwind-repo-branch', default='master', help='Target branch in the PiWind GitHub repository to clone')

    parser.add_argument('-t', '--piwind-clone-target', default=os.path.abspath('.'), help='Local parent folder in which to clone PiWind - default is script run directory')

    parser.add_argument('-g', '--git-transfer-protocol', default='ssh', help='Git transfer protocol - https" or "ssh"')

    parser.add_argument('-p', '--default-pip-path', default=get_default_pip_path(), help='Default pip path')

    args = vars(parser.parse_args())

    if not os.path.isabs(args['piwind_clone_target']):
        args['piwind_clone_target'] = os.path.abspath(args['piwind_clone_target'])

    return args

def mdk_pkg_exists():
    cmd_str = 'pip freeze | grep oasislmf'
    try:
        resp = run(cmd_str.split(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True).stdout
    except CalledProcessError:
        return False
    else:
        if not resp:
            raise MDKRuntimeTesterException('Unable to run command "pip freeze | grep oasislmf"')
        return 'oasislmf' in resp.decode('utf-8').strip()


def pip_uninstall(pkg_name, options_str='-v', pip_path=get_default_pip_path()):
    if mdk_pkg_exists():
        cmd_str = 'pip uninstall {} {}'.format(options_str, pkg_name)
        run(cmd_str.split(), check=True)


def pip_install(pkg_name_or_uri, options_str='-v', pip_path=get_default_pip_path()):
    if not mdk_pkg_exists():
        cmd_str = '{} install {} {}'.format(pip_path, options_str, pkg_name_or_uri)
        run(cmd_str.split(), check=True)


def git_clone(repo_url, options_str=''):
    cmd_str = 'git clone {} {}'.format(options_str, repo_url)
    run(cmd_str.split(), check=True)


def clone_piwind(target, home=os.getcwd(), transfer_protocol='ssh'):
    if not os.path.exists(target):
        os.mkdir(target)

    piwind_target = os.path.join(target, 'OasisPiWind')
    if os.path.exists(piwind_target):
        shutil.rmtree(piwind_target)

    os.chdir(target)

    piwind_repo_url = 'git+{}://git@github.com/OasisLMF/OasisPiWind'.format(transfer_protocol)

    git_clone(piwind_repo_url)

    os.chdir(home)


def run_model_via_mdk(model_mdk_config_fp, model_run_dir=os.path.abspath('.')):
    cmd_str = 'oasislmf model run -C {} -r {}'.format(model_mdk_config_fp, model_run_dir)
    run(cmd_str.split(), check=True)


def cleanup(pip_path=get_default_pip_path(), piwind_target=None):
    pip_uninstall('oasislmf', options_str='-v -y', pip_path=pip_path)
    if piwind_target:
        shutil.rmtree(piwind_target)


if __name__ == "__main__":

    args = parse_args()

    if not args['default_pip_path']:
        raise MDKRuntimeTesterException('Default pip path could not be determined and/or no default pip path provided when calling the script')

    if args['git_transfer_protocol'] not in ['https', 'ssh']:
        args['git_transfer_protocol'] = 'ssh'

    pkg_uri = 'git+{}://git@github.com/OasisLMF/OasisLMF.git@{}#egg=oasislmf'.format(args['git_transfer_protocol'], args['package_repo_branch'])

    print('\nInstalling MDK package {}'.format(pkg_uri))

    try:
        pip_install(pkg_uri, pip_path=args['default_pip_path'])
    except CalledProcessError as e:
        raise MDKRuntimeTesterException('\nError trying to pip install package: {}'.format(e))

    print('\nMDK package successfully installed from branch {}'.format(args['package_repo_branch']))

    print('\nCloning PiWind in \n\t{}\n'.format(args['piwind_clone_target']))

    try:
        piwind_fp = clone_piwind(args['piwind_clone_target'], transfer_protocol=args['git_transfer_protocol'])
    except CalledProcessError as e:
        raise MDKRuntimeTesterException('\nError while trying to clone PiWind repository: {}\n'.format(e))

    print('\nPiWind successfully cloned in {}\n'.format(args['piwind_clone_target']))

    piwind_fp = os.path.join(args['piwind_clone_target'], 'OasisPiWind')
    piwind_mdk_config_fp = os.path.join(piwind_fp, 'oasislmf-oed.json')
    print('\nRunning PiWind end-to-end via MDK using config. file {}\n'.format(piwind_mdk_config_fp))

    try:
        run_model_via_mdk(piwind_mdk_config_fp, model_run_dir=piwind_fp)
    except CalledProcessError as e:
        raise MDKRuntimeTesterException('\nError while trying to run PiWind via MDK: {}'.format(e))

    sys.exit(0)
