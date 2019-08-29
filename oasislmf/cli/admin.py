__all__ = [
    'AdminCmd',
    'CreateComplexModelCmd',
    'CreateSimpleModelCmd',
    'EnableBashCompleteCmd',
]

import os
import sys
import re
import subprocess

from argparse import RawDescriptionHelpFormatter
from platform import system
from subprocess import (
    CalledProcessError,
    run
)

from ..utils.exceptions import OasisException
from ..utils.path import (
    as_path,
    empty_dir,
)
from ..utils.data import get_utctimestamp
from ..utils.path import as_path

from .base import (
    InputValues,
    OasisBaseCommand,
)


class EnableBashCompleteCmd(OasisBaseCommand):
    """
    Adds required command to `.bashrc` Linux or .bash_profile for mac
    so that Command autocomplete works for oasislmf CLI
    """
    formatter_class = RawDescriptionHelpFormatter

    def add_args(self, parser):
        """
        Adds arguments to the argument parser.

        :param parser: The argument parser object
        :type parser: ArgumentParser
        """
        parser.add_argument(
            '-p', '--bash-conf-filepath', default=None, required=False, 
             help='Cookiecutter JSON file path with all options provided in the file'
        )

    def action(self, args):
        # read in user set target install
        inputs = InputValues(args)
        bash_output_fp = as_path(
            inputs.get('bash_conf_filepath'),
            'Complex lookup config JSON file path', preexists=False
        )

        # select default bashrc if not set
        if not bash_output_fp:
            default_file = '.bash_profile' if system == 'Darwin' else '.bashrc'
            bash_output_fp = os.path.join(
                os.path.expanduser('~'), 
                default_file
            )

        # Prompt user, and then install 
        msg_user = 'Running this will append a command to the following file:\n'
        if self.install_confirm(question_str="{} {}".format(msg_user, bash_output_fp)):
            self.install_autocomplete(bash_output_fp)


    def install_autocomplete(self, target_file=None):
        msg_success = 'Auto-Complete installed.' 
        msg_failed = 'install failed'
        msg_installed = 'Auto-Complete feature is already enabled.' 
        cmd_header = '# Added by OasisLMF\n'
        cmd_autocomplete = 'complete -C completer_oasislmf oasislmf\n'
        
        try:
            if os.path.isfile(target_file):
                # Check command is in file
                with open(target_file, "r") as rc:
                    if cmd_autocomplete in rc.read():
                        self.logger.info(msg_installed)
                        sys.exit(0)
            else:
                # create new file at set location            
                basedir = os.path.dirname(target_file)
                if not os.path.isdir(basedir):
                    os.makedirs(basedir)
                
            # Add complete command  
            with open(target_file, "a") as rc:
                    rc.write(cmd_header)
                    rc.write(cmd_autocomplete)
                    self.logger.info(msg_success)
        except Exception as e: 
            self.logger.error('{}: {}'.format(msg_failed, e))    
        

    def install_confirm(self, override=False, question_str='Add line to <somefile>'):
        self.logger.debug('Prompt user for confirmation')
        if override:
            self.logger.debug('Defaulting to YES')
            return True

        try:
            check = str(input("%s (Y/N): " % question_str)).lower().strip()
            if check[0] == 'y':
                return True
            elif check[0] == 'n':
                return False
            else:
                print('Invalid Input')
                return self.confirm_action(question_str)
        except KeyboardInterrupt as e:
            self.logger.error('\nKeyboard Interrupt, exiting.')



class CreateSimpleModelCmd(OasisBaseCommand):
    """
    Creates a local Git repository for a "simple model" (using the
    ``cookiecutter`` package) on a "simple model" repository template
    on GitHub)
    """
    formatter_class = RawDescriptionHelpFormatter

    cookiecutter_template_uri = 'git+ssh://git@github.com/OasisLMF/CookiecutterOasisSimpleModel'

    def add_args(self, parser):
        """
        Adds arguments to the argument parser.

        :param parser: The argument parser object
        :type parser: ArgumentParser
        """
        super(self.__class__, self).add_args(parser)

        parser.add_argument(
            '-p', '--preset-cookiecutter-json', default=None, required=False, help='Cookiecutter JSON file path with all options provided in the file'
        )
        parser.add_argument(
            '-i', '--no-input', default=None, help='Do not prompt for parameters and only use cookiecutter.json file content', action='store_true'
        )
        parser.add_argument(
            '-r', '--replay', default=None, help='Do not prompt for parameters and only use information entered previously', action='store_true'
        )
        parser.add_argument(
            '-f', '--overwrite-if-exists', default=None, help='Overwrite the contents of any preexisting project directory of the same name', action='store_true'
        )
        parser.add_argument(
            '-o', '--output-dir', default=None, required=False, help='Where to generate the project'
        )
        parser.add_argument(
            '-b', '--verbose-mode', default=None, required=False, help='Run cookiecutter in verbose mode', action='store_true'
        )
        parser.add_argument(
            '-c', '--cookiecutter-version', default=None, required=False, help='Cookiecutter version', action='store_true'
        )

    def action(self, args):
        """
        Command action

        :param args: The arguments from the command line
        :type args: Namespace
        """
        inputs = InputValues(args)

        preset_cookiecutter_json = as_path(
            inputs.get('preset_cookiecutter_json', required=False, is_path=True), label='Preset cookiecutter JSON', is_dir=False, preexists=True
        )

        no_prompt = inputs.get('no_input', default=False, required=False)

        replay = inputs.get('replay', default=False, required=False)

        overwrite = inputs.get('overwrite_if_exists', default=True, required=False)

        pkg_dir = os.path.join(os.path.dirname(__file__), os.pardir)

        target_dir = as_path(
            inputs.get('output_dir', default=os.path.join(pkg_dir, 'cookiecutter-run'), required=False, is_path=True), label='Target directory', is_dir=True, preexists=False
        )

        verbose = inputs.get('verbose_mode', default=False, required=False)

        cookiecutter_version = inputs.get('cookiecutter_version', default=False, required=False)

        def run_cmd(cmd_str):
            run(cmd_str.split(), check=True)

        cmd_str = 'cookiecutter'

        if not cookiecutter_version:
            cmd_str += ''.join([ 
                (' --no-input' if no_prompt or preset_cookiecutter_json else ''),
                (' --replay' if replay else ''), 
                (' --overwrite-if-exists' if overwrite else ''), 
                (' --output-dir {}'.format(target_dir) if target_dir else ''),
                (' --verbose ' if verbose else ' '), 
                self.cookiecutter_template_uri
            ])
        else:
            cmd_str += ' -V'

        self.logger.info('\nRunning cookiecutter command: {}\n'.format(cmd_str))

        try:
            run_cmd(cmd_str)
        except CalledProcessError as e:
            self.logger.error(e)


class CreateComplexModelCmd(OasisBaseCommand):
    """
    Creates a local Git repository for a "complex model" (using the
    ``cookiecutter`` package) on a "simple model" repository template
    on GitHub)
    """
    formatter_class = RawDescriptionHelpFormatter

    cookiecutter_template_uri = 'git+ssh://git@github.com/OasisLMF/CookiecutterOasisComplexModel' 

    def add_args(self, parser):
        """
        Adds arguments to the argument parser.

        :param parser: The argument parser object
        :type parser: ArgumentParser
        """
        super(self.__class__, self).add_args(parser)

        parser.add_argument(
            '-p', '--preset-cookiecutter-json', default=None, required=False, help='Cookiecutter JSON file path with all options provided in the file'
        )
        parser.add_argument(
            '-i', '--no-input', default=None, help='Do not prompt for parameters and only use cookiecutter.json file content', action='store_true'
        )
        parser.add_argument(
            '-r', '--replay', default=None, help='Do not prompt for parameters and only use information entered previously', action='store_true'
        )
        parser.add_argument(
            '-f', '--overwrite-if-exists', default=None, help='Overwrite the contents of any preexisting project directory of the same name', action='store_true'
        )
        parser.add_argument(
            '-o', '--output-dir', default=None, required=False, help='Where to generate the project'
        )
        parser.add_argument(
            '-b', '--verbose-mode', default=None, required=False, help='Run cookiecutter in verbose mode', action='store_true'
        )
        parser.add_argument(
            '-c', '--cookiecutter-version', default=None, required=False, help='Cookiecutter version', action='store_true'
        )

    def action(self, args):
        """
        Command action

        :param args: The arguments from the command line
        :type args: Namespace
        """
        inputs = InputValues(args)

        preset_cookiecutter_json = as_path(
            inputs.get('preset_cookiecutter_json', required=False, is_path=True), label='Preset cookiecutter JSON', is_dir=False, preexists=True
        )

        no_prompt = inputs.get('no_input', default=False, required=False)

        replay = inputs.get('replay', default=False, required=False)

        overwrite = inputs.get('overwrite_if_exists', default=True, required=False)

        pkg_dir = os.path.join(os.path.dirname(__file__), os.pardir)

        target_dir = as_path(
            inputs.get('output_dir', default=os.path.join(pkg_dir, 'cookiecutter-run'), required=False, is_path=True), label='Target directory', is_dir=True, preexists=False
        )

        verbose = inputs.get('verbose_mode', default=False, required=False)

        cookiecutter_version = inputs.get('cookiecutter_version', default=False, required=False)

        def run_cmd(cmd_str):
            run(cmd_str.split(), check=True)

        cmd_str = 'cookiecutter'

        if not cookiecutter_version:
            cmd_str += ''.join([ 
                (' --no-input' if no_prompt or preset_cookiecutter_json else ''), 
                (' --replay' if replay else ''), 
                (' --overwrite-if-exists' if overwrite else ''), 
                (' --output-dir {}'.format(target_dir) if target_dir else ''), 
                (' --verbose ' if verbose else ' '), 
                self.cookiecutter_template_uri
            ])
        else:
            cmd_str += ' -V'

        self.logger.info('\nRunning cookiecutter command: {}\n'.format(cmd_str))

        try:
            run_cmd(cmd_str)
        except CalledProcessError as e:
            self.logger.error(e)

class AdminCmd(OasisBaseCommand):
    """
    Admin subcommands::

        * creates a local Git repository for a "simple model" (using the
          ``cookiecutter`` package) on a "simple model" repository template
          on GitHub)
        * creates a local Git repository for a "complex model" (using the
          ``cookiecutter`` package) on a "complex model" repository template
          on GitHub)
    """
    sub_commands = {
        'create-simple-model': CreateSimpleModelCmd,
        'create-complex-model': CreateComplexModelCmd,
        'enable-bash-complete': EnableBashCompleteCmd
    }
