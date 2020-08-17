__all__ = [
    "HelperTabComplete",
]


from ..base import ComputationStep

import os
import sys
from platform import system


class HelperTabComplete(ComputationStep):
    """
    Adds required command to `.bashrc` Linux or .bash_profile for mac
    so that Command autocomplete works for oasislmf CLI
    """
    step_params = [
        {'name': 'bash_rc_file', 'flag': '-p', 'help': 'Path to bash configuration RC file, "~/.bashrc". '},
        {'name': 'no_confirm', 'flag': '-y', 'action': 'store_true', 'default': False, 'help': 'Skip the confirmation prompt'},
    ]

    def confirm_action(self, question_str, no_confirm=False):
        self.logger.debug('Prompt user for confirmation')
        if no_confirm:
            return True
        try:
            check = str(input("%s (Y/N): " % question_str)).lower().strip()
            if check[:1] == 'y':
                return True
            elif check[:1] == 'n':
                return False
            else:
                self.logger.error('Enter "y" for Yes, "n" for No or Ctrl-C to exit.\n')
                return self.confirm_action(question_str)
        except KeyboardInterrupt:
            self.logger.error('\nexiting.')

    def install_autocomplete(self, target_file=None):
        msg_success = 'Auto-Complete installed.'
        msg_failed = 'install failed'
        msg_installed = 'Auto-Complete feature is already enabled.'
        msg_reload_bash = '\n To activate reload bash by running: \n     source {}'.format(target_file)
        cmd_header = '# Added by OasisLMF\n'
        cmd_autocomplete = 'complete -C completer_oasislmf oasislmf\n'

        try:
            if os.path.isfile(target_file):
                # Check command is in file
                with open(target_file, "r") as rc:
                    if cmd_autocomplete in rc.read():
                        self.logger.info(msg_installed)
                        self.logger.info(msg_reload_bash)
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
                self.logger.info(msg_reload_bash)
        except Exception as e:
            self.logger.error('{}: {}'.format(msg_failed, e))

    def run(self):

        # select default bashrc if not set
        if not self.bash_rc_file:
            default_file = '.bash_profile' if system == 'Darwin' else '.bashrc'
            self.bash_rc_file = os.path.join(
                os.path.expanduser('~'),
                default_file
            )

        # Prompt user, and then install
        msg_user = 'Running this will append a command to the following file:\n'
        if self.confirm_action("{} {}".format(msg_user, self.bash_rc_file), self.no_confirm):
            self.install_autocomplete(self.bash_rc_file)
