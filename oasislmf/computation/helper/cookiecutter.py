__all__ = [
    "CreateModelRepo",
    "CreateComplexModelRepo",
]

from subprocess import (
    CalledProcessError,
    run
)

from ..base import ComputationStep


class CreateModelBase(ComputationStep):
    """
    Creates a local Git repository for a "simple model" (using the
    ``cookiecutter`` package) on a "simple model" repository template
    on GitHub)
    """

    step_params = [
        {'name': 'output_dir', 'flag': '-o', 'is_path': True, 'pre_exist': False, 'help': 'Where to generate the project'},
        {'name': 'preset_cookiecutter_json', 'flag': '-p', 'is_path': True, 'pre_exist': True, 'help': 'Cookiecutter JSON file path with all options provided in the file'},
        {'name': 'no_input', 'flag': '-i', 'action': 'store_true', 'help': 'Do not prompt for parameters and only use cookiecutter.json file content'},
        {'name': 'replay', 'flag': '-r', 'action': 'store_true', 'help': 'Do not prompt for parameters and only use information entered previously'},
        {'name': 'overwrite_if_exists', 'flag': '-f', 'action': 'store_true', 'help': 'Overwrite the contents of any preexisting project directory of the same name'},
        {'name': 'cookiecutter_version', 'flag': '-v', 'action': 'store_true', 'help': 'Cookiecutter version'},
    ]

    def run(self):
        cmd_str = 'cookiecutter'
        if not self.cookiecutter_version:
            cmd_str += ''.join([
                (' --no-input' if self.no_input or self.preset_cookiecutter_json else ''),
                (' --replay' if self.replay else ''),
                (' --overwrite-if-exists' if self.overwrite_if_exists else ''),
                (' --output-dir {}'.format(self.output_dir) if self.output_dir else ''),
                (' --verbose ' if (self.logger.level < 20) else ' '),
                self.cookiecutter_uri()
            ])
        else:
            cmd_str += ' -V'

        self.logger.info('\nRunning cookiecutter command: {}\n'.format(cmd_str))

        try:
            run(cmd_str.split(), check=True)
        except CalledProcessError as e:
            self.logger.error(e)

    def cookiecutter_uri(self):
        raise NotImplementedError('Methode run must be implemented, this method returns the cookiecutter template url')


class CreateModelRepo(CreateModelBase):
    """
    Creates a local Git repository for a "simple model" (using the
    ``cookiecutter`` package) on a "simple model" repository template
    on GitHub)
    """
    def cookiecutter_uri(self):
        return 'git+ssh://git@github.com/OasisLMF/CookiecutterOasisSimpleModel'


class CreateComplexModelRepo(CreateModelBase):
    """
    Creates a local Git repository for a "complex model" (using the
    ``cookiecutter`` package) on a "simple model" repository template
    on GitHub)
    """
    def cookiecutter_uri(self):
        return 'git+ssh://git@github.com/OasisLMF/CookiecutterOasisComplexModel'
