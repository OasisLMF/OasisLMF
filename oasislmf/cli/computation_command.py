__all__ = ["OasisComputationCommand"]

from ..manager import OasisManager as om

from .inputs import InputValues
from .base import OasisBaseCommand


class OasisComputationCommand(OasisBaseCommand):
    """
    Eventually, the Parent class for all Oasis Computation Command
    create the command line interface from parameter define in the associated computation step
    """

    def add_args(self, parser):
        """
        Adds arguments to the argument parser.

        :param parser: The argument parser object
        :type parser: ArgumentParser
        """
        super().add_args(parser)

        for param in om.computations_params[self.computation_name]:
            add_argument_kwargs = {key: param.get(key) for key in ['action', 'nargs', 'const', 'type', 'choices',
                                                                   'help', 'metavar', 'dest']}
            arg_name = f"--{param['name'].replace('_', '-')}"
            if param.get('flag'):
                parser.add_argument(param.get('flag'), arg_name, **add_argument_kwargs)
            else:
                parser.add_argument(arg_name, **add_argument_kwargs)

    def action(self, args):
        """
        Generic method that call the correct manager method from the child class computation_name

        :param args: The arguments from the command line
        :type args: Namespace
        """
        self.logger.info(f'\nProcessing arguments - {self.computation_name}')
        inputs = InputValues(args)

        _kwargs = {
            param['name']: inputs.get(param['name'], required=param.get('required'), is_path=param.get('is_path')) for
            param in om.computations_params[self.computation_name]}

        manager_method = getattr(om(), om.computation_name_to_method(self.computation_name))
        manager_method(**_kwargs)
