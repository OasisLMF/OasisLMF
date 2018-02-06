from ..model_execution.bin import create_binary_files, create_binary_tar_file, check_conversion_tools, \
    check_inputs_directory, cleanup_bin_directory
from .cleaners import PathCleaner
from .base import OasisBaseCommand


class BuildCmd(OasisBaseCommand):
    """
    Builds input binary files for model execution.
    """

    def add_args(self, parser):
        """
        Adds arguments to the argument parser.

        :param parser: The argument parser object
        :type parser: ArgumentParser
        """
        super(BuildCmd, self).add_args(parser)

        parser.add_argument(
            'source', default='.', type=PathCleaner('Source', preexists=True), nargs='?',
            help='The csv source directory.'
        )

        parser.add_argument(
            'destination', default=None, type=PathCleaner('Destination', preexists=False), nargs='?',
            help='The binary destination directory, by default this is the same as the source.'
        )

        parser.add_argument(
            '--do-il', action='store_true',
            help='Flag to build insured loss calculation binaries.'
        )

        parser.add_argument(
            '--build-tar', action='store_true',
            help='Flag to build the binary tar.'
        )

    def action(self, args):
        """
        Builds the input binary files

        :param args: The arguments from the command line
        :type args: Namespace
        """
        destination = args.destination or args.source

        check_conversion_tools(do_il=args.do_il)
        check_inputs_directory(args.source, do_il=args.do_il, check_binaries=False)
        create_binary_files(args.source, destination, do_il=args.do_il)

        if args.build_tar:
            create_binary_tar_file(destination)


class CleanCmd(OasisBaseCommand):
    """
    Cleans up all binary files.
    """

    def add_args(self, parser):
        """
        Adds arguments to the argument parser.

        :param parser: The argument parser object
        :type parser: ArgumentParser
        """
        super(CleanCmd, self).add_args(parser)

        parser.add_argument(
            'target', default='.', type=PathCleaner('Target'), nargs='?',
            help='The directory to clean.'
        )

    def action(self, args):
        """
        Deletes all input binary files

        :param args: The arguments from the command line
        :type args: Namespace
        """
        cleanup_bin_directory(args.target)


class CheckCmd(OasisBaseCommand):
    """
    Checks the required conversion tools and input files are present to build the input binaries.
    """

    def add_args(self, parser):
        """
        Adds arguments to the argument parser.

        :param parser: The argument parser object
        :type parser: ArgumentParser
        """
        super(CheckCmd, self).add_args(parser)

        parser.add_argument(
            'target', default='.', type=PathCleaner('Target'), nargs='?',
            help='The directory to clean.'
        )

        parser.add_argument(
            '--do-il', action='store_true',
            help='Flag to check insured loss calculation inputs.'
        )

        parser.add_argument(
            '--check-binaries', action='store_true',
            help='Flag to check if binary files exist.'
        )

    def action(self, args):
        """
        Checks the required conversion tools and input files are present to build the input binaries.

        :param args: The arguments from the command line
        :type args: Namespace
        """
        check_inputs_directory(args.target, do_il=args.do_il, check_binaries=args.check_binaries)
        check_conversion_tools(do_il=args.do_il)


class BinCmd(OasisBaseCommand):
    """
    Build, clean and check input binary files
    """

    sub_commands = {
        'build': BuildCmd,
        'clean': CleanCmd,
        'check': CheckCmd,
    }
