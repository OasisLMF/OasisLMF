from argparsetree import BaseCommand

from .test import Test


class RootCmd(BaseCommand):
    sub_commands = {
        'test': Test
    }
