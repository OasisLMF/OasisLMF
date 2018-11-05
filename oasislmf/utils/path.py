from contextlib import contextmanager

import os


@contextmanager
def setcwd(path):
    pwd = os.getcwd()
    os.chdir(str(path))
    yield path
    os.chdir(pwd)
