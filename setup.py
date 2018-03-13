from __future__ import print_function, division

import glob
import os
import io
import re
import shutil
import sys
import tarfile
from contextlib import contextmanager
from distutils.log import INFO, WARN, ERROR
from tempfile import mkdtemp
from time import sleep

from setuptools import find_packages, setup, Command
from setuptools.command.install import install

try:
    from urllib.request import urlopen
    from urllib.error import URLError
except ImportError:
    from urllib2 import urlopen, URLError

KTOOLS_VERSION = '0_0_392_0'
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))


def get_readme():
    with io.open(os.path.join(SCRIPT_DIR, 'README.rst'), encoding='utf-8') as readme:
        return readme.read()


def get_install_requirements():
    with io.open(os.path.join(SCRIPT_DIR, 'requirements-package.in'), encoding='utf-8') as reqs:
        return reqs.readlines()


def get_version():
    """
    Return package version as listed in `__version__` in `init.py`.
    """
    with io.open(os.path.join(SCRIPT_DIR, 'oasislmf', '__init__.py'), encoding='utf-8') as init_py:
        return re.search('__version__ = [\'"]([^\'"]+)[\'"]', init_py.read()).group(1)


@contextmanager
def temp_dir():
    d = mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


version = get_version()
reqs = get_install_requirements()
readme = get_readme()


class PostInstallKtools(install):
    command_name = 'install'
    user_options = install.user_options + [
        ('ktools', None, 'Only install ktools components'),
    ]
    boolean_options = install.boolean_options + ['ktools']

    def __init__(self, *args, **kwargs):
        self.ktools_components = []
        install.__init__(self, *args, **kwargs)

    def run(self):
        self.install_ktools()
        install.run(self)

    def get_outputs(self):
        return install.get_outputs(self) + self.ktools_components

    def fetch_ktools_tar(self, location, attempts=3, timeout=5, cooldown=1):
        self.announce('Retrieving ktools {}'.format(KTOOLS_VERSION), INFO)

        last_error = None
        request = None
        for i in range(attempts):
            try:
                request = urlopen('https://github.com/OasisLMF/ktools/archive/OASIS_{}.tar.gz'.format(KTOOLS_VERSION), timeout=timeout * 1000)
                break
            except URLError as e:
                self.announce('Failed to get ktools tar (attempt {})'.format(i + 1), WARN)
                last_error = e
                sleep(cooldown)
        else:
            self.announce('Failed to get ktools tar after {} attempts'.format(attempts), ERROR)
            if last_error:
                raise last_error

        with open(location, 'wb') as f:
            f.write(request.read())

    def unpack_tar(self, tar_location, extract_location):
        self.announce('Unpacking ktools', INFO)
        with tarfile.open(tar_location) as tar:
            if not os.path.exists(extract_location):
                os.makedirs(extract_location)
            tar.extractall(extract_location)

    def build_ktools(self, extract_location):
        self.announce('Building ktools', INFO)
        build_dir = os.path.join(extract_location, 'ktools-OASIS_{}'.format(KTOOLS_VERSION))

        os.system('cd {build_dir} && ./autogen.sh && ./configure && make && make check'.format(build_dir=build_dir))
        return build_dir

    def add_ktools_to_path(self, build_dir):
        print('Installing ktools')

        if not os.path.exists(self.install_scripts):
            os.makedirs(self.install_scripts)

        for p in glob.glob(os.path.join(build_dir, 'src', '*', '*')):
            split = p.split(os.path.sep)

            # if the file name is the same as the directory we have found a
            # component executable
            if split[-1] == split[-2]:
                component_path = os.path.join(self.install_scripts, split[-1])
                shutil.copy(p, component_path)
                yield component_path

    def install_ktools(self):
        with temp_dir() as d:
            local_tar_path = os.path.join(d, 'ktools.tar.gz')
            local_extract_path = os.path.join(d, 'extracted')

            self.fetch_ktools_tar(local_tar_path)
            self.unpack_tar(local_tar_path, local_extract_path)
            build_dir = self.build_ktools(local_extract_path)
            self.ktools_components = list(self.add_ktools_to_path(build_dir))


try:
    from wheel.bdist_wheel import bdist_wheel

    class BdistWheel(bdist_wheel):
        command_name = 'bdist_wheel'
        user_options = bdist_wheel.user_options

        def finalize_options(self):
            bdist_wheel.finalize_options(self)
            self.root_is_pure = False

        def get_tag(self):
            python, abi, plat = bdist_wheel.get_tag(self)
            python, abi = 'py2.py3', 'none'
            return python, abi, plat

except ImportError:
    BdistWheel = None

# allow setup.py to be run from any path
os.chdir(os.path.normpath(os.path.join(os.path.abspath(__file__), os.pardir)))


class Publish(Command):
    command_name = 'publish'
    user_options = [
        ('wheel', None, 'Publish the wheel'),
        ('sdist', None, 'Publish the sdist tar'),
        ('no-clean', None, 'Don\'t clean the build artifacts'),
        ('sign', None, 'Sign the artifacts using GPG')
    ]
    boolean_options = ['wheel', 'sdist']

    def initialize_options(self):
        self.wheel = False
        self.sdist = False
        self.no_clean = False
        self.sign = False

    def finalize_options(self):
        if not (self.wheel or self.sdist):
            self.announce('Either --wheel and/or --sdist must be provided', ERROR)
            sys.exit(1)

    def run(self):
        if os.system('pip freeze | grep twine'):
            self.announce('twine not installed.\nUse `pip install twine`.\nExiting.', WARN)
            sys.exit(1)

        if self.sdist:
            os.system('python setup.py sdist')

        if self.wheel:
            os.system('python setup.py bdist_wheel')

        if self.sign:
            for p in glob.glob('dist/*'):
                os.system('gpg --detach-sign -a {}'.format(p))

        os.system('twine upload dist/*')
        print('You probably want to also tag the version now:')
        print('  git tag -a {v} -m \'version {v}\''.format(v=version))
        print('  git push --tags')

        if not self.no_clean:
            shutil.rmtree('dist')
            shutil.rmtree('build')
            shutil.rmtree('oasislmf.egg-info')


setup(
    name='oasislmf',
    version=version,
    packages=find_packages(exclude=('tests', 'tests.*', 'tests.*.*')),
    include_package_data=True,
    package_data={
        '': [
            'requirements-package.in',
            'LICENSE',
        ],
        'oasislmf/_data/': ['*']
    },
    exclude_package_data={
        '': ['__pycache__', '*.py[co]'],
    },
    scripts=['bin/oasislmf'],
    license='BSD 3-Clause',
    description='Core loss modelling framework.',
    long_description=readme,
    url='https://github.com/OasisLMF/oasislmf',
    author='Dan Bate,Sandeep Murthy',
    author_email="Dan Bate <dan.bate@wildfish.com>,Sandeep Murthy <sandeep.murthy@oasislmf.org>",
    keywords='oasis lmf loss modeling framework',
    install_requires=reqs,
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    cmdclass={
        'install': PostInstallKtools,
        'bdist_wheel': BdistWheel,
        'publish': Publish,
    },
)
