from __future__ import print_function, division

import glob
import json
import os
import io
import re
import shutil
import sys
import tarfile
from contextlib import contextmanager
from tempfile import mkdtemp

from setuptools import find_packages, setup, Extension
from setuptools.command.install import install

try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib import urlretrieve

# allow setup.py to be run from any path
os.chdir(os.path.normpath(os.path.join(os.path.abspath(__file__), os.pardir)))

KTOOLS_VERSION = '0_0_392_0'


def get_readme():
    with io.open(os.path.join(os.path.dirname(__file__), 'README.rst'), encoding='utf-8') as readme:
        return readme.read()


def get_install_requirements():
    with io.open(os.path.join(os.path.dirname(__file__), 'requirements-package.in'), encoding='utf-8') as reqs:
        return reqs.readlines()


def get_version():
    """
    Return package version as listed in `__version__` in `init.py`.
    """
    with io.open(os.path.join(os.path.dirname(__file__), 'oasislmf', '__init__.py'), encoding='utf-8') as init_py:
        return re.search('__version__ = [\'"]([^\'"]+)[\'"]', init_py.read()).group(1)


@contextmanager
def temp_dir():
    d = mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


version = get_version()

ktools_module = Extension('oasislmf.ktools', ['ktools/ktools.cpp', 'ktools/src/hello.cpp'])


if sys.argv[-1] == 'publish':
    if os.system('pip freeze | grep wheel'):
        print('wheel not installed.\nUse `pip install wheel`.\nExiting.')
        sys.exit()
    if os.system('pip freeze | grep twine'):
        print('twine not installed.\nUse `pip install twine`.\nExiting.')
        sys.exit()
    os.system('python setup.py sdist bdist_wheel')
    os.system('twine upload dist/*')
    print('You probably want to also tag the version now:')
    print('  git tag -a {v} -m \'version {v}\''.format(v=version))
    print('  git push --tags')
    shutil.rmtree('dist')
    shutil.rmtree('build')
    shutil.rmtree('oasis_lmf.egg-info')
    sys.exit()


class PostInstallKtools(install):
    command_name = 'install'
    user_options = install.user_options + [
        ('ktools', None, 'Only install ktools components'),
    ]
    boolean_options = install.boolean_options + ['ktools']

    def __init__(self, *args, **kwargs):
        self.ktools = False
        super(PostInstallKtools, self).__init__(*args, **kwargs)

    def run(self):
        if not self.ktools:
            super(PostInstallKtools, self).run()
        self.install_ktools()

    def fetch_ktools_tar(self, location):
        print('Retrieving ktools {}'.format(KTOOLS_VERSION))

        def _report(block_count, block_size, total_size, end='\r'):
            bar_size = 40
            filled = bar_size * (block_count * block_size) // total_size

            print('[{}{}]'.format('#' * filled, ' ' * (bar_size - filled)), end=end)

        urlretrieve('https://github.com/OasisLMF/ktools/archive/OASIS_{}.tar.gz'.format(KTOOLS_VERSION), location, _report)
        _report(1, 1, 1, end=' Done\n')

    def unpack_tar(self, tar_location, extract_location):
        print('Unpacking ktools')
        with tarfile.open(tar_location) as tar:
            os.mkdir(extract_location)
            tar.extractall(extract_location)

    def build_ktools(self, extract_location):
        print('Building ktools')
        build_dir = os.path.join(extract_location, 'ktools-OASIS_{}'.format(KTOOLS_VERSION))

        os.system('cd {build_dir} && ./autogen.sh && ./configure && make && make check'.format(build_dir=build_dir))
        return build_dir

    def add_ktools_to_path(self, build_dir):
        print('Installing ktools')
        components_file = os.path.join(self.install_scripts, '.ktools_components')
        components = set()

        # check if we already have any components installed so that they are not lost on future uninstall
        if os.path.exists(components_file):
            with open(components_file) as f:
                components = set(json.load(f))

        for p in glob.glob(os.path.join(build_dir, 'src', '*', '*')):
            split = p.split(os.path.sep)

            # if the file name is the same as the directory we have found a
            # component executable
            if split[-1] == split[-2]:
                shutil.copy(p, os.path.join(self.install_scripts, split[-1]))
                components.add(split[-1])

        # store all the installed components so that they can be uninstalled later
        with open(components_file, 'w') as f:
            json.dump(list(components), f)

    def install_ktools(self):
        with temp_dir() as d:
            local_tar_path = os.path.join(d, 'ktools.tar.gz')
            local_extract_path = os.path.join(d, 'extracted')

            self.fetch_ktools_tar(local_tar_path)
            self.unpack_tar(local_tar_path, local_extract_path)
            build_dir = self.build_ktools(local_extract_path)
            self.add_ktools_to_path(build_dir)


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
    long_description=get_readme(),
    url='https://github.com/OasisLMF/oasislmf',
    author='OasisLMF',
    author_email=' oasis@oasislmf.org',
    keywords='oasis lmf loss modeling framework',
    install_requires=get_install_requirements(),
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
    cmdclass={'install': PostInstallKtools},
)
