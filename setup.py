import glob
import os
import io
import re
import shutil
import sys
import platform
import tarfile
from contextlib import contextmanager
from distutils.log import INFO, WARN, ERROR
from distutils.spawn import find_executable
from platform import machine, system
from tempfile import mkdtemp
from time import sleep

from setuptools import find_packages, setup, Command
from setuptools.command.install import install
from setuptools.command.develop import develop

try:
    from urllib import request as urlrequest
    from urllib.error import URLError
except ImportError:
    from urllib2 import urlopen, URLError


KTOOLS_VERSION = '3.9.7'

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))


def get_readme():
    with io.open(os.path.join(SCRIPT_DIR, 'README.md'), encoding='utf-8') as readme:
        return readme.read()


def get_install_requirements():
    with io.open(os.path.join(SCRIPT_DIR, 'requirements-package.in'), encoding='utf-8') as reqs:
        return reqs.readlines()

def get_optional_requirements():
    with io.open(os.path.join(SCRIPT_DIR, 'optional-package.in'), encoding='utf-8') as reqs:
        return {"extra": reqs.readlines()}

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
reqs_extra = get_optional_requirements()
readme = get_readme()


class InstallKtoolsMixin(object):
    def __init__(self):
        self.ktools_components = []

    def install_ktools(self):
        '''
        If system arch matches Ktools static build try to install from pre-build
        with a fallback of compile ktools from source
        '''
        bin_install_kwargs = self.try_get_bin_install_kwargs()


        if bin_install_kwargs:
            # This only executes if 'KTOOLS_TAR_FILE_DIR' is set and the directory contains a correctly named ktools tar
            if os.path.isfile(bin_install_kwargs.get('ktools_tar_override', '')):
                self.install_ktools_local(**bin_install_kwargs)
                return

            try:
                self.install_ktools_bin(**bin_install_kwargs)
            except:
                print('Fallback - building ktools from source')
                self.install_ktools_source(**bin_install_kwargs)
        else:
            self.install_ktools_source()

    def fetch_ktools_tar(self, location, url, attempts=3, timeout=15, cooldown=1):
        last_error = None
        proxy_config = urlrequest.getproxies()
        self.announce('Retrieving ktools from: {}'.format(url), INFO)
        self.announce('Proxy configuration: {}'.format(proxy_config), INFO)

        if proxy_config:
            # Handle Proxy config
            proxy_handler = urlrequest.ProxyHandler(proxy_config)
            opener = urlrequest.build_opener(proxy_handler)
            urlrequest.install_opener(opener)

        for i in range(attempts):
            try:
                if proxy_config:
                    # Proxied connection
                    req = urlrequest.urlopen(urlrequest.Request(url), timeout=timeout)
                    break
                else:
                    # Non proxied connection
                    req = urlrequest.urlopen(url, timeout=timeout)
                    break

            except URLError as e:
                self.announce('Fetch ktools tar failed: {} (attempt {})'.format(e, (i+1)), WARN)
                last_error = e
                sleep(cooldown)
        else:
            self.announce('Failed to get ktools tar after {} attempts'.format(attempts), ERROR)
            if last_error:
                raise last_error

        with open(location, 'wb') as f:
            f.write(req.read())

    def unpack_tar(self, tar_location, extract_location):
        self.announce('Unpacking ktools', INFO)
        with tarfile.open(tar_location) as tar:
            if not os.path.exists(extract_location):
                os.makedirs(extract_location)
            tar.extractall(extract_location)

    def ktools_inpath(self):
        ktools_bin_subset = [
            'eve',
            'getmodel',
            'gulcalc',
            'fmcalc',
            'summarycalc',
        ]
        for ktools_bin in ktools_bin_subset:
            if find_executable(ktools_bin) is None:
                return False
        return True

    def build_ktools(self, extract_location, system_os):
        self.announce('Building ktools', INFO)
        print('Installing ktools from source')
        print(f' :::::  system_os {system_os} :::::')

        build_dir = os.path.join(extract_location, 'ktools-{}'.format(KTOOLS_VERSION))

        system_os_flag = '--enable-osx ' if system_os == 'Darwin' else ''

        exit_code = os.system(f'cd {build_dir} && ./autogen.sh && ./configure {system_os_flag} && make && make check')
        if(exit_code != 0):
            print('Ktools build failed.\n')
            sys.exit(1)
        return build_dir

    def add_ktools_build_to_path(self, build_dir):

        if not os.path.exists(self.get_bin_dir()):
            os.makedirs(self.get_bin_dir())

        for p in glob.glob(os.path.join(build_dir, 'src', '*', '*')):
            split = p.split(os.path.sep)

            # if the file name is the same as the directory we have found a
            # component executable
            if split[-1] == split[-2]:
                component_path = os.path.join(self.get_bin_dir(), split[-1])
                shutil.copy(p, component_path)
                yield component_path

    def add_ktools_bins_to_path(self, extract_path):
        print('Installing ktools from pre-built binaries')

        if not os.path.exists(self.get_bin_dir()):
            os.makedirs(self.get_bin_dir())

        for p in glob.glob(os.path.join(extract_path, '*')):
            split = p.split(os.path.sep)
            component_path = os.path.join(self.get_bin_dir(), split[-1])
            shutil.copy(p, component_path)
            yield component_path

    def try_get_bin_install_kwargs(self):
        if '--plat-name' in sys.argv:
            PLATFORM = sys.argv[sys.argv.index('--plat-name') + 1]
            OS, ARCH = PLATFORM.split('_', 1)
        else:
            try:
                ARCH = platform.machine()
                OS = platform.system()
            except Exception:
                ARCH = None
                OS = None

        # ENV OVERRIDE TO install a localy copy of ktools
        # TAR_DIR = os.path.abspath(os.getenv('KTOOLS_TAR_FILE_DIR', ''))

        if os.getenv('KTOOLS_TAR_FILE_DIR', None):
            TAR_OVERRIDE = os.path.join(os.path.abspath(os.getenv('KTOOLS_TAR_FILE_DIR')), '{}_{}.tar.gz'.format(OS, ARCH))
            return {"system_os": OS, "system_architecture": ARCH, 'ktools_tar_override': TAR_OVERRIDE}
        else:    
            return {"system_os": OS, "system_architecture": ARCH}

    def install_ktools_source(self, system_os=None, system_architecture=None, ktools_tar_override=None):
        with temp_dir() as d:
            local_tar_path = os.path.join(d, 'ktools.tar.gz')
            local_extract_path = os.path.join(d, 'extracted')
            source_url = 'https://github.com/OasisLMF/ktools/archive/v{}.tar.gz'.format(KTOOLS_VERSION)

            self.fetch_ktools_tar(local_tar_path, source_url)
            self.unpack_tar(local_tar_path, local_extract_path)
            build_dir = self.build_ktools(local_extract_path, system_os)
            self.ktools_components = list(self.add_ktools_build_to_path(build_dir))

    def install_ktools_bin(self, system_os, system_architecture, ktools_tar_override=None):
        with temp_dir() as d:
            local_tar_path = os.path.join(d, '{}_{}.tar.gz'.format(system_os, system_architecture))
            local_extract_path = os.path.join(d, 'extracted')
            bin_url = 'https://github.com/OasisLMF/ktools/releases/download/v{}/{}_{}.tar.gz'.format(KTOOLS_VERSION, system_os, system_architecture)
            self.fetch_ktools_tar(local_tar_path, bin_url)
            
            self.unpack_tar(local_tar_path, local_extract_path)
            self.ktools_components = list(self.add_ktools_bins_to_path(local_extract_path))

    def install_ktools_local(self, system_os, system_architecture, ktools_tar_override):
        print(f"OVERRIDE: Ktools installation from local file: '{ktools_tar_override}'")
        with temp_dir() as d:
            local_extract_path = os.path.join(d, 'extracted')
            self.unpack_tar(ktools_tar_override, local_extract_path)
            self.ktools_components = list(self.add_ktools_bins_to_path(local_extract_path))


class PostInstallKtools(InstallKtoolsMixin, install):
    command_name = 'install'
    user_options = install.user_options + [
        ('ktools', None, 'Only install ktools components'),
    ]
    boolean_options = install.boolean_options + ['ktools']

    def __init__(self, *args, **kwargs):
        install.__init__(self, *args, **kwargs)

    def run(self):
        self.install_ktools()
        install.run(self)

    def get_outputs(self):
        return install.get_outputs(self) + self.ktools_components

    def get_bin_dir(self):
        return self.install_scripts


class PostDevelopKtools(InstallKtoolsMixin, develop):
    command_name = 'develop'
    user_options = develop.user_options
    boolean_options = develop.boolean_options

    def __init__(self, *args, **kwargs):
        develop.__init__(self, *args, **kwargs)

    def run(self):
        self.install_ktools()
        develop.run(self)

    def get_outputs(self):
        return develop.get_outputs(self) + self.ktools_components

    def get_bin_dir(self):
        return self.script_dir


try:
    from wheel.bdist_wheel import bdist_wheel

    # https://github.com/pypa/wheel/blob/master/wheel/bdist_wheel.py#L43
    class BdistWheel(bdist_wheel):
        command_name = 'bdist_wheel'
        user_options = bdist_wheel.user_options

        def initialize_options(self):
            super(BdistWheel, self).initialize_options()

        def finalize_options(self):
            bdist_wheel.finalize_options(self)
            self.root_is_pure = False

        def get_tag(self):
            python, abi, plat = bdist_wheel.get_tag(self)
            python, abi = 'py3', 'none'
            plat = plat.lower().replace('linux', 'manylinux1')
            plat = plat.lower().replace('darwin_x86_64', 'macosx_10_6_intel')
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
    scripts=['bin/oasislmf', 'bin/completer_oasislmf', 'bin/ktools_monitor.sh'],
    entry_points={
        'console_scripts': [
            'complex_itemtobin=oasislmf.execution.complex_items_to_bin:main',
            'complex_itemtocsv=oasislmf.execution.complex_items_to_csv:main',
            'load_balancer=oasislmf.execution.load_balancer:main',
            'fmpy=oasislmf.pytools.fmpy:main',
            'modelpy=oasislmf.pytools.modelpy:main',
            'footprintconvpy=oasislmf.pytools.footprintconv:footprintconvpy',
            'vulntoparquet=oasislmf.pytools.getmodel.vulnerability:main',
        ]
    },
    license='BSD 3-Clause',
    description='Core loss modelling framework.',
    long_description=readme,
    long_description_content_type='text/markdown',
    url='https://github.com/OasisLMF/oasislmf',
    author='Oasis LMF',
    author_email="support@oasislmf.org",
    keywords='oasis lmf loss modeling framework',
    python_requires='>=3.6',
    install_requires=reqs,
    extras_require=reqs_extra,
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.6',
    ],
    cmdclass={
        'install': PostInstallKtools,
        'develop': PostDevelopKtools,
        'bdist_wheel': BdistWheel,
        'publish': Publish,
    },
)
