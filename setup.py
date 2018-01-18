import os
import re
import shutil
import sys
from setuptools import setup, find_packages

# allow setup.py to be run from any path
os.chdir(os.path.normpath(os.path.join(os.path.abspath(__file__), os.pardir)))


def get_readme():
    with open(os.path.join(os.path.dirname(__file__), 'README.rst')) as readme:
        return readme.read()


def get_install_requirements():
    with open(os.path.join(os.path.dirname(__file__), 'requirements-package.in')) as reqs:
        return [r for r in reqs.readlines() if not r.startswith('-e')]


def get_dependency_links():
    with open(os.path.join(os.path.dirname(__file__), 'requirements-package.in')) as reqs:
        return [r[3:] for r in reqs.readlines() if r.startswith('-e')]


def get_version():
    """
    Return package version as listed in `__version__` in `init.py`.
    """
    with open(os.path.join(os.path.dirname(__file__), 'oasislmf', '__init__.py')) as init_py:
        return re.search('__version__ = [\'"]([^\'"]+)[\'"]', init_py.read()).group(1)


version = get_version()

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

setup(
    name='oasislmf',
    version=version,
    packages=find_packages(),
    include_package_data=True,
    package_data={
        '': [
            'requirements-package.in',
            'LICENSE',
        ],
    },
    exclude_package_data={
        '': ['__pycache__', '*.py[co]'],
    },
    scripts=['bin/oasislmf-cli'],
    license='BSD 3-Clause',
    description='Core loss modelling framework.',
    long_description=get_readme(),
    url='https://github.com/OasisLMF/oasislmf',
    author='OasisLMF',
    author_email=' oasis@oasislmf.org',
    keywords='oasis lmf loss modeling framework',
    install_requires=get_install_requirements(),
    dependency_links=get_dependency_links(),
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
)
