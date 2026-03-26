from Cython.Build import cythonize
from setuptools import Extension, setup
import numpy as np

ext_modules = cythonize(
    Extension(
        'oasis_writecsv',
        ['oasislmf/pytools/common/oasis_writecsv.pyx'],
        include_dirs=[np.get_include()],
    ),
    compiler_directives={'language_level': '3'},
)

setup(
    scripts=['bin/completer_oasislmf', 'bin/oasis_exec_monitor.sh'],
    ext_modules=ext_modules,
)
