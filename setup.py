from setuptools import setup

try:
    from Cython.Build import cythonize
    import numpy as np
    ext_modules = cythonize(
        'oasislmf/pytools/common/_write_csv_cython.pyx',
        compiler_directives={'language_level': '3'},
    )
    include_dirs = [np.get_include()]
except ImportError:
    ext_modules = []
    include_dirs = []

setup(
    scripts=['bin/completer_oasislmf', 'bin/oasis_exec_monitor.sh'],
    ext_modules=ext_modules,
    include_dirs=include_dirs,
)
