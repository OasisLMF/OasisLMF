Sphinx Docs
===========

This repository is enabled with Sphinx documentation for the Python
modules, and the documentation is published to
`https://oasislmf.github.io/oasislmf <https://oasislmf.github.io/oasislmf>`_
automatically via GitHub pages on updates to the GitHub repository.

Setting up Sphinx
-----------------

To work on the Sphinx docs for this package you must have Sphinx
installed on your system or in your virtual environment (``virtualenv``
is recommended).

Building and publishing
-----------------------

The Sphinx documentation source files are reStructuredText files, and
are contained in the ``docs`` subfolder, which also contains the ``Makefile``
for the build. To do a new build make sure you are in the ``docs`` subfolder
and run

::

    make html

You should see a new set of HTML files and assets in the ``_build/html``
subfolder (the build directory can be changed to ``docs`` itself in the
``Makefile`` but that is not recommended). The ``docs`` subfolder should
always contain the latest copy of the built HTML and assets so first
copy the files from ``_build/html`` to ``docs`` using

::

    cp -R _build/html/* .

Add and commit these files to the local repository, and then update the
remote repository on GitHub - GitHub pages will automatically publish
the new documents to `https://oasislmf.github.io/oasislmf <https://oasislmf.github.io/oasislmf>`_.