Installation
============

Dependencies
------------

-----
Linux
-----

* **Debian**: g++ compiler build-essential, libtool, zlib1g-dev autoconf on debian distros
* **Red Hat**: 'Development Tools' and zlib-devel

-------
Windows
-------

Cygwin 64-bit is required for the Windows native build.  Cygwin is a Linux environment running in Windows.
http://www.cygwin.com/

Download and run the set-up program for Cygwin.
The following Cygwin add-in packages are required;

* gcc-g++
* gcc-core
* make
* diffutils
* automake
* libtools
* zlib-devel
* git


To build native Windows 64-bit executables;

* mingw64-x86_64-gcc-g++
* mingw64-x86_64-gcc-core
* mingw64-x86_64-zlib

Search for 'mingw', gcc', 'make' and 'diffutils' to find all of the relevant packages (Only 'gcc' illustrated below).
![alt text](docs/img/cygwin1.jpg "Add-in packages")

Install With Pip
----------------

The latest released version of the package can be installed using pip
by running::

    pip install oasislmf

Alternatively you can install the latest development version using::

    pip install -s git+https://git@github.com/OasisLMF/OasisLMF
