
# Installation for Mac ARM64

When it comes to installing OasisLMF on Mac ARM64, the pre-complied binaries are not available. Therefore, the 
installation process is a bit more complicated and the compilation needs to be local. The following steps are required 
to install OasisLMF on Mac ARM64:

First we clone the [ktools](https://github.com/OasisLMF/ktools) repo with the following command:

```bash
git clone https://github.com/OasisLMF/ktools.git
```

Before we run anything, we need to ensure that the following dependencies are installed:

```bash
brew install \
    autoconf \
    automake \
    libtool \
    zlib-ng
```

We then move into the ktools directory with `cd ktools` and run the following commands:

```bash
./autogen.sh
```

We then configure the ktools with the following command:

```bash
./configure --enable-osx --enable-o3 --prefix=<BIN_PATH>
```
We could use something like ```$HOME/ktools/bin``` as the ```<BIN_PATH>```.
We then run the checks and installation with the following command:

```bash
make check
make install
```

We then have to package our compiled binary files with the command below:

```bash
tar -zcvf ../../<OS_PLATFORM>.tar.gz ./
```

We then move back to the root directory (`cd ..`) and clone the [oasislmf](https://github.com/OasisLMF/OasisLMF) 
repository, move into the oasislmf directory and run the following commands:

```bash
git clone https://github.com/OasisLMF/OasisLMF.git
cd OasisLMF
pip install pip-tools
```

We are now ready to install the oasislmf package with the following command:

```bash
export OASISLMF_KTOOLS_BIN_PATH=<BIN_PATH>
python setup.py install bdist_wheel --plat-name <OS_PLATFORM>
```

We then have to ensure that the required packages are installed with the following command:

```bash
pip install -r requirements-package.in
```

If we want to install the extras needed we can run the following command:

```bash
pip install -r optional-package.in
```

## Tool Development

We aim to build tools that automate such processes. We are currently developing the 
[water seller](https://github.com/OasisLMF/water_seller) tool that automates the tasks. 
