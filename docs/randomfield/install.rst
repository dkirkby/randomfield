Installation
============

Install the latest stable release using::

    pip install randomfield

Alternatively, you can install the latest developer version from github::

    github clone https://github.com/dkirkby/randomfield.git
    cd randomfield
    python setup.py install

Optional Dependency: matplotlib
-------------------------------

The ``matplotlib`` package enables optional plotting capabilities and should
already be installed if you are using the recommended `anaconda scientific
python distribution <https://store.continuum.io/cshop/anaconda/>`__.

Optional Dependency: classy
---------------------------

The ``classy`` package provides python bindings to the C++ `CLASS package
<http://class-code.net>`__ and enables power spectra to be calculated
for arbitrary cosmologies.  If you do not install ``classy``, you can still
generate fields for arbitrary cosmology using an external program such as
`CAMB <http://camb.info>`__ to write tabulated power spectra to files.

Optional Dependency: pyFFTW
---------------------------

The `pyFFTW package <http://hgomersall.github.io/pyFFTW/index.html>`_ provides a wrapper around the `FFTW C library <http://www.fftw.org/>`_.  Installation of
pyFFTW is straightforward but requires that multiple versions of the FFTW library are already installed.

Install the float/double/quad and single/threaded versions of the FFTW library using each of the following configurations::

    # Single precision
    ./configure --enable-sse --enable-float --enable-shared
    ./configure --enable-sse --enable-float --enable-shared --enable-threads

    # Double precision
    ./configure --enable-sse2 --enable-shared
    ./configure --enable-sse2 --enable-shared --enable-threads

    # Long-double precision
    ./configure --enable-long-double --enable-shared
    ./configure --enable-long-double --enable-shared --enable-threads

After entering each `configure` command above, build and install the corresponding library with::

    make
    sudo make install

Install the `pyFFTW package`_ using::

    pip install pyfftw
