Installation
============

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

Install the `pyfftw package <http://hgomersall.github.io/pyFFTW/>`_::

    pip install pyfftw
