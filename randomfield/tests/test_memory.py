# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import print_function, division

from astropy.tests.helper import pytest
from ..memory import *
import numpy as np


def test_1d():
    for use_pyfftw in (False, True):
        buf = allocate(10, dtype=np.float32, use_pyfftw=use_pyfftw)
        assert buf.shape == (10,)
        assert buf.dtype == np.float32
        if use_pyfftw:
            try:
                import pyfftw
                assert pyfftw.n_byte_align(buf, 16) is buf
            except ImportError:
                pass


def test_3d():
    for use_pyfftw in (False, True):
        buf = allocate((4, 6, 8), dtype=np.complex64, use_pyfftw=use_pyfftw)
        assert buf.shape == (4, 6, 8)
        assert buf.dtype == np.complex64
        if use_pyfftw:
            try:
                import pyfftw
                assert pyfftw.n_byte_align(buf, 16) is buf
            except ImportError:
                pass


def test_c2r_view():
    for use_pyfftw in (False, True):
        buf1 = allocate((4, 6, 5), dtype=np.complex64, use_pyfftw=use_pyfftw)
        buf2 = buf1.view(np.float32).reshape(4, 6, 10)[:, :, :8]
        assert (buf2.base is buf1) or (buf2.base is buf1.base)


def test_r2c_view():
    for use_pyfftw in (False, True):
        buf1 = allocate((4, 6, 10), dtype=np.float32, use_pyfftw=use_pyfftw)
        buf2 = buf1.view(np.complex64).reshape(4, 6, 5)
        buf1 = buf1[:, :, :8]
        assert (buf2.base is buf1) or (buf2.base is buf1.base)
