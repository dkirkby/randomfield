# Licensed under a 3-clause BSD style license - see LICENSE.rst

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
