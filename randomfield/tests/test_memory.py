# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy.tests.helper import pytest
from ..memory import allocate
import numpy as np

def test_1d():
    buf = allocate(10, dtype=np.float32)
    assert buf.shape == (10,)
    assert buf.dtype == np.float32

def test_3d():
    buf = allocate((4, 6, 8), dtype=np.complex64)
    assert buf.shape == (4, 6, 8)
    assert buf.dtype == np.complex64
