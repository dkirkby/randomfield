# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import print_function, division

from astropy.tests.helper import pytest
from ..random import *
import numpy as np

seed = 123


def test_randomize():
    np.random.seed(seed)
    nx, ny, nz = 40, 60, 80
    sigma = 1 + np.arange(nx * ny * nz).reshape(nx, ny, nz)
    data = np.empty((nx, ny, nz), dtype=np.complex64)
    data.real = sigma
    randomize(data)
    data /= sigma
    assert abs(np.mean(data.real)) < 5e-3
    assert abs(np.mean(data.imag)) < 5e-3
    assert abs(np.std(data.real) - 1) < 5e-3
    assert abs(np.std(data.imag) - 1) < 5e-3
