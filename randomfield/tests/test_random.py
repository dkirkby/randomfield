# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import print_function, division

from astropy.tests.helper import pytest
from ..random import *
import numpy as np

seed = 123


def test_randomize():
    nx, ny, nz = 40, 60, 80
    sigma = 1 + np.arange(nx * ny * nz).reshape(nx, ny, nz)
    data = np.empty((nx, ny, nz), dtype=np.complex64)
    data.real = sigma
    randomize(data, seed)
    data /= sigma
    assert abs(np.mean(data.real)) < 5e-3
    assert abs(np.mean(data.imag)) < 5e-3
    assert abs(np.std(data.real) - 1) < 5e-3
    assert abs(np.std(data.imag) - 1) < 5e-3


def test_reproducible():
    nx, ny, nz = 4, 6, 8
    data1 = np.ones((nx, ny, nz), dtype=np.complex64)
    randomize(data1, seed)
    data2 = np.ones((nx, ny, nz), dtype=np.complex64)
    randomize(data2, seed)
    assert np.array_equal(data1, data2)
