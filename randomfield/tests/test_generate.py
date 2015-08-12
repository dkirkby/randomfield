# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import print_function, division

from astropy.tests.helper import pytest
from ..generate import *
import numpy as np

from ..power import load_default_power


def test_generate():
    spacing = 2.5
    nx, ny, nz = 64, 64, 64
    power = load_default_power()
    data = generate(nx, ny, nz, spacing, power, seed=123)
    assert data.shape == (nx, ny, nz)
    assert data.dtype == np.float32
    assert abs(np.mean(data)) < 1e-3
