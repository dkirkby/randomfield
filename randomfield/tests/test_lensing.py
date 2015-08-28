# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import print_function, division

from astropy.tests.helper import pytest
from ..lensing import *
import numpy as np

from astropy.cosmology import Planck13


def test_weights_scaling_by_h():
    z = np.linspace(0.0, 2.5, 26)
    weights = calculate_lensing_weights(Planck13, z, scaled_by_h=False)
    weights_by_h = calculate_lensing_weights(Planck13, z, scaled_by_h=True)
    assert np.allclose(weights_by_h * Planck13.h, weights)
