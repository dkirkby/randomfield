# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import print_function, division

from astropy.tests.helper import pytest
from ..lensing import *
import numpy as np

import astropy.units as u

from ..cosmotools import create_cosmology, get_growth_function
from ..powertools import load_default_power


def test_weights_scaling_by_h():
    z = np.linspace(0.0, 2.5, 26)
    cosmo = create_cosmology()
    weights = calculate_lensing_weights(cosmo, z, scaled_by_h=False)
    weights_by_h = calculate_lensing_weights(cosmo, z, scaled_by_h=True)
    assert np.allclose(weights_by_h * cosmo.h, weights)


def test_variances_scaling_by_h():
    ell = np.logspace(1., 3., 20)
    z = np.linspace(0.5, 2.5, 100)
    cosmo = create_cosmology()
    growth = get_growth_function(cosmo, z)
    power = load_default_power()
    # Use Mpc units.
    DA = cosmo.comoving_transverse_distance(z).to(u.Mpc).value
    variances = tabulate_3D_variances(ell, DA, growth, power)
    # Use Mpc/h units.
    DA_by_h = DA * cosmo.h
    power_by_h = np.copy(power)
    power_by_h['k'] /= cosmo.h
    power_by_h['Pk'] *= cosmo.h**3
    variances_by_h = tabulate_3D_variances(ell, DA_by_h, growth, power_by_h)
    assert np.allclose(variances_by_h, variances)
