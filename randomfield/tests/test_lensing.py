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
    assert np.allclose(weights_by_h * cosmo.h**2, weights)


def test_variances_scaling_by_h():
    ell = np.logspace(1., 3., 21)
    z = np.linspace(0.5, 2.5, 101)
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


def test_shear_power():
    ell = np.logspace(1., 3., 3)
    z = np.linspace(0.1, 1.0, 91)
    # Calculate distances, growth and power in Mpc/h units.
    cosmo = create_cosmology()
    DC = cosmo.comoving_distance(z).to(u.Mpc).value * cosmo.h
    DA = cosmo.comoving_transverse_distance(z).to(u.Mpc).value * cosmo.h
    growth = get_growth_function(cosmo, z)
    power = load_default_power(scaled_by_h=True)
    # Calculate the weights and variances needed for the convolution integral.
    weights = calculate_lensing_weights(cosmo, z, DC, DA, scaled_by_h=True)
    variances = tabulate_3D_variances(ell, DA, growth, power)
    # Do the convolution integral.
    shear_power = calculate_shear_power(DC, DA, weights, variances)
    # Check the shear power for zsrc = 1 and ell=100.
    assert z[90] == 1.0
    assert ell[1] == 100.0
    assert abs(shear_power[90, 1] - 1.32737339588e-05) < 1e-6
    # Calculate the shear-shear cross power.
    cross_power = calculate_shear_power(DC, DA, weights, variances,
                                        mode='shear-shear-cross')
    # Check that the diagonal cross power equals the auto power.
    print(cross_power.shape, np.diagonal(cross_power).shape, shear_power.shape)
    assert np.array_equal(np.diagonal(cross_power).T, shear_power)
