# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import print_function, division

from astropy.tests.helper import pytest
from ..cosmology import *
from ..powertools import load_default_power
import numpy as np


def test_class_setup():
    cosmology = astropy.cosmology.Planck13
    assert cosmology.Om0 == cosmology.Odm0 + cosmology.Ob0
    assert 1 == (cosmology.Om0 + cosmology.Ode0 + cosmology.Ok0 +
                 cosmology.Ogamma0 + cosmology.Onu0)
    class_parameters = get_class_parameters(cosmology)
    try:
        from classy import Class
        cosmo = Class()
        cosmo.set(class_parameters)
        cosmo.compute()
        assert cosmo.h() == cosmology.h
        assert cosmo.T_cmb() == cosmology.Tcmb0.value
        assert cosmo.Omega_b() == cosmology.Ob0
        # Calculate Omega(CDM)_0 two ways:
        assert abs((cosmo.Omega_m() - cosmo.Omega_b()) -
                   (cosmology.Odm0 - cosmology.Onu0)) < 1e-8
        assert abs(cosmo.Omega_m() - (cosmology.Om0 - cosmology.Onu0)) < 1e-8
        # CLASS calculates Omega_Lambda itself so this is a non-trivial test.
        calculated_Ode0 = cosmo.get_current_derived_parameters(
            ['Omega_Lambda'])['Omega_Lambda']
        assert abs(calculated_Ode0 - (cosmology.Ode0 + cosmology.Onu0)) < 1e-5
        cosmo.struct_cleanup()
        cosmo.empty()
    except ImportError:
        pass


def test_default_power():
    default_power = load_default_power()
    assert default_power.dtype == [('k', float), ('Pk', float)]
    """
    try:
        # Re-calculate the default power if CLASS is installed.
        # This is relatively slow, but its useful to track any changes
        # to what CLASS calculates given the same input configuration.
        import classy
        k_min, k_max = default_power['k'][[0,-1]]
        calculated_power = calculate_power(k_min, k_max)
        assert np.allclose(calculated_power['k'], default_power['k'])
        assert np.allclose(calculated_power['Pk'], default_power['Pk'])
    except ImportError:
        pass
    """
