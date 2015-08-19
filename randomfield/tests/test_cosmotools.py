# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import print_function, division

from astropy.tests.helper import pytest
from ..cosmotools import *
from ..powertools import load_default_power
import numpy as np


def test_create_cosmology():
    assert create_cosmology() is astropy.cosmology.Planck13
    assert create_cosmology('Planck13') is astropy.cosmology.Planck13
    assert create_cosmology(H0=65, Om0=0.25).h == 0.65
    with pytest.raises(TypeError):
        create_cosmology('Planck13', H0=65, Om0=0.25)
    with pytest.raises(TypeError):
        create_cosmology('Planck13', 'WMAP9')
    with pytest.raises(TypeError):
        create_cosmology(H0=65)
    with pytest.raises(TypeError):
        create_cosmology(Om0=0.25)
    with pytest.raises(TypeError):
        create_cosmology(H0=65, Om0=0.25, blah=123)


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


def test_redshifts():
    # Test redshifts calculated for a matter-only flat universe where
    # z = r*(4-r) / (r-2)**2 with r = Dc(z)/(c/H0)
    nz = 100
    spacing = 2.5

    data = np.empty((1, 2, nz))

    from astropy.constants import c
    model = create_cosmology(H0=70, Om0=1, Tcmb0=0)
    Dc0 = (c / model.H0).to(u.Mpc).value

    for scale in (1, model.h):
        r = np.arange(nz) * spacing / scale / Dc0
        analytic_redshifts = r * (4 - r) / (r - 2)**2
        computed_redshifts = get_redshifts(
            model, data, spacing=spacing, scaled_by_h=(scale != 1))
        assert np.allclose(analytic_redshifts, computed_redshifts)


def test_growth():
    # Test growth calculation for a matter-only flat universe where
    # G(z) = 1/(1 + z).
    nz = 100
    z = np.linspace(0., 1., nz).reshape(1, nz, 1)
    model = create_cosmology(H0=70, Om0=1, Tcmb0=0)
    Gz = get_growth_function(model, z)
    assert Gz.shape == (1, nz, 1)
    assert np.allclose(Gz, 1/(1 + z))


def test_lognormal():
    growth = 0.3
    sigma = 2.5
    np.random.seed(123)
    delta = np.empty((64, 64, 128), dtype=np.float32)
    delta[:] = sigma * np.random.normal(size=delta.shape)
    assert abs(np.mean(delta)) < 1e-2
    assert abs(np.std(delta) - sigma) < 1e-2 * sigma
    rho = apply_lognormal_transform(delta, growth, sigma=2.5)
    assert rho.shape == delta.shape
    assert rho.base is delta.base
    assert np.all(rho > 0)
    assert abs(np.mean(rho) - 1.) < 1e-3
    assert abs(np.std(rho) - growth * sigma) < 1e-2 * sigma


def test_default_power():
    default_power = load_default_power()
    assert default_power.dtype == [('k', float), ('Pk', float)]
    model = create_cosmology()
    """
    try:
        # Re-calculate the default power if CLASS is installed.
        # This is relatively slow, but its useful to track any changes
        # to what CLASS calculates given the same input configuration.
        import classy
        k_min, k_max = default_power['k'][[0,-1]]
        calculated_power = calculate_power(model, k_min, k_max)
        assert np.allclose(calculated_power['k'], default_power['k'])
        assert np.allclose(calculated_power['Pk'], default_power['Pk'])
    except ImportError:
        pass
    """
