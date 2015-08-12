# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import print_function, division

from astropy.tests.helper import pytest
from ..power import *
from ..transform import Plan
import numpy as np


spacing = 2.5
nx, ny, nz = 4, 6, 8


def test_fill():
    kx0 = 2 * np.pi / (spacing * nx)
    ky0 = 2 * np.pi / (spacing * ny)
    kz0 = 2 * np.pi / (spacing * nz)
    for packed in (True, False):
        plan = Plan(shape=(nx, ny, nz), packed=packed)
        fill_with_ksquared(plan.data_in, spacing=spacing, packed=packed)
        for ix in range(nx):
            jx = ix if ix <= nx//2 else ix - nx
            for iy in range(ny):
                jy = iy if iy <= ny//2 else iy - ny
                for iz in range(nz):
                    if packed and iz > nz//2:
                        continue
                    jz = iz if iz <= nz//2 else iz - nz
                    ksq = (jx * kx0)**2 + (jy * ky0)**2 + (jz * kz0)**2
                    assert abs(plan.data_in[ix, iy, iz] - ksq) < 1e-6


def test_bounds():
    plan = Plan(shape=(nx, ny, nz))
    fill_with_ksquared(plan.data_in, spacing=spacing)
    # Find the limits by brute force.
    kmax1 = np.sqrt(np.max(plan.data_in))
    assert plan.data_in[0, 0, 0] == 0
    plan.data_in[0, 0, 0] = kmax1**2
    kmin1 = np.sqrt(np.min(plan.data_in))
    kmin2, kmax2 = get_k_bounds(plan.data_in, spacing=spacing)
    assert abs((kmin1 - kmin2)/kmin1) < 1e-7
    assert abs((kmax1 - kmax2)/kmax1) < 1e-7


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
    try:
        # Re-calculate the default power if CLASS is installed.
        import classy
        k_min, k_max = default_power['k'][[0,-1]]
        calculated_power = calculate_power(k_min, k_max)
        assert np.allclose(calculated_power['k'], default_power['k'])
        assert np.allclose(calculated_power['Pk'], default_power['Pk'])
    except ImportError:
        pass
