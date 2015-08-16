# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import print_function, division

from astropy.tests.helper import pytest
from ..powertools import *
from ..transform import Plan
import numpy as np


spacing = 2.5
nx, ny, nz = 4, 6, 8


def test_fill():
    kx0 = 2 * np.pi / (spacing * nx)
    ky0 = 2 * np.pi / (spacing * ny)
    kz0 = 2 * np.pi / (spacing * nz)
    for packed in (True, False):
        plan = Plan(shape=(nx, ny, nz), dtype_in=np.complex64, packed=packed)
        fill_with_log10k(plan.data_in, spacing=spacing, packed=packed)
        for ix in range(nx):
            jx = ix if ix <= nx//2 else ix - nx
            for iy in range(ny):
                jy = iy if iy <= ny//2 else iy - ny
                for iz in range(nz):
                    if packed and iz > nz//2:
                        continue
                    jz = iz if iz <= nz//2 else iz - nz
                    ksq = (jx * kx0)**2 + (jy * ky0)**2 + (jz * kz0)**2
                    if ix == 0 and iy == 0 and iz == 0:
                        assert np.isinf(plan.data_in[ix, iy, iz])
                    else:
                        log10k = np.log10(np.sqrt(ksq))
                        assert abs(plan.data_in[ix, iy, iz] - log10k) < 1e-6


def test_bounds():
    plan = Plan(shape=(nx, ny, nz), dtype_in=np.complex64)
    fill_with_log10k(plan.data_in, spacing=spacing)
    # Find the limits by brute force.
    kmax1 = 10**np.max(plan.data_in)
    print(plan.data_in)
    assert np.isinf(plan.data_in[0, 0, 0])
    plan.data_in[0, 0, 0] = np.log10(kmax1)
    kmin1 = 10**np.min(plan.data_in)
    print(kmin1, kmax1)
    kmin2, kmax2 = get_k_bounds(plan.data_in, spacing=spacing)
    assert abs((kmin1 - kmin2)/kmin1) < 1e-7
    assert abs((kmax1 - kmax2)/kmax1) < 1e-7


def test_valid_power():
    power = np.zeros((10,), [('k', float), ('Pk', float)])
    power['k'] = np.arange(1,11)
    validate_power(power)


def test_invalid_power():
    with pytest.raises(ValueError):
        power = np.zeros((10,), [('k', float), ('Pk', float)])
        validate_power(power)
    with pytest.raises(ValueError):
        power = np.ones((10,), [('k', float), ('Pk', float)])
        validate_power(power)
    with pytest.raises(ValueError):
        power = np.zeros((10,), [('bad', float), ('Pk', float)])
        power['k'] = np.arange(1,11)
        validate_power(power)
    with pytest.raises(ValueError):
        power = np.zeros((10,), [('k', float), ('bad', float)])
        power['k'] = np.arange(1,11)
        validate_power(power)
    with pytest.raises(ValueError):
        power = np.zeros((10,), [('k', float), ('Pk', float)])
        power['k'] = np.arange(1,11)
        power['k'][-1] = np.inf
        validate_power(power)
    with pytest.raises(ValueError):
        power = np.zeros((10,), [('k', float), ('Pk', float)])
        power['k'] = np.arange(1,11)
        power['Pk'][0] = np.nan
        validate_power(power)
    with pytest.raises(ValueError):
        power = np.zeros((10,), [('k', float), ('Pk', float)])
        power['k'] = np.arange(10,0,-1)
        validate_power(power)
    with pytest.raises(ValueError):
        power = np.zeros((10,), [('k', float), ('Pk', float)])
        power['Pk'] = -1
        validate_power(power)


def test_validate_default_power():
    default_power = load_default_power()
    validate_power(default_power)


def test_tabulate_sigmas():
    # Verify equation (60) of http://arxiv.org/abs/astro-ph/0506540
    N3 = nx * ny * nz
    Vbox = N3 * spacing**3
    power = load_default_power()
    Pk = scipy.interpolate.interp1d(power['k'], power['Pk'])
    kx0 = 2 * np.pi / (spacing * nx)
    ky0 = 2 * np.pi / (spacing * ny)
    kz0 = 2 * np.pi / (spacing * nz)
    for packed in (True, False):
        plan = Plan(shape=(nx, ny, nz), dtype_in=np.complex64, packed=packed)
        fill_with_log10k(plan.data_in, spacing=spacing, packed=packed)
        tabulate_sigmas(plan.data_in, power, spacing, packed=packed)
        assert plan.data_in[0, 0, 0] == 0
        for ix in range(nx):
            jx = ix if ix <= nx//2 else ix - nx
            for iy in range(ny):
                jy = iy if iy <= ny//2 else iy - ny
                for iz in range(nz):
                    if packed and iz > nz//2:
                        continue
                    jz = iz if iz <= nz//2 else iz - nz
                    ksq = (jx * kx0)**2 + (jy * ky0)**2 + (jz * kz0)**2
                    if ix == 0 and iy == 0 and iz == 0:
                        continue
                    k = np.sqrt(ksq)
                    sigma = N3 * np.sqrt(Pk(k)/(2 * Vbox))
                    # Match with a loose tolerance since we are using a
                    # different interpolation scheme to calculate sigma here.
                    assert abs(plan.data_in[ix, iy, iz] - sigma) < 1e-3 * sigma
