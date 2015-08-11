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
