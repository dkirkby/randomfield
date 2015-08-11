# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import print_function, division

import numpy as np

from transform import expanded_shape


def fill_with_ksquared(data, spacing, packed=True):
    """
    Fill an array with wavenumber squared magnitudes |k|**2.
    """
    nx, ny, nz = expanded_shape(data, packed=packed)
    lambda0 = spacing / (2 * np.pi)
    kx = np.fft.fftfreq(nx, lambda0)
    ky = np.fft.fftfreq(ny, lambda0)
    kz = np.fft.fftfreq(nz, lambda0)
    if packed:
        kz = kz[:nz//2 + 1]
    kx2_grid, ky2_grid, kz2_grid = np.meshgrid(
        kx**2, ky**2, kz**2, sparse=True, indexing='ij')
    np.add(kx2_grid, ky2_grid, out=data)
    np.add(data, kz2_grid, out=data)
    return data


def get_k_bounds(data, spacing, packed=True):
    """
    Return the bounds of |k|^2 values for the specified grid.
    """
    nx, ny, nz = expanded_shape(data, packed=packed)
    k0 = (2 * np.pi) / spacing
    k_min = k0 / max(nx, ny, nz)
    k_max = k0 * np.sqrt(3) / 2
    return k_min, k_max
