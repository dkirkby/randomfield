# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Utilities for working with tabulated power spectra.
"""
from __future__ import print_function, division

import inspect
import os.path

import numpy as np
import scipy.interpolate

import transform


def get_k_bounds(data, spacing, packed=True):
    """
    Return the bounds of wavenumber values for the specified grid.
    """
    nx, ny, nz = transform.expanded_shape(data, packed=packed)
    k0 = (2 * np.pi) / spacing
    k_min = k0 / max(nx, ny, nz)
    k_max = k0 * np.sqrt(3) / 2
    return k_min, k_max


def create_ksq_grids(data, spacing, packed):
    nx, ny, nz = transform.expanded_shape(data, packed=packed)
    lambda0 = spacing / (2 * np.pi)
    kx = np.fft.fftfreq(nx, lambda0)
    ky = np.fft.fftfreq(ny, lambda0)
    kz = np.fft.fftfreq(nz, lambda0)
    if packed:
        kz = kz[:nz//2 + 1]
    # With the sparse option, the memory usage of these grids is
    # O(nx+ny+nz) rather than O(nx*ny*nz).
    return np.meshgrid(kx**2, ky**2, kz**2, sparse=True, indexing='ij')


def fill_with_log10k(data, spacing, packed=True):
    """
    Fill an array with values of log10(k).

    Note that the value at [0, 0, 0] will be log10(0) = -inf.
    """
    kx2_grid, ky2_grid, kz2_grid = create_ksq_grids(data, spacing, packed)
    # Zero the imaginary components of data.
    data.imag = 0
    # Calculate in place: data = kx**2 + ky**2
    np.add(kx2_grid, ky2_grid, out=data.real)
    # Calculate in place: data = kx**2 + ky**2 + kz**2 = |k|**2
    np.add(data.real, kz2_grid, out=data.real)
    # Calculate in place: data = log10(|k|**2) = 2*log10(|k|)
    # Ignore the RuntimeWarning for log10(0) = -inf.
    old_settings = np.seterr(divide='ignore')
    np.log10(data.real, out=data.real)
    np.seterr(**old_settings)
    ##data[0, 0, 0] = -np.inf
    # Calculate in place: data = log10(|k|)
    data.real *= 0.5
    return data


def validate_power(power):
    """
    Validates a power spectrum.
    """
    if not isinstance(power, np.ndarray):
        raise ValueError('Invalid type for power: {0}.'.format(type(power)))
    if not 'k' in power.dtype.names or not 'Pk' in power.dtype.names:
        raise ValueError('Missing required fields "k", "Pk" in power.')
    if not np.all(np.isfinite(power['k'])):
        raise ValueError('Power spectrum has some invalid values of k.')
    if not np.all(np.isfinite(power['Pk'])):
        raise ValueError('Power spectrum has some invalid values of P(k).')
    if not np.array_equal(power['k'], np.unique(power['k'])):
        raise ValueError('Power spectrum k values are not strictly increasing.')
    if power['k'][0] <= 0:
        raise ValueError('Power spectrum min(k) is <= 0.')
    if np.any(power['Pk'] < 0):
        raise ValueError('Power values P(k) are not all non-negative.')
    return power


def filter_power(power, sigma, out=None):
    """
    Apply a Gaussian filtering to a power spectrum.

    The smoothing length sigma should be in the inverse units of wavenumbers in
    the input power['k'].  A delta field generated from a filtered P(k) is
    effectively convolved with a 3D Gaussian smoothing kernel with the
    specified sigma::

        exp( - |r|**2 / (2 * sigma**2)) / (2*pi)**(3/2) / sigma**3

    This is achieved by scaling each delta(k) by the factor::

        exp( - |k|**2 * sigma**2 / 2)

    or, equivalently, by using the filtered power spectrum::

        P(k) -> P(k) * exp( - |k|**2 * sigma**2)

    Note the factor of 2 difference in the last two expressions, since::

        P(k) ~ < delta(k)**2 >

    See section 5 of arXiv:astro-ph/0506540 for details.
    """
    if sigma < 0:
        raise ValueError('Invalid smoothing sigma: {0}.'.format(sigma))
    if out is None:
        out = np.copy(power)
    elif out is not power:
        validate_power(power)
        if out.shape != power.shape:
            raise ValueError(
                'Output power has wrong shape: {0}.'.format(out.shape))
        out[:] = power
    if sigma > 0:
        out['Pk'] *= np.exp(-(power['k'] * sigma)**2)
    return out


def tabulate_sigmas(data, power, spacing, packed=True):
    """
    Replace an array of log10(k) values with the corresponding sigmas.

    Note that the scaling from P(k) to variance depends on convention for
    normalizing the inverse FFT.  Since we divide the inverse FFT by
    nx * ny * nz, the appropriate scaling here is::

        sigma**2 = (nx * ny * nz) * P(k) / (2 * Vbox)
                 = P(k) / (2 * spacing**3)

    """
    validate_power(power)

    nx, ny, nz = transform.expanded_shape(data, packed=packed)
    N3 = nx * ny * nz
    Vbox = N3 * spacing**3

    power_k_min, power_k_max = np.min(power['k']), np.max(power['k'])
    if power_k_min <= 0:
        raise ValueError('Power uses min(k) <= 0: {0}.'.format(power_k_min))
    data_k_min, data_k_max = get_k_bounds(data, spacing=spacing, packed=packed)
    if power_k_min > data_k_min or power_k_max < data_k_max:
        raise ValueError(
        'Power k range [{0}:{1}] does not cover data k range [{2}:{3}].'
        .format(power_k_min, power_k_max, data_k_min, data_k_max))

    # Build an interpolater of sigma(|k|) that is linear in log10(|k|).
    log10_k = np.log10(power['k'])
    sigma = N3 * np.sqrt(power['Pk'] / (2 * Vbox))
    interpolator = scipy.interpolate.interp1d(
        log10_k, sigma, kind='linear', copy=False,
        bounds_error=False, fill_value=0)

    # Calculate interpolated values of sigma.
    # We would ideally do this in place, but scipy.interpolate methods
    # do not support this.  A slower but memory efficient alternative would
    # be to interpolate in batches.  For now we just do the simplest thing.
    data.real = interpolator(data.real)
    return data


def load_default_power():
    """
    Loads a default power spectrum P(k,z) with 1e-4 <= k <= 22 and z = 0.

    The default power spectrum was created using::

        from randomfield.power import calculate_power
        result = calculate_power(1e-4, 22.)
        np.savetxt('default_power.dat', result)

    The range of k values used here is sufficient to cover grids with
    spacing >= 0.25 Mpc/h and and box dimensions <= 50 Gpc/h.
    """
    try:
        import powertools
        package_path = os.path.dirname(inspect.getfile(powertools))
        data_path = os.path.join(package_path, 'data', 'default_power.dat')
        return np.loadtxt(data_path, dtype=[('k', float), ('Pk', float)])
    except ImportError:
        raise RuntimeError('Unable to locate default_power.dat.')
    except IOError:
        raise RuntimeError('Unable to load default_power.dat')
