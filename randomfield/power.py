# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import print_function, division

import inspect
import os.path

import numpy as np
import scipy.interpolate

from transform import expanded_shape


def get_k_bounds(data, spacing, packed=True):
    """
    Return the bounds of |k|^2 values for the specified grid.
    """
    nx, ny, nz = expanded_shape(data, packed=packed)
    k0 = (2 * np.pi) / spacing
    k_min = k0 / max(nx, ny, nz)
    k_max = k0 * np.sqrt(3) / 2
    return k_min, k_max


def fill_with_log10k(data, spacing, packed=True):
    """
    Fill an array with values of log10(|k|).

    Note that the value at [0, 0, 0] will be log10(0) = -inf.
    """
    nx, ny, nz = expanded_shape(data, packed=packed)
    lambda0 = spacing / (2 * np.pi)
    kx = np.fft.fftfreq(nx, lambda0)
    ky = np.fft.fftfreq(ny, lambda0)
    kz = np.fft.fftfreq(nz, lambda0)
    if packed:
        kz = kz[:nz//2 + 1]
    # With the sparse option, the memory usage of these grids is
    # O(nx+ny+nz) rather than O(nx*ny*nz).
    kx2_grid, ky2_grid, kz2_grid = np.meshgrid(
        kx**2, ky**2, kz**2, sparse=True, indexing='ij')
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


def tabulate_sigmas(data, power, spacing, packed=True):
    """
    Replace an array of log10(|k|) values with the corresponding sigmas.

    Note that the scaling from P(k) to variance depends on convention for
    normalizing the inverse FFT.  Since we divide the inverse FFT by
    nx * ny * nz, the appropriate scaling here is::

        sigma**2 = (nx * ny * nz) * P(k) / (2 * Vbox)
                 = P(k) / (2 * spacing**3)

    """
    if not isinstance(power, np.ndarray):
        raise ValueError('Power must be a structured numpy array.')
    if not ('k' in power.dtype.names and 'Pk' in power.dtype.names):
        raise ValueError('Power must have fields named k, Pk.')

    nx, ny, nz = expanded_shape(data, packed=packed)
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
        import memory
        package_path = os.path.dirname(inspect.getfile(memory))
        data_path = os.path.join(package_path, 'data', 'default_power.dat')
        return np.loadtxt(data_path, dtype=[('k', float), ('Pk', float)])
    except ImportError:
        raise RuntimeError('Unable to locate default_power.dat.')
    except IOError:
        raise RuntimeError('Unable to load default_power.dat')
