# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Utilities for weak lensing calculations.
"""
from __future__ import print_function, division

import numpy as np

import scipy.interpolate
import scipy.integrate

import astropy.constants
import astropy.units


def calculate_lensing_weights(cosmology, z, DC=None, DA=None, scaled_by_h=True):
    """
    Calculate the geometric weight function for lensing.

    There is no standard definition of the lensing weight function, so we
    adopt the following function:

    .. math::

        W(D, D_{src}) = \left[ \\frac{3}{2} (H_0/c)^2 \Omega_m \\right]^2
        (1 + z(D))^2 \\frac{D_A(D)^3 D_A(D_{src}-D)^2}{D_A(D_{src})^2}

    where :math:`D` and :math:`D_{src}` are the comoving distance to the lensing
    mass and source galaxy, respectively, and :math:`z` is the lensing mass
    redshift.  This function determines the geometric weight (i.e., independent
    of the 2D wavenumber :math:`\ell`) of a mass at :math:`D` lensing a source
    at :math:`D_{src}`. The weight dimensions are inverse length in units of
    either Mpc/h or Mpc (depending on the ``scaled_by_h`` parameter).  The
    limiting values are:

    .. math::

        W(0, D_{src}) = W(D_{src}, D_{src}) = 0

    and the function is broadly peaked with a maximum near
    :math:`z_{lens} = z_{src}/2` for realistic cosmologies.

    Parameters
    ----------
    cosmology : instance of astropy.cosmology.FLRW
        Background cosmology that specifies the values of Omega_k, Omega_m and
        (if ``scaled_by_h`` is False) H0 to use.  If ``DA`` is not provided, the
        cosmology is also used to calculate ``DA`` values from the input ``DC``
        values.
    z: numpy array
        1D array of redshifts to use for the grid of source and lens positions
        in the calculated weights. Values must be increasing but do not need to
        be equally spaced.
    DC : numpy array, optional
        Array of :meth:`comoving distances
        <astropy.cosmology.FLRW.comoving_distance>` along the line of sight
        corresponding to each redshift.  Units should be either Mpc (when
        ``scaled_by_h`` is False) or Mpc/h (``scaled_by_h`` is True). Will be
        calculated from the cosmology if not specified.
    DA : numpy array, optional
        Array of :meth:`comoving transverse distances
        <astropy.cosmology.FLRW.comoving_transverse_distance>` corresponding to
        each redshift.  Units should be either Mpc (when ``scaled_by_h`` is
        False) or Mpc/h (``scaled_by_h`` is True). Will be calculated from the
        cosmology if not specified.
    scaled_by_h: bool, optional
        Specifies the units of the input ``DC`` and ``DA`` arrays and the output
        weights.  When True, inputs must be in Mpc/h and the result is in h/Mpc.
        When False, inputs must be in Mpc and the result is in 1/Mpc.

    Returns
    -------
    out : numpy array
        Two dimensional array with shape (nz, nz) where nz is the number of
        input redshifts.  The value ``out[i,j]`` gives the weight function for a
        source at redshift ``z[i]`` as a function of lensing redshift ``z[j]``
        with dimensions of inverse length (h/Mpc or 1/Mpc, depending on the
        value of ``scaled_by_h``).  Note that all values with j > i are zero,
        since they correspond to sources in front of the lensing mass.
    """
    try:
        assert len(z.shape) == 1
    except (AssertionError, AttributeError):
        raise ValueError('Redshifts must be a 1D array.')
    if not np.array_equal(np.unique(z), z) or z[0] < 0:
        raise ValueError('Redshifts must be increasing values >= 0.')
    n = z.size

    # Calculate distances if necessary.
    try:
        scale = cosmology.h if scaled_by_h else 1.
        if DC is None:
            DC = (cosmology.comoving_distance(z)
                .to(astropy.units.Mpc).value * scale)
        if DA is None:
            DA = (cosmology.comoving_transverse_distance(z)
                .to(astropy.units.Mpc).value * scale)
    except AttributeError:
        raise ValueError('Invalid cosmology: cannot calculate distances.')

    if DC.shape != z.shape:
        raise ValueError('Shapes not compatible: z and DC.')
    if DA.shape != z.shape:
        raise ValueError('Shapes not compatible: z and DA.')

    # Calculate the dimensioned curvature constant K in units of (Mpc/h)**-2.
    clight_km_s = astropy.constants.c.to(
        astropy.units.km / astropy.units.s).value
    H0 = 100. if scaled_by_h else 100. * cosmology.h
    K = -cosmology.Ok0 * (H0 / clight_km_s)**2
    # Tabulate cosK(D), which is dimensionless.
    if K < 0:
        sqrt_abs_K = np.sqrt(-K)
        cosK =  np.cosh(sqrt_abs_K * DC)
    elif K > 0:
        sqrt_K = np.sqrt(K)
        cosK = np.cos(sqrt_K * DC)
    else:
        cosK = np.ones_like(DA, dtype=float)

    # Tabulate DA(D) / DA(Dsrc) on an n x n grid, using the value 1 when D=Dsrc.
    DAratio = np.empty((n, n), dtype=float)
    DAratio[0] = 1.
    DAratio[1:] = DA / DA[1:, np.newaxis]

    # Tabulate q(D,Dsrc) * DA(D) = cosK(D) - cosK(Dsrc) * DA(D) / DA(Dsrc)
    # on an n x n grid, in units of (Mpc/h)**-1.
    weights = cosK - cosK[:,np.newaxis] * DAratio
    # Zero entries < 0 where D > Dsrc.
    weights[weights < 0] = 0
    # Multiply by 1 + z(D)
    weights *= 1 + z
    # Multiply by the constant 3/2 (H0/c)**2 Omega_m in units of (h/Mpc)**4.
    weights *= 1.5 * (H0 / clight_km_s)**2 * cosmology.Om0
    # Square the weights and multiply by DA**3.
    weights = np.square(weights, out=weights)
    weights *= DA**3

    return weights


def tabulate_3D_variances(ell, DA, growth, power):
    """
    Tabulate 3D matter power contributions to shear power variances.

    This function is defined as:

    .. math::

        V(\ell, D_A) = \\frac{\pi}{\ell} G(D)^2
        \Delta^2_\delta(k=\ell/D_A(D), z=0)

    where :math:`D` is the comoving distance corresponding to the comoving
    transverse distance :math:`D_A` and :math:`\ell` is a 2D wavenumber.

    Parameters
    ----------
    ell: numpy array
        1D array of 2D wavenumbers where shear power variances should be
        tabulated.  Values must be positive and increasing, but do not need
        to be equally spaced.
    DA: numpy array
        1D array of :meth:`comoving transverse distances
        <astropy.cosmology.FLRW.comoving_transverse_distance>` :math:`D_A(z)`
        where shear power variances should be tabulated. Values must be
        positive and increasing, but do not need to be equally spaced. Values
        can be in either Mpc/h or Mpc, but must be consistent with the power
        spectrum normalization.
    growth: numpy array
        1D array of growth function values :math:`G(z)` corresponding to each
        input :math:`DA(z)` value.  Must have the same number of elements as
        DA.  Can be calculated using
        :func:`randomfield.cosmotools.get_growth_function`.
    power: structured numpy array
        Power spectrum to use, which meets the criteria tested by
        :func:`randomfield.powertools.validate_power`.  Can be calculated using
        :func:`randomfield.cosmotools.calculate_power` if the optional
        ``classy`` package is installed.  Values of k and P(k) can either be
        in Mpc/h or Mpc units, but must be consistent with the values of DA.

    Returns
    -------
    out : numpy array
        Two dimensional array with shape (nell, nDA) where nell = len(ell) and
        nDA = len(DA).  The value ``out[i,j]`` gives the contribution to the
        shear variance :math:`\Delta^2_{EE}` at 2D wavenumber ``ell[i]`` from
        lensing by mass inhomogeneities at comoving transverse distance
        ``DA[j]``.  The output is dimensionless.
    """
    try:
        assert len(ell.shape) == 1
    except (AssertionError, AttributeError):
        raise ValueError('Wavenumbers ell must be a 1D array.')
    if not np.array_equal(np.unique(ell), ell) or ell[0] <= 0:
        raise ValueError('Wavenumbers ell must be increasing values > 0.')

    try:
        assert len(DA.shape) == 1
    except (AssertionError, AttributeError):
        raise ValueError('Distances DA must be a 1D array.')
    if not np.array_equal(np.unique(DA), DA) or DA[0] <= 0:
        raise ValueError('Distances DA must be increasing values > 0.')

    try:
        assert DA.shape == growth.shape
    except (AssertionError, AttributeError):
        raise ValueError('Growth array must match DA array dimensions.')

    # Build an interpolator for the dimensionless function
    # Delta**2(k) = k**3/(2*pi**2) * P(k) that is linear in log10(k).
    k_grid = power['k']
    Delta2_grid = k_grid**3 / (2 * np.pi**2) * power['Pk']
    Delta2 = scipy.interpolate.interp1d(np.log10(k_grid), Delta2_grid,
                                        kind='linear', copy=False)
    # Tabulate a 2D array of k values for each (ell, DA).
    log10k_of_DA = np.log10(ell[:, np.newaxis] / DA)
    # Tabulate pi/ell * Delta**2(k) * G(DA)**2 values on this grid.
    return (np.pi / ell[:, np.newaxis]) * Delta2(log10k_of_DA) * growth**2
