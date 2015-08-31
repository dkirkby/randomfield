# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Utilities for weak lensing calculations.
"""
from __future__ import print_function, division

import numpy as np

import scipy.interpolate
import scipy.integrate
import scipy.special

import astropy.constants
import astropy.units

import powertools


def calculate_lensing_weights(cosmology, z, DC=None, DA=None, scaled_by_h=True):
    """
    Calculate the geometric weights for lensing.

    We adopt the following convention for the lensing weight function:

    .. math::

        \omega_E(D, D_{src}) = \\frac{3}{2} (H_0/c)^2 \Omega_m
        (1 + z(D)) \\frac{D_A(D_{src}-D)}{D_A(D_{src})} \Theta(D_{src}-D)

    where :math:`D` and :math:`D_{src}` are the comoving distance to the lensing
    mass and source galaxy, respectively, and :math:`z` is the lensing mass
    redshift. The weight dimensions are inverse length squared, in units of
    either Mpc/h or Mpc (depending on the ``scaled_by_h`` parameter).

    The combination:

    .. math::

        W_{EE}(D, D_{src}) = \omega_E(D, D_{src})^2 D_A(D)^3

    determines the geometric weight (i.e., independent of the 2D wavenumber
    :math:`\ell`) of a mass at :math:`D` lensing a source at :math:`D_{src}`.
    The limiting values are:

    .. math::

        W_{EE}(0, D_{src}) = W_{EE}(D_{src}, D_{src}) = 0

    and the function is broadly peaked with a maximum near
    :math:`z_{lens} = z_{src}/2` for realistic cosmologies.

    Use :func:`calculate_shear_power` to calculate the lensing shear power
    associated with the returned weight functions.

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
        1D array of :meth:`comoving distances
        <astropy.cosmology.FLRW.comoving_distance>` along the line of sight
        corresponding to each redshift.  Units should be either Mpc (when
        ``scaled_by_h`` is False) or Mpc/h (``scaled_by_h`` is True). Will be
        calculated from the cosmology if not specified.
    DA : numpy array, optional
        1D array of :meth:`comoving transverse distances
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
    # Multiply by the constant 3/2 (H0/c)**2 Omega_m with units of
    # (h/Mpc)**2 or (1/Mpc)**2, depending on scaled_by_h.
    weights *= 1.5 * (H0 / clight_km_s)**2 * cosmology.Om0

    return weights


def tabulate_3D_variances(ell, DA, growth, power):
    """
    Tabulate 3D matter-power contributions to shear power variances.

    This function is defined as:

    .. math::

        V(\ell, D_A) = \\frac{\pi}{\ell} G(D)^2
        \Delta^2_\delta(k=\ell/D_A(D), z=0)

    where :math:`D` is the comoving distance corresponding to the comoving
    transverse distance :math:`D_A` and :math:`\ell` is a 2D wavenumber.

    Use :func:`calculate_shear_power` to calculate the lensing shear power
    associated with the returned 3D matter-power contributions.

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
    power = powertools.validate_power(power)
    k_grid = power['k']
    Delta2_grid = k_grid**3 / (2 * np.pi**2) * power['Pk']
    Delta2 = scipy.interpolate.interp1d(np.log10(k_grid), Delta2_grid,
                                        kind='linear', copy=False)
    # Tabulate a 2D array of k values for each (ell, DA).
    log10k_of_DA = np.log10(ell[:, np.newaxis] / DA)
    # Tabulate pi/ell * Delta**2(k) * G(DA)**2 values on this grid.
    return (np.pi / ell[:, np.newaxis]) * Delta2(log10k_of_DA) * growth**2


def calculate_shear_power(DC, DA, weights, variances, mode='shear-shear-auto'):
    """
    Calculate the shear power spectrum as a function of source position.

    The result is given as (X = E,g):

    .. math::

        \Delta^2_{EX}(\ell) = \\frac{\ell^2}{2\pi} C_{EX}(\ell)

    and calculated as a convolution of the input weight functions
    :math:`\omega_E(D, D_{src})` and 3D variances :math:`V(\ell, D_A)`. There
    are three possible calculations, depending on the ``mode`` parameter.
    Shear-shear auto power is calculated as:

    .. math::

        \Delta^2_{EE}(D_{src}, \ell) = \int_{D_{min}}^{D_{src}}
        \omega_E(D, D_{src})^2 D_A(D)^3 V(\ell, D_A(D)) dD

    Shear-shear cross power is calculated as:

    .. math::

        \Delta^2_{EE}(D_1, D_2, \ell) = \int_{D_{min}}^{\min(D_1,D_2)}
        \omega_E(D, D_1) \omega_E(D, D_2) D_A(D)^3 V(\ell, D_A(D)) dD

    Shear-galaxy cross power is calculated as:

    .. math::

        \Delta^2_{Eg}(D_{src}, D_g, \ell) =
        \omega_E(D_g, D_{src}) D_A(D_g)^3 V(\ell, D_A(D_g))

    Note that no bias factor is included in the shear-galaxy calculation, so
    results should be interpreted as :math:`\Delta^2_{Eg} / b_g`.

    The convolution integrals above are estimated using :func:`Simpson's rule
    <scipy.integrate.simps>` and finer grids will generally yield more accurate
    results.

    Note that the integrals above are truncated at :math:`D_{min} > 0` equal to
    ``DC[0]``, in order to set a finite upper bound on the
    3D wavenumber :math:`k = \ell/D_A`. This truncation procedure effectively
    ignores any lensing by mass inhomogeneities closer than ``DC[0]``.
    Reducing :math:`D_{min}` will increase the accuracy of the calculation
    but also requires either increasing the range of 3D wavenumbers :math:`k`
    or decreasing the range of 2D wavenumbers :math:`\ell` used to calculate
    :math:`V(\ell, D_A(D))`.

    It is the caller's responsibility to ensure that the inputs are all
    calculated on consistent grids and with consistent units (Mpc or Mpc/h).

    Parameters
    ----------
    DC : numpy array
        1D array of :meth:`comoving distances
        <astropy.cosmology.FLRW.comoving_distance>` along the line of sight
        corresponding to each redshift.  Units can be either Mpc or Mpc/h,
        but must be consistent with how the weights and variances were
        calculated.
    DA: numpy array
        1D array of :meth:`comoving transverse distances
        <astropy.cosmology.FLRW.comoving_transverse_distance>` corresponding to
        each redshift.  Units can be in either Mpc/h or Mpc, but must be
        consistent with DC and with how the weights and variances were
        calculated.
    weights: numpy array
        2D array of geometric lensing weights :math:`\omega_E(D, D_{src})`,
        normally obtained by calling :func:`calculate_lensing_weights`.  Units
        (Mpc or Mpc/h) must be consistent with those used for ``DC`` and
        ``variances``. The shape must be (nDC, nDC) where nDC = len(DC).
    variances: numpy array
        2D array of 3D matter power contributions :math:`V(\ell, D_A)` to the
        shear variance, normally obtained by calling
        :func:`tabulate_3D_variances`. Must be tabulated using units that are
        consistent with those used for ``DC`` and ``weights``.  The shape must
        be (nell, nDC) where nDC = len(DC).
    mode: str, optional
        Must be one of 'shear-shear-auto', 'shear-shear-cross', or
        'shear-galaxy-cross'. Specifies the type of power spectrum that is
        calculated and determines the shape of the result.

    Returns
    -------
    out : numpy array
        For 'shear-shear-auto', the result is a 2D array of lensing power
        spectra with shape (nDC, nell) where nDC = len(DC) and nell =
        weights.shape[0].  The value ``out[i,n]`` gives :math:`\Delta^2_{EE}`
        as a function of ``ell[n]`` for a lensed source at distance ``DC[i]``
        (where ``ell`` is the array of 2D wavenumbers used to :func:`tabulate
        the input variances <tabulate_3D_variances>`).

        For 'shear-shear-cross', the result is a 3D array of lensing power
        cross spectra with shape (nDC, nDC, nell).  The value ``out[i,j,n]``
        gives :math:`\Delta^2_{EE}` as a function of ``ell[n]`` for lensed
        sources at distances ``DC[i]`` and ``DC[j]``.  The result is symmetric
        in ``i`` and ``j``, and the diagonal ``i = j`` equals the
        'shear-shear-auto' result.

        For 'shear-galaxy-cross', the result is a 3D array of galaxy-lensing
        cross spectra with shape (nDC, nDC, nell). The value ``out[i,j,n]``
        gives :math:`\Delta^2_{Eg}` as a function of ``ell[n]`` for a lensed
        source at distance ``DC[i]`` and a galaxy at distance ``DC[j]``.
    """
    modes = ('shear-shear-auto', 'shear-shear-cross', 'shear-galaxy-cross')
    if mode not in modes:
        raise ValueError('Invalid mode. Pick one of: {0}.'
                         .format(','.join(modes)))

    try:
        assert len(DC.shape) == 1
    except (AssertionError, AttributeError):
        raise ValueError('Distances DC must be a 1D array.')
    if not np.array_equal(np.unique(DC), DC) or DC[0] <= 0:
        raise ValueError('Distances DC must be increasing values > 0.')
    nDC = DC.size

    try:
        assert len(DA.shape) == 1
    except (AssertionError, AttributeError):
        raise ValueError('Distances DA must be a 1D array.')
    if DA.size != DC.size:
        raise ValueError('Distance arrays DC and DA must have same size.')

    try:
        assert len(weights.shape) == 2
    except (AssertionError, AttributeError):
        raise ValueError('Weights must be a 2D array.')
    if weights.shape != (nDC, nDC):
        raise ValueError('Weights have wrong shape. Expected ({0}, {1}).'
                         .format(nDC, nDC))

    try:
        assert len(variances.shape) == 2
    except (AssertionError, AttributeError):
        raise ValueError('Variances must be a 2D array.')
    if variances.shape[1] != nDC:
        raise ValueError('Variances has wrong second dimension. Expected {0}.'
                         .format(nDC))
    nell = variances.shape[0]

    if mode == 'shear-shear-auto':
        # Initialize the result array.
        Delta2 = np.empty((nDC, nell), dtype=float)
        # Loop over lensed source positions.
        for i in range(nDC):
            integrand = weights[i]**2 * DA**3 * variances
            Delta2[i] = scipy.integrate.simps(y=integrand, x=DC, axis=-1)
    elif mode == 'shear-shear-cross':
        # Initialize the result array.
        Delta2 = np.empty((nDC, nDC, nell), dtype=float)
        # Loop over lensed source positions.
        for i in range(nDC):
            for j in range(i+1):
                integrand = weights[i] * weights[j] * DA**3 * variances
                Delta2[i, j] = scipy.integrate.simps(y=integrand, x=DC, axis=-1)
                if i > j:
                    Delta2[j, i] = Delta2[i, j]
    elif mode == 'shear-galaxy-cross':
        # Initialize the result array.
        Delta2 = np.empty((nDC, nDC, nell), dtype=float)
        # Loop over lensed source positions.
        for i in range(nDC):
            Delta2[i] = np.transpose(weights[i] * DA**3 * variances)

    return Delta2


def calculate_correlation_function(Delta2, ell, theta, order=0):
    """
    Transform a function of ell into a function of angular separation.

    The transform is defined as:

    .. math::

        \\xi(\Delta\\theta) = (-1)^{\\nu/2} \int_{\ell_{min}}^{\ell_{max}}
        \Delta^2(\ell) J_\\nu(\ell \Delta\\theta) \\frac{d\ell}{\ell}

    where the order :math:`\\nu` should be 0 and 4 for the shear-shear
    correlations :math:`\\xi_+` and :math:`\\xi_-`, respectively, and 2 for
    the shear-galaxy correlation :math:`\\xi_{Eg}`.

    The result is only approximate since the integral above is truncated at
    the limits :math:`\\ell_{min}` and :math:`\\ell_{max}` corresponding to
    ``ell[0]`` and ``ell[-1]``, respectively. The resulting finite integral
    is estimated using :func:`Simpson's rule <scipy.integrate.simps>` (in
    :math:`\log\ell`) and a finer grid in ``ell`` will generally yield more
    accurate results.

    Parameters
    ----------
    Delta2 : numpy array
        3D array of cross power spectra :math:`\Delta^2(z_1,z_2,\ell)`.
    ell : numpy array
        1D array of 2D wavenumbers :math:`\ell` where the input cross power
        spectra are tabulated.
    dtheta: numpy array
        1D array of 2D angular separations :math:`\Delta\\theta` where the
        output cross correlations should be tabulated.
    order: int
        Order :math:`\\nu` of the Bessel function :math:`J_{\\nu}` to use
        for the transform.  Must be 0, 2, or 4.

    Returns
    -------
    out : numpy array
        Array of cross correlations :math:`\\xi(z_1,z_2,\Delta\\theta)`.
        The result is only calculated for :math:`z_1 \ge z_2`.  If order
        equals 0 or 4, the result is symmetrized.  Otherwise, values for
        :math:`z_1 < z_2` are returned as zero.
    """
    if order not in (0, 2, 4):
        raise ValueError('Invalid order. Choose from 0, 2, 4.')

    try:
        assert len(ell.shape) == 1
    except (AssertionError, AttributeError):
        raise ValueError('Wavenumbers ell must be a 1D array.')
    if not np.array_equal(np.unique(ell), ell) or ell[0] <= 0:
        raise ValueError('Wavenumbers ell must be increasing values > 0.')
    num_ell = ell.size

    try:
        assert len(Delta2.shape) == 3
    except (AssertionError, AttributeError):
        raise ValueError('Delta2 must be a 3D array.')
    num_z = len(Delta2)
    if Delta2.shape != (num_z, num_z, num_ell):
        raise ValueError('Delta2 has wrong shape {0}.  Expected {1}.'
                         .format(Delta2.shape, (num_z, num_z, num_ell)))

    try:
        assert len(theta.shape) == 1
    except (AssertionError, AttributeError):
        raise ValueError('Separations theta must be a 1D array.')
    if not np.array_equal(np.unique(theta), theta) or theta[0] <= 0:
        raise ValueError('Separations theta must be increasing values > 0.')
    num_theta = theta.size

    theta_min = 2 * np.pi / ell[-1]
    theta_max = 2 * np.pi / ell[0]
    if theta[0] < theta_min or theta[-1] > theta_max:
        raise ValueError(
            'Maximum allowed theta coverage is [{0:.3f}, {1:.3f}] rad'
            .format(theta_min, theta_max))

    # Calculate the kernel as a 2D array where
    # kernel[i,j] = Jn(ell[j] * theta[i])
    kernel = scipy.special.jv(order, ell * theta[:, np.newaxis])

    # Tabulate values of log10(ell), which are the integration abscissas.
    log_ell = np.log10(ell)

    xi_shape = (num_z, num_z, num_theta)
    if order == 2:
        xi = np.zeros(xi_shape, dtype=float)
    else:
        xi = np.empty(xi_shape, dtype=float)

    for i in range(num_z):
        for j in range(i+1):
            integrand = Delta2[i, j] * kernel
            xi[i, j] = scipy.integrate.simps(y=integrand, x=log_ell, axis=-1)
            if order != 2 and i > j:
                xi[j, i] = xi[i, j]

    if order == 2:
        # Use a unfunc to ensure this happens in place.
        xi = np.negative(xi, out=xi)

    return xi
