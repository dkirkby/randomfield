# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Calculate cosmological quantities.
"""
from __future__ import print_function, division

import numpy as np
import scipy.interpolate
import scipy.integrate
import astropy.cosmology
import astropy.units as u


def create_cosmology(*args, **kwargs):
    """
    Create a background cosmology model.

    The created model will implement the :class:`astropy.cosmology.FLRW` API.
    Input can either be a pre-defined name, or else a dictionary of parameters
    that can be used to initialize a new
    :class:`astropy.cosmology.FlatLambdaCDM`. The valid names are
    'WMAP5', 'WMAP7', 'WMAP9', 'Planck13'.

    If no arguments are specified, creates the 'Planck13' model.
    """
    if len(args) > 0 and len(kwargs) > 0:
        raise TypeError('Cannot specify both a name and parameters.')
    elif len(args) > 1:
        raise TypeError('Invalid arguments: expected a name or parameters.')
    elif len(args) == 1 and not isinstance(args[0], basestring):
        raise TypeError('Invalid name: {0}.'.format(args[0]))

    if len(args) == 0 and len(kwargs) == 0:
        args = ['Planck13']

    # Convert a named cosmology to the corresponding astropy.cosmology model.
    if len(args) == 1:
        if args[0] not in astropy.cosmology.parameters.available:
            raise ValueError(
                'Not a recognized cosmology: {0}.'.format(args[0]))
        cosmology = astropy.cosmology.__dict__[args[0]]
    else:
        cosmology = astropy.cosmology.FlatLambdaCDM(**kwargs)

    return cosmology


def get_class_parameters(cosmology):
    """
    Get CLASS parameters corresponding to an astropy cosmology model.
    """
    class_parameters = {}
    try:
        class_parameters['h'] = cosmology.h
        class_parameters['T_cmb'] = cosmology.Tcmb0.value
        class_parameters['Omega_b'] = cosmology.Ob0
        # CDM = DM - massive neutrinos.
        class_parameters['Omega_cdm'] = cosmology.Odm0 - cosmology.Onu0
        class_parameters['Omega_k'] = cosmology.Ok0
        # Dark energy sector. CLASS will calculate whichever of
        # Omega_Lambda/fld/scf we do not provide.
        class_parameters['Omega_scf'] = 0.
        try:
            # Only subclasses of wCDM have the w0 attribute.
            class_parameters['w0_fld'] = cosmology.w0
            class_parameters['wa_fld'] = 0.
            class_parameters['Omega_Lambda'] = 0.
        except AttributeError:
            class_parameters['Omega_fld'] = 0.
        # Neutrino sector.
        if cosmology.has_massive_nu:
            m_nu = cosmology.m_nu
            try:
                num_massive = len(m_nu)
            except TypeError:
                num_massive = 3
                m_nu = [num_massive * m_nu]
            if num_massive > 3:
                raise RuntimeError(
                    'Cannot calculate cosmology with >3 massive neutrinos.')
            m_ncdm = []
            for mass in m_nu:
                if mass > 0:
                    m_ncdm.append(repr(mass.value))
                else:
                    num_massive -= 1
            if num_massive > 0:
                class_parameters['m_ncdm'] = ','.join(m_ncdm)
        else:
            num_massive = 0
        class_parameters['N_ncdm'] = num_massive
        num_light = 3 - num_massive
        class_parameters['N_ur'] = (3.00641 - num_massive +
            num_light*(cosmology.Neff - 3.00641)/3.)
    except AttributeError:
        raise ValueError('Cosmology is missing required attributes.')
    return class_parameters


def calculate_power(cosmology, k_min, k_max, z=0, num_k=500, scaled_by_h=True,
                    n_s=0.9619, logA=3.0980):
    """
    Calculate the power spectrum P(k,z) over the range k_min <= k <= k_max.
    """
    try:
        from classy import Class
        cosmo = Class()
    except ImportError:
        raise RuntimeError('power.calculate_power requires classy.')

    class_parameters = get_class_parameters(cosmology)
    class_parameters['output'] = 'mPk'
    if scaled_by_h:
        class_parameters['P_k_max_h/Mpc'] = k_max
    else:
        class_parameters['P_k_max_1/Mpc'] = k_max
    class_parameters['n_s'] = n_s
    class_parameters['ln10^{10}A_s'] = logA
    cosmo.set(class_parameters)
    cosmo.compute()

    if scaled_by_h:
        k_scale = cosmo.h()
        Pk_scale = cosmo.h()**3
    else:
        k_scale = 1.
        Pk_scale = 1.

    result = np.empty((num_k,), dtype=[('k', float), ('Pk', float)])
    result['k'][:] = np.logspace(np.log10(k_min), np.log10(k_max), num_k)
    for i, k in enumerate(result['k']):
        result['Pk'][i] = cosmo.pk(k * k_scale, z) * Pk_scale

    cosmo.struct_cleanup()
    cosmo.empty()

    return result


def get_redshifts(cosmology, data, spacing, scaled_by_h=True,
                  z_axis=2, num_interpolation_points=100):
    """
    Calculate the redshift grid.

    The output array is 3D with the same size as the data along z_axis but
    length 1 along the other two axes, so it is broadcastable with data.
    We use the plane-parallel approximation.
    """
    if len(data.shape) != 3:
        raise ValueError('Input data is not 3D.')
    try:
        nz = data.shape[z_axis]
    except (IndexError, TypeError):
        raise ValueError('Invalid z_axis: {0}.'.format(z_axis))
    output_shape = np.ones((3,), dtype=int)
    output_shape[z_axis] = nz

    comoving_distances = np.arange(nz) * spacing
    if scaled_by_h:
        comoving_distances /= cosmology.h
    max_redshift = astropy.cosmology.z_at_value(
        cosmology.comoving_distance, comoving_distances[-1] * u.Mpc)

    # Build an interpolator for z(d) where d is in Mpc (not Mpc/h).
    redshift_grid = np.linspace(
        0., 1.05 * max_redshift, num_interpolation_points)
    comoving_distance_grid = cosmology.comoving_distance(redshift_grid)
    z_interpolator = scipy.interpolate.interp1d(
        comoving_distance_grid, redshift_grid, kind='cubic', copy=False)
    redshifts = z_interpolator(comoving_distances)

    # Make sure the first element is exactly zero.
    redshifts[0] = 0
    # Make the returned array broadcastable with the input data.
    return redshifts.reshape(output_shape)


def get_mean_matter_densities(cosmology, redshifts):
    """
    Calculate mean densities in g / cm**3.
    """
    mean_density0 = (
        cosmology.critical_density0 * cosmology.Om0).to(u.gram / u.cm**3).value
    return mean_density0 * (1 + redshifts)**3


def get_growth_function(cosmology, redshifts):
    """
    Calculate the growth function.

    For now, we use the Linder 2005 approximation. See equations (14-16) of
    Weinberg 2012 for details.
    """
    z_axis = np.arange(3)[np.argsort(redshifts.shape)][-1]
    try:
        w0 = cosmology.w0
    except AttributeError:
        w0 = -1
    gamma = 0.55 + 0.05 * (1 + w0)
    integrand = np.power(cosmology.Om(redshifts), gamma) / (1 + redshifts)
    exponent = scipy.integrate.cumtrapz(
        y=integrand, x=redshifts, axis=z_axis, initial=0)
    return np.exp(-exponent)


def apply_lognormal_transform(delta, growth, sigma=None):
    """
    Transform delta values drawn from a normal distribution with mean zero and
    standard deviation sigma to have a log-normal distribution with mean one
    and standard deviation growth * sigma. Transforms are applied in place,
    overwriting the input delta field.  If sigma is not specified, np.std(delta)
    will be used.
    """
    if sigma is None:
        sigma = np.std(delta)
    t = 1 + (sigma * growth)**2
    delta /= sigma
    delta *= np.sqrt(np.log(t))
    delta = np.exp(delta, out=delta)
    delta /= np.sqrt(t)
    return delta
