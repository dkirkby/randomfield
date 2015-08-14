# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Calculate cosmological quantities.
"""
from __future__ import print_function, division

import numpy as np
import scipy.interpolate
import astropy.cosmology
import astropy.units as u


def get_cosmology(cosmology='Planck13'):
    """
    Return an astropy.cosmology object.

    Input can either be a pre-defined name, an instance of
    :class:`astropy.cosmology.FRW`, or else a dictionary of parameters that can
    be used to initialize a new :class:`astropy.cosmology.FlatLambdaCDM`.
    """
    # Convert a named cosmology to the corresponding astropy.cosmology model.
    if isinstance(cosmology, basestring):
        if cosmology not in astropy.cosmology.parameters.available:
            raise ValueError(
                'Not a recognized cosmology: {0}.'.format(cosmology))
        cosmology = astropy.cosmology.__dict__[cosmology]
    elif not isinstance(cosmology, astropy.cosmology.FLRW):
        try:
            cosmology = astropy.cosmology.FlatLambdaCDM(**cosmology)
        except TypeError:
            raise ValueError(
                'Cannot initialize a cosmology using: {0}.'.format(cosmology))
    return cosmology


def get_class_parameters(cosmology='Planck13'):
    """
    Get CLASS parameters corresponding to an astropy cosmology model.
    """
    cosmology = get_cosmology(cosmology)

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


def calculate_power(k_min, k_max, z=0, num_k=500, scaled_by_h=True,
                    cosmology='Planck13', n_s=0.9619, logA=3.0980):
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


def get_redshifts(data, spacing, scaled_by_h=True, cosmology='Planck13',
                  z_axis=2, num_interpolation_points=100):
    """
    Calculate the redshift grid.

    The output array is 3D with the same size as the data along z_axis but
    length 1 along the other two axes, so it is broadcastable with data.
    """
    cosmology = get_cosmology(cosmology)

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


def convert_delta_to_density(data, spacing, cosmology='Planck13', z_axis=-1):
    """
    Convert a delta field into a density field with light-cone evolution.
    """
    pass
