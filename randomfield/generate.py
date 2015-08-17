# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
High-level functions to generate random fields.
"""
from __future__ import print_function, division

import random

import numpy as np
import scipy.interpolate

import transform
import powertools
import cosmotools


class Generator(object):
    """
    Manage random field generation for a specified geometry.

    All memory allocation is performed by the constructor.

    Parameters
    ----------
    nx : int
        Size of the generator grid along the x axis, corresponding to the
        right-ascension direction relative to the line of sight.
    ny : int
        Size of the generator grid along the y axis, corresponding to the
        declination direction relative to the line of sight.
    nz : int
        Size of the generator grid along the y axis, corresponding to the
        line-of-sight drection.
    grid_spacing_Mpc_h: float
        Uniform grid spacing in Mpc/h.
    cosmology: :class:`astropy.cosmology.FLRW`, optional
        Homogeneous background cosmology to use for distances and
        for calculating the power spectrum of inhomogeneities. Should be
        an instance of :class:`astropy.cosmology.FLRW`. Simple cases can
        be conveniently created using
        :func:`randomfield.cosmotools.create_cosmology`. If no cosmology is
        specified, a default :attr:`Planck13 cosmology
        <astropy.cosmology.Planck13>` will be used.
    power: numpy.ndarray, optional
        Power spectrum to use, which meet the criteria tested by
        :func:`randomfield.powertools.validate_power`. If not specified, the
        power spectrum will be calculated for the specified cosmology.
    verbose: bool, optional.
        Print a summary of this generator's parameters.
    """
    def __init__(self, nx, ny, nz, grid_spacing_Mpc_h,
                 cosmology=None, power=None, verbose=False):
        self.plan_c2r = transform.Plan(
            shape=(nx, ny, nz), dtype_in=np.complex64,
            packed=True, overwrite=True, inverse=True, use_pyfftw=True)
        self.plan_r2c = self.plan_c2r.create_reverse_plan(
            reuse_output=True, overwrite=True)
        self.grid_spacing_Mpc_h = grid_spacing_Mpc_h
        self.k_min, self.k_max = powertools.get_k_bounds(
            self.plan_c2r.data_in, grid_spacing_Mpc_h, packed=True)

        if cosmology is None:
            self.cosmology = cosmotools.create_cosmology()
            if power is None:
                power = powertools.load_default_power()
        else:
            self.cosmology = cosmology
            if power is None:
                power = cosmotools.calculate_power(
                    self.cosmology, k_min=self.k_min, k_max=self.k_max,
                    scaled_by_h=True)
        self.power = powertools.validate_power(power)

        self.redshifts = cosmotools.get_redshifts(
            self.cosmology, self.plan_c2r.data_out,
            spacing=self.grid_spacing_Mpc_h, scaled_by_h=True,
            z_axis=2).reshape(nz)
        self.redshift_to_index = scipy.interpolate.interp1d(
            self.redshifts, np.arange(nz), kind='linear', bounds_error=False)

        # Calculate angular spacing in degrees per transverse (x,y) grid unit,
        # indexed by position along the z-axis.
        DA = self.cosmology.comoving_transverse_distance(self.redshifts).value
        self.angular_spacing = np.zeros_like(self.redshifts)
        self.angular_spacing[1:] = ((180 / np.pi) *
            (self.grid_spacing_Mpc_h / self.cosmology.h) / DA[1:])
        self.z_max = self.redshifts.flat[-1]
        # Use the plane-parallel approximation here.
        self.x_fov = nx * self.angular_spacing.flat[-1]
        self.y_fov = ny * self.angular_spacing.flat[-1]

        self.verbose = verbose
        if self.verbose:
            Mb = (self.plan_c2r.nbytes_allocated +
                  self.plan_r2c.nbytes_allocated) / 2.0**20
            print('Allocated {0:.1f} Mb for {1} x {2} x {3} grid.'
                  .format(Mb, nx, ny, nz))
            print('{0} Mpc/h spacing covered by k = {1:.5f} - {2:.5f} h/Mpc.'
                  .format(self.grid_spacing_Mpc_h, self.k_min, self.k_max))
            print('Grid has z < {0:.3f} with {1:.4f} deg x {2:.4f} deg'
                  .format(self.z_max, self.x_fov, self.y_fov),
                  ' = {0:.4f} deg**2 field of view.'
                  .format(self.x_fov * self.y_fov))

    def generate_delta_field(self, smoothing_length_Mpc_h=0., seed=None):
        """
        Generate a delta-field realization.

        Parameters
        ----------
        smoothing_length : float, optional
            Length scale on which to smooth the generated delta field in Mpc/h.
            If not specified, no smoothing will be applied.
        seed: int, optional
            Random number seed to use. Specifying an explicit seed enables you
            to generate a reproducible delta field.  If no seed is specified,
            a randomized seed will be used.
        """
        powertools.fill_with_log10k(
            self.plan_c2r.data_in, spacing=self.grid_spacing_Mpc_h, packed=True)
        smoothed = powertools.filter_power(self.power, smoothing_length_Mpc_h)
        powertools.tabulate_sigmas(self.plan_c2r.data_in, power=smoothed,
                                   spacing=self.grid_spacing_Mpc_h, packed=True)
        random.randomize(self.plan_c2r.data_in, seed=seed)
        transform.symmetrize(self.plan_c2r.data_in, packed=True)
        delta = self.plan_c2r.execute()
        if self.verbose:
            print('Delta field has standard deviation {0:.3f}.'
                  .format(np.std(delta.flat)))
        return delta
