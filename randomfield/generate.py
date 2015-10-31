# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
High-level functions to generate random fields.
"""
from __future__ import print_function, division

import random

import numpy as np
import scipy.interpolate
import scipy.integrate

import astropy.units as u
import astropy.constants

import transform
import powertools
import cosmotools


class Generator(object):
    """
    Manage random field generation for a specified geometry.

    The constructor allocates the (potentially large) memory buffer required
    to store field values but does not initialize it.  You will normally
    initialize the field using :meth:`generate_delta_field`.

    The optional ``classy`` package is used to calculate the power spectrum
    of an arbitrary cosmology on the fly.  If ``classy`` is not installed,
    you can either use the default Planck13 cosmology (do not set the
    ``cosmology`` or ``power`` parameters) or else specify your own cosmology
    (using :func:`create_cosmology <randomfield.cosmotools.create_cosmology>`
    for simple cases or else :mod:`astropy.cosmology`) and provided your own
    externally calculated tabulated power spectrum (from CAMB, for example).

    The optional ``matplotlib`` package is used to draw 2D slices of the
    3D field after each processing step. The default options to generator
    methods do not create any plots, but plots can be enabled using each
    method's ``show_plot`` and ``save_plot_name`` options if ``matplotlib`` is
    available.

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
    grid_spacing_Mpc_h : float
        Uniform grid spacing in Mpc/h.
    num_plot_sections : int
        Number of sections to split the z-axis into for slice plots.
    cosmology : :class:`astropy.cosmology.FLRW`, optional
        Homogeneous background cosmology to use for distances and
        for calculating the power spectrum of inhomogeneities. Should be
        an instance of :class:`astropy.cosmology.FLRW`. Simple cases can
        be conveniently created using
        :func:`randomfield.cosmotools.create_cosmology`. If no cosmology is
        specified, a default :attr:`Planck13 cosmology
        <astropy.cosmology.Planck13>` will be used.
    power : numpy.ndarray, optional
        Power spectrum to use, which meets the criteria tested by
        :func:`randomfield.powertools.validate_power`. If not specified, the
        power spectrum will be calculated for the specified cosmology
        using the optional ``classy`` package.
    verbose : bool, optional.
        Print a summary of this generator's parameters.
    """
    def __init__(self, nx, ny, nz, grid_spacing_Mpc_h, num_plot_sections=4,
                 cosmology=None, power=None, verbose=False):
        self.plan_c2r = transform.Plan(
            shape=(nx, ny, nz), dtype_in=np.complex64,
            packed=True, overwrite=True, inverse=True, use_pyfftw=True)
        self.plan_r2c = self.plan_c2r.create_reverse_plan(
            reuse_output=True, overwrite=True)
        self.grid_spacing_Mpc_h = grid_spacing_Mpc_h
        self.k_min, self.k_max = powertools.get_k_bounds(
            self.plan_c2r.data_in, grid_spacing_Mpc_h, packed=True)
        self.potential = None

        if nz % num_plot_sections != 0:
            raise ValueError(
                'Z-axis does not evenly divided into {0} plot sections.'
                .format(num_plot_sections))
        self.num_plot_sections = num_plot_sections

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

        # Tabulate the comoving distance DC and transverse separation DA along
        # the line of sight in units of Mpc/h.
        self.DC = np.arange(nz) * self.grid_spacing_Mpc_h
        self.DA = (self.cosmology.comoving_transverse_distance(self.redshifts)
            .to(u.Mpc).value) * self.cosmology.h

        # Calculate angular spacing in transverse (x,y) grid units per degree,
        # indexed by position along the z-axis.
        self.angular_spacing = (
            self.DA * (np.pi / 180.) / self.grid_spacing_Mpc_h)
        self.z_max = self.redshifts.flat[-1]
        # Use the plane-parallel approximation here.
        self.x_fov = nx / self.angular_spacing.flat[-1]
        self.y_fov = ny / self.angular_spacing.flat[-1]

        self.growth_function = cosmotools.get_growth_function(
            self.cosmology, self.redshifts)
        self.mean_matter_density = cosmotools.get_mean_matter_densities(
            self.cosmology, self.redshifts)

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

    def generate_delta_field(self, smoothing_length_Mpc_h=0., seed=None,
                             save_potential=True,
                             show_plot=False, save_plot_name=None):
        """
        Generate a delta-field realization.

        The delta field is calculated at redshift zero and sampled from a
        distribution with mean zero and k-space variance proportional to
        the smoothed power spectrum.

        No new memory is allocated unless the ``save_potential`` option is
        selected.

        Parameters
        ----------
        smoothing_length : float, optional
            Length scale on which to smooth the generated delta field in Mpc/h.
            If not specified, no smoothing will be applied.
        seed : int, optional
            Random number seed to use. Specifying an explicit seed enables you
            to generate a reproducible delta field.  If no seed is specified,
            a randomized seed will be used.
        save_potential : bool, optional
            Save the k-space field delta(kx,ky,kz) / k**2 so that it can be
            used for later calculations of the lensing potential or the
            bulk velocity vector field. The first time this option is used,
            additional memory is allocated, approximately doubling the total
            memory usage.
        show_plot : bool, optional
            Show a (y,z) slice through the generated delta field using the
            optional matplotlib library. The plot will need to be dismissed
            after it displays before the program continues.  Use the
            ``save_plot_name`` option to generate and save the plot without
            requiring any user interaction.
        save_plot_name : str, optional
            Name of a file where the generated delta field slice plot should
            be saved.  The file extension provided determines the image file
            format that will be used. This option can be used with ``show_plot``
            either ``True`` or ``False``.

        Returns
        -------
        numpy.narray
            3D numpy array of delta field values. The returned array is a
            view into our internal memory buffer and will be overwritten by
            subsequent operations.
        """
        powertools.fill_with_log10k(
            self.plan_c2r.data_in, spacing=self.grid_spacing_Mpc_h, packed=True)
        self.smoothed_power = powertools.filter_power(
            self.power, smoothing_length_Mpc_h)
        powertools.tabulate_sigmas(
            self.plan_c2r.data_in, power=self.smoothed_power,
            spacing=self.grid_spacing_Mpc_h, packed=True)
        random.randomize(self.plan_c2r.data_in, seed=seed)
        transform.symmetrize(self.plan_c2r.data_in, packed=True)
        if save_potential:
            # Fill self.potential with values of k**2.
            if self.potential is None:
                self.potential = np.empty_like(self.plan_c2r.data_in)
            self.potential.imag = 0.
            kx2_grid, ky2_grid, kz2_grid = powertools.create_ksq_grids(
                self.potential, spacing=self.grid_spacing_Mpc_h, packed=True)
            np.add(kx2_grid, ky2_grid, out=self.potential.real)
            self.potential.real += kz2_grid
            # Replace k**2 with 1 / k**2 except at k=0.
            old_settings = np.seterr(divide='ignore')
            np.reciprocal(self.potential.real, out=self.potential.real)
            np.seterr(**old_settings)
            self.potential[0, 0, 0] = 0.
            # Multiply by delta(k).
            self.potential *= self.plan_c2r.data_in
        else:
            self.potential = None
        delta = self.plan_c2r.execute()
        self.delta_field_rms = np.std(delta.flat)
        if self.verbose:
            print('Delta field has standard deviation {0:.3f}.'
                  .format(self.delta_field_rms))

        if show_plot or save_plot_name is not None:
            self.plot_slice(
                show_plot=show_plot, save_plot_name=save_plot_name,
                clip_symmetric=True, label='Matter inhomogeneity ' +\
                    '$\delta(r) = \\rho(r)/\overline{\\rho} - 1$')

        return delta

    def convert_delta_to_density(self, apply_lognormal_transform=True,
                                 show_plot=False, save_plot_name=None):
        """
        Convert a delta field into a density field with light-cone evolution.

        Results are in units of g / cm**3.  The density at each grid point is
        calculated at a lookback time equal to its distance from the observer.
        We use the plane-parallel approximation.

        Parameters
        ----------
        apply_lognormal_transform : bool, optional
            Use :meth:`randomfield.cosmotools.apply_lognormal_transform` to
            transform the distribution of density fluctuations so that all
            densities are positive.
        show_plot : bool, optional
            Show a (y,z) slice through the calculated density field using the
            optional matplotlib library. The plot will need to be dismissed
            after it displays before the program continues.  Use the
            ``save_plot_name`` option to generate and save the plot without
            requiring any user interaction.
        save_plot_name : str, optional
            Name of a file where the calculated density field slice plot should
            be saved.  The file extension provided determines the image file
            format that will be used. This option can be used with ``show_plot``
            either ``True`` or ``False``.

        Returns
        -------
        numpy.ndarray
            3D numpy array of light-cone density field values in g/cm**3.
            The returned array is a view into our internal memory buffer and
            will be overwritten by subsequent operations.
        """
        delta = self.plan_c2r.data_out
        if apply_lognormal_transform:
            delta = cosmotools.apply_lognormal_transform(
                delta, self.growth_function, sigma=self.delta_field_rms)
        else:
            delta *= self.growth_function
            delta += 1
        delta *= self.mean_matter_density

        if show_plot or save_plot_name is not None:
            self.plot_slice(
                show_plot=show_plot, save_plot_name=save_plot_name,
                label='Matter density $\\rho(r)$ (g/cm$^3$)', clip_percent=2.)

        return delta

    def calculate_newtonian_potential(self, light_cone=True,
                                      show_plot=False, save_plot_name=None):
        """
        Calculate the Newtonian potential Phi(r).

        Variations in the potential at z=0 are calculated as the inverse
        Fourier transform of:

        .. math::

            \delta\\tilde{\Phi}(\\vec{k},0) = -\\frac{1}{k^2}\,
            \\frac{3}{2} H_0^2 \Omega_m\, \\tilde{\delta}(\\vec{k},0)

        The redshift evolution is calculated as:

        .. math::

            \delta\\Phi(\\vec{r},z) =
            \\frac{G(z)}{1+z}\, \delta\\Phi(\\vec{r}, 0)

        This method must be called after :meth:`generate_delta_field` using
        the ``save_potential`` option set to ``True``.  The saved potential
        is not modified so this method can be called multiple times with
        different options.

        Parameters
        ----------
        light_cone : bool, optional
            Return the potential on our light cone when this is True, or else
            at redshift zero.
        show_plot : bool, optional
            Show a (y,z) slice through the calculated potential using the
            optional matplotlib library. The plot will need to be dismissed
            after it displays before the program continues.  Use the
            ``save_plot_name`` option to generate and save the plot without
            requiring any user interaction.
        save_plot_name : str, optional
            Name of a file where the calculated potential slice plot should
            be saved.  The file extension provided determines the image file
            format that will be used. This option can be used with ``show_plot``
            either ``True`` or ``False``.

        Returns
        -------
        numpy.ndarray
            3D numpy array of Newtonian potential values in units of s**-2.
            The returned array is a view into our internal memory buffer and
            will be overwritten by subsequent operations.
        """
        if self.potential is None:
            raise RuntimeError('No saved potential field.')
        self.plan_c2r.data_in[:] = self.potential

        # We are working in Mpc/h so H0 = 100 (km/s)/Mpc
        H0 = 100 * (u.km / u.s) / u.Mpc
        scale = (-1.5 * H0**2 * self.cosmology.Om0).to(u.s**-2).value
        self.plan_c2r.data_in *= scale
        field = self.plan_c2r.execute()

        if light_cone:
            field *= self.growth_function
            field /= 1 + self.redshifts

        if show_plot or save_plot_name is not None:
            self.plot_slice(clip_symmetric=True,
                show_plot=show_plot, save_plot_name=save_plot_name,
                label='Newtonian potential $\Phi(r)$ [s$^{-2}$]')

        return field


    def calculate_lensing_potential(self, i_min=None, show_plot=False,
                                    save_plot_name=None):
        """
        Calculate the lensing potential psi(r).

        The calculation is:

        .. math::

            \psi(\\vec{r}, z) = \int_{D_{min}}^{D_{src}} \left[
            \cot_K(D) - \cot_K(D_{src}) \\right] \delta\Phi(\\vec{r}, z)

        You will normally call :func:`calculate_newtonian_potential`
        with ``light_cone=True`` just before this.

        The ``i_min`` argument truncates the lensing potential integral
        at a minimum comoving distance :math:`D_{min}` from the observer
        equal to ``self.DC[i_min]``.  The default is 1/32 of the grid size
        along the line of sight.
        """
        nDC = self.DC.size
        if i_min is None:
            i_min = nDC // 32
        if i_min < 0 or i_min >= nDC:
            raise ValueError(
                'Invalid i_min {}. Expected 0 - {}.'.format(i_min, nDC - 1))

        # Calculate the curvature constant K in units of (Mpc/h)**-2.
        clight_km_s = astropy.constants.c.to(u.km / u.s).value
        H0 = 100. # in Mpc/h units
        K = -self.cosmology.Ok0 * (H0 / clight_km_s)**2
        # Tabulate cosK(D), which is dimensionless.
        if K < 0:
            sqrt_abs_K = np.sqrt(-K)
            cosK =  np.cosh(sqrt_abs_K * self.DC)
        elif K > 0:
            sqrt_K = np.sqrt(K)
            cosK = np.cos(sqrt_K * self.DC)
        else:
            cosK = np.ones_like(self.DA, dtype=float)
        # Tabulate cotK(D), which is also dimensionless.
        cotK = np.ones_like(cosK)
        cotK[1:] = cosK[1:] / self.DA[1:]

        # Allocate memory for the result.
        dPhi = self.plan_c2r.data_out
        psi = np.empty_like(dPhi)

        # Loop over redshift slices, starting from the most distant
        # and working backwards.
        for i in range(nDC, i_min, -1):
            # Overwrite psi with the integrand for this slice.
            psi[:, :, i_min:i] = dPhi[:, :, i_min:i]
            psi[:, :, i_min:i] *= -2 * (cotK[i_min:i] - cotK[i-1])
            # Integrate the result up to this slice and save the result in the slice itself.
            psi[:, :, i-1] = scipy.integrate.simps(
                y=psi[:, :, i_min:i], x=self.DC[i_min:i], axis=-1)
        if i_min > 0:
            psi[:, :, :i_min] = 0.

        if show_plot or save_plot_name is not None:
            self.plot_slice(field=psi, clip_percent=15.0, clip_symmetric=True,
                show_plot=show_plot, save_plot_name=save_plot_name,
                label='Lensing potential $\psi(r)$ [(Mpc/h) s$^{-2}$]')

        return psi

    def plot_slice(self, field=None, slice_index=0, figure_width=10.,
                   label='Field values', cmap='jet',
                   clip_percent=1.0, clip_symmetric=False, axis_dz=0.1,
                   field_of_view_deg=3.5, show_plot=False, save_plot_name=None):
        """
        Plot a 2D slice of the most recently calculated real-valued field, with
        the redshift direction (axis=2) displayed horizontally and the
        declination direction (axis=1) displayed vertically.  The plot is
        divided into ``num_sections`` sections in the redshift direction and
        a histogram of all field values is displayed using the colormap.

        This function will fail with an ImportError if matplotlib is not
        installed.

        You do not normally need to call this function directly.  Instead, it
        is invoked using the ``show_plot`` and ``save_plot_name`` options to
        other generator methods.
        """
        if not show_plot and save_plot_name is False:
            return

        import matplotlib.pyplot as plt
        import matplotlib.gridspec
        import matplotlib.colors

        if field is None:
            field = self.plan_c2r.data_out
        redshifts = self.redshifts
        zfunc = self.redshift_to_index
        num_sections = self.num_plot_sections

        ny, nz = field.shape[1:3]
        sections = np.linspace(0, nz, 1 + num_sections, dtype=int)
        x_limits = (0, nz // num_sections - 1)
        y_center = 0.5 * (ny - 1)
        y_limits = (0, ny - 1)

        vmin, vmax = np.percentile(
            field[slice_index], (0.5*clip_percent, 100 - 0.5*clip_percent))
        if clip_symmetric:
            vlim = max(abs(vmin), abs(vmax))
            vmin, vmax = -vlim, +vlim
        cmap = plt.get_cmap(cmap)

        wpad, hpad = 8, 16
        height = num_sections * (ny + hpad)
        width = (nz//num_sections + wpad) / 0.8
        figure_height = height * (figure_width / width)
        plt.figure(figsize=(figure_width, figure_height))

        lhs = matplotlib.gridspec.GridSpec(num_sections, 1)
        lhs.update(left=0.02, right=0.75, top=0.98, bottom=0.02,
                   hspace=0., wspace=0.)

        for i in range(num_sections):
            iz1, iz2 = sections[i:i+2]
            ax = plt.subplot(lhs[i, 0])
            ax.imshow(field[slice_index, :, iz1:iz2], vmin=vmin, vmax=vmax)

            x12 = np.arange(iz1, iz2)
            dy = 0.5 * field_of_view_deg * self.angular_spacing[x12]
            plt.fill_between(x12 - iz1, y_center + dy, ny, facecolor='black',
                alpha=0.2, edgecolor='none')
            plt.fill_between(x12 - iz1, y_center - dy, 0, facecolor='black',
                alpha=0.2, edgecolor='none')
            plt.plot(x12 - iz1, y_center + dy, 'w-')
            plt.plot(x12 - iz1, y_center - dy, 'w-')
            plt.xlim(*x_limits)
            plt.ylim(*y_limits)

            z1, z2 = redshifts[[iz1, iz2-1]]
            iz_min = np.ceil(z1/axis_dz)
            iz_max = np.floor(z2/axis_dz)
            z_tick_values = np.arange(iz_min, iz_max + 1) * axis_dz
            z_tick_positions = zfunc(z_tick_values) - iz1
            plt.xticks(z_tick_positions, z_tick_values)

            plt.gca().yaxis.set_visible(False)

        rhs = plt.axes([0.82, 0.02, 0.16, 0.96])
        bin_counts, bin_edges, bin_patches = rhs.hist(field.reshape(field.size),
            bins=200, range=(vmin,vmax), orientation='horizontal')
        plt.ylim(vmin, vmax)
        plt.ylabel(
            label, rotation=-90., verticalalignment='top', fontsize='large')
        plt.grid(axis='y')
        norm = matplotlib.colors.Normalize(vmin, vmax)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        plt.setp(bin_patches, 'edgecolor', 'none')
        for x, p in zip(bin_centers, bin_patches):
            p.set_facecolor(cmap(norm(x)))
        plt.gca().xaxis.set_visible(False)

        if save_plot_name:
            plt.savefig(save_plot_name)
        if show_plot:
            plt.show()
