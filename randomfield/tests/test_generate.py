# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import print_function, division

from astropy.tests.helper import pytest
from ..generate import *
import numpy as np
from scipy.special import erf

from ..cosmotools import create_cosmology

spacing = 2.5
nxyz = 64
seed = 123

def test_generate():
    generator = Generator(nxyz, nxyz, nxyz, spacing)
    data = generator.generate_delta_field(seed=seed)
    assert data.shape == (nxyz, nxyz, nxyz)
    assert data.dtype == np.float32
    assert abs(np.mean(data)) < 1e-3


def test_gaussian_variance():
    """
    Tabulate a Gaussian power spectrum::

        P(k) = P0*exp(-(k*sigma)**2/2)

    corresponding to a real-space Gaussian smoothing of white noise, with
    correlation function::

        xi(r) = P0*exp(-r**2/(2*sigma**2))/(2*pi)**(3/2)/sigma**3

    The corresponding variance integrated over a grid with
    kmin <= kx,ky,kz <= kmax is::

        P0/(2*pi)**(3/2)/sigma**3 *
            (erf(kmax*sigma/sqrt(2))**3 - erf(kmin*sigma/sqrt(2))**3)
    """
    kmin = (2 * np.pi) / (spacing * nxyz)
    kmax = np.pi / spacing
    sigma = 2.5 * spacing
    P0 = 1.23

    calculated_var = P0 / (2 * np.pi)**1.5 / sigma**3 * (
        erf(kmax * sigma / np.sqrt(2))**3 -
        erf(kmin * sigma / np.sqrt(2))**3)

    power = np.empty(100, dtype=[('k', float), ('Pk', float)])
    power['k'] = np.linspace(kmin, np.sqrt(3)*kmax, len(power))
    power['Pk'] = P0 * np.exp(-0.5 * (power['k'] * sigma)**2)

    ntrials = 10
    measured_var = 0
    generator = Generator(nxyz, nxyz, nxyz, spacing, power=power)
    for trial in range(ntrials):
        data = generator.generate_delta_field(seed=seed + trial)
        measured_var += np.var(data)
    measured_var /= ntrials

    assert abs(measured_var - calculated_var) < 0.01 * calculated_var
