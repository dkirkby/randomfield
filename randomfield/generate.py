# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
High-level functions to generate random fields.
"""
from __future__ import print_function, division

import numpy as np

import transform
import powertools
import random


def generate(nx, ny, nz, spacing, power, seed):
    """
    Generate a Gaussian random field with a specified power spectrum.

    Note that the results are not guaranteed to be identical with the same
    seed because FFTW does not always use the same algorithm.
    """
    plan = transform.Plan(
        shape=(nx, ny, nz), dtype_in=np.complex64,
        packed=True, overwrite=True, inverse=True, use_pyfftw=True)
    powertools.fill_with_log10k(plan.data_in, spacing=spacing, packed=True)
    powertools.tabulate_sigmas(plan.data_in, power=power,
                               spacing=spacing, packed=True)
    random.randomize(plan.data_in, seed=seed)
    transform.symmetrize(plan.data_in, packed=True)
    return plan.execute()
