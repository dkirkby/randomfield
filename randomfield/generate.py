# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
High-level functions to generate random fields.
"""
from __future__ import print_function, division

import numpy as np

import transform
import powertools
import random


class Generator(object):
    """
    Manage random field generation and processing.
    """
    def __init__(self, nx, ny, nz, spacing):
        self.plan_c2r = transform.Plan(
            shape=(nx, ny, nz), dtype_in=np.complex64,
            packed=True, overwrite=True, inverse=True, use_pyfftw=True)
        self.plan_r2c = self.plan_c2r.create_reverse_plan(
            reuse_output=True, overwrite=True)
        self.spacing = spacing

    def generate_delta_field(self, power, seed):
        """
        Generate a delta-field realization.
        """
        powertools.fill_with_log10k(
            self.plan_c2r.data_in, spacing=self.spacing, packed=True)
        powertools.tabulate_sigmas(self.plan_c2r.data_in, power=power,
                                   spacing=self.spacing, packed=True)
        random.randomize(self.plan_c2r.data_in, seed=seed)
        transform.symmetrize(self.plan_c2r.data_in, packed=True)
        return self.plan_c2r.execute()
