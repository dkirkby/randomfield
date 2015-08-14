# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Generate random numbers.
"""
from __future__ import print_function, division

import numpy as np

import transform


def randomize(data, seed=None):
    """
    Randomize data by multiplying existing sigma values by normal deviates.
    """
    # Replicate the sigma values in data.real so they are in data.imag as well.
    data.imag = data.real
    # Generate a view of the real and imaginary parts as a 1D array of
    # real-valued sigmas.
    real_type = transform.scalar_type(data.dtype)
    real_size = 2 * data.size
    sigmas = data.view(real_type).reshape(real_size)
    # Seed the random generator state without disturbing the default state.
    generator = np.random.RandomState(seed)
    # Scale each sigma by a normal deviate.  Should break this up into
    # chunks to avoid a large temporary value, but do the simplest thing
    # for now.
    sigmas *= generator.normal(size=real_size)
    return data
