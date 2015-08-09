# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np

def allocate(shape, dtype):
    """
    Allocate a contiguous block of un-initialized typed memory.

    If the pyfftw module is importable, allocates 16-byte aligned memory using
    for improved SIMD instruction performance. Otherwise, uses the
    :func:`numpy.empty` function.  When shape is multi-dimensional, the
    returned memory is initialized for C (row-major) storage order.

    Parameters
    ----------
    shape : int or tuple of ints
        Shape of the empty array to allocate.
    dtype : numpy data-type
        Data type to assign to the empty array.

    Returns
    -------
    out : numpy array
        Array of un-initialized data with the requested shape and data type.
        The storage order of multi-dimensional arrays is always C-type
        (row-major).
    """
    return np.empty(shape, dtype, order='C')
