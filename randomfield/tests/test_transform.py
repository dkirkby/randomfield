# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import print_function, division

from astropy.tests.helper import pytest
from ..transform import *
import numpy as np
from itertools import product


shape = (4, 6, 8)
seed = 123
TF = (True, False)
complex_types = (np.complex64, np.complex128)


def test_c2c_result_type():
    for use_pyfftw, inverse, overwrite, dtype in \
        product(TF, TF, TF, complex_types):
        result = Plan(shape=shape, in_dtype=dtype, inverse=inverse,
            overwrite=overwrite, use_pyfftw=use_pyfftw).execute()
        assert result.shape == shape
        assert result.dtype == dtype


def test_c2c_round_trip():
    np.random.seed(seed)
    for use_pyfftw, inverse_first, overwrite, dtype in \
        product(TF, TF, TF, complex_types):
        plan_f = Plan(
            shape=shape, in_dtype=dtype, inverse=inverse_first,
            overwrite=overwrite, use_pyfftw=use_pyfftw)
        plan_r = Plan(
            shape=shape, in_dtype=dtype, inverse=not inverse_first,
            overwrite=overwrite, use_pyfftw=use_pyfftw)
        real_size = 2 * plan_f.data_in.size
        real_dtype = plan_f.data_in.real.dtype
        plan_f.data_in.view(real_dtype).reshape(real_size)[:] = (
            np.random.normal(size=real_size))
        original = np.copy(plan_f.data_in)
        plan_r.data_in[:] = plan_f.execute()
        result = plan_r.execute()
        assert np.allclose(original, result, rtol=1e-4), (
            use_pyfftw, inverse_first, dtype)


def test_full_symmetry():
    np.random.seed(seed)
    for use_pyfftw, overwrite in product(TF, TF):
        plan = Plan(
            shape=shape, in_dtype=np.complex64, inverse=False,
            overwrite=overwrite, use_pyfftw=use_pyfftw)
        plan.data_in[:] = np.random.normal(size=shape)
        result = plan.execute()
        assert is_hermitian(result)


def test_full_symmetrized():
    np.random.seed(seed)
    for use_pyfftw, overwrite in product(TF, TF):
        plan = Plan(
            shape=shape, in_dtype=np.complex64, inverse=True,
            overwrite=overwrite, use_pyfftw=use_pyfftw)
        real_size = 2 * plan.data_in.size
        real_dtype = plan.data_in.real.dtype
        plan.data_in.view(real_dtype).reshape(real_size)[:] = (
            np.random.normal(size=real_size))
        symmetrize(plan.data_in)
        assert is_hermitian(plan.data_in)
        result = plan.execute()
        assert np.allclose(result.imag, 0)
