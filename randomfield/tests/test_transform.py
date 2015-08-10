# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import print_function, division

from astropy.tests.helper import pytest
from ..transform import *
import numpy as np


shape = (4, 6, 8)
seed = 123


def test_transform_result_type():
    for use_pyfftw in (True, False):
        for inverse in (True, False):
            for dtype in (np.complex64, np.complex128):
                result = Plan(shape=shape, dtype=dtype, inverse=inverse,
                    use_pyfftw=use_pyfftw).execute()
                assert result.shape == shape
                assert result.dtype == dtype


def test_round_trip():
    np.random.seed(seed)
    for use_pyfftw in (True, False):
        for inverse_first in (True, False):
            for dtype in (np.complex64, np.complex128):
                plan_f = Plan(
                    shape=shape, dtype=dtype, inverse=inverse_first,
                    use_pyfftw=use_pyfftw)
                plan_r = Plan(
                    shape=shape, dtype=dtype, inverse=not inverse_first,
                    use_pyfftw=use_pyfftw)
                real_size = 2 * plan_f.data.size
                real_dtype = plan_f.data.real.dtype
                plan_f.data.view(real_dtype).reshape(real_size)[:] = (
                    np.random.normal(size=real_size))
                original = np.copy(plan_f.data)
                plan_r.data[:] = plan_f.execute()
                result = plan_r.execute()
                assert np.allclose(original, result, rtol=1e-4)


def test_full_symmetry():
    np.random.seed(seed)
    for use_pyfftw in (True, False):
        plan = Plan(
            shape=shape, dtype=np.complex64, inverse=False,
            use_pyfftw=use_pyfftw)
        plan.data[:] = np.random.normal(size=shape)
        result = plan.execute()
        assert is_hermitian(result)


def test_full_symmetrized():
    np.random.seed(seed)
    for use_pyfftw in (True, False):
        plan = Plan(
            shape=shape, dtype=np.complex64, inverse=True,
            use_pyfftw=use_pyfftw)
        real_size = 2 * plan.data.size
        real_dtype = plan.data.real.dtype
        plan.data.view(real_dtype).reshape(real_size)[:] = (
            np.random.normal(size=real_size))
        symmetrize(plan.data)
        assert is_hermitian(plan.data)
        result = plan.execute()
        assert np.allclose(result.imag, 0)
