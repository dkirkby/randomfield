# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import print_function, division

from astropy.tests.helper import pytest
from ..transform import *
import numpy as np
from itertools import product


shape = (4, 6, 8)
packed_shape = (4, 6, 5)
seed = 123
TF = (True, False)
complex_types = (np.complex64, np.complex128)
float_types = (np.float32, np.float64)


def test_1d():
    for use_pyfftw in (False, True):
        buf = allocate(10, dtype=np.float32, use_pyfftw=use_pyfftw)
        assert buf.shape == (10,)
        assert buf.dtype == np.float32
        if use_pyfftw:
            try:
                import pyfftw
                assert pyfftw.n_byte_align(buf, 16) is buf
            except ImportError:
                pass


def test_3d():
    for use_pyfftw in (False, True):
        buf = allocate((4, 6, 8), dtype=np.complex64, use_pyfftw=use_pyfftw)
        assert buf.shape == (4, 6, 8)
        assert buf.dtype == np.complex64
        if use_pyfftw:
            try:
                import pyfftw
                assert pyfftw.n_byte_align(buf, 16) is buf
            except ImportError:
                pass


def test_c2r_view():
    for use_pyfftw in (False, True):
        buf1 = allocate((4, 6, 5), dtype=np.complex64, use_pyfftw=use_pyfftw)
        buf2 = buf1.view(np.float32).reshape(4, 6, 10)[:, :, :8]
        assert (buf2.base is buf1) or (buf2.base is buf1.base)


def test_r2c_view():
    for use_pyfftw in (False, True):
        buf1 = allocate((4, 6, 10), dtype=np.float32, use_pyfftw=use_pyfftw)
        buf2 = buf1.view(np.complex64).reshape(4, 6, 5)
        buf1 = buf1[:, :, :8]
        assert (buf2.base is buf1) or (buf2.base is buf1.base)


def test_good_types():
    assert scalar_type(np.complex64) == np.float32
    assert scalar_type(np.complex128) == np.float64
    assert complex_type(np.float32) == np.complex64
    assert complex_type(np.float64) == np.complex128
    assert scalar_type('complex') == np.float_
    assert complex_type('float') == np.complex_
    assert scalar_type(complex) == np.float_
    assert complex_type(float) == np.complex_


def test_bad_types():
    with pytest.raises(ValueError):
        scalar_type(np.float32)
    with pytest.raises(ValueError):
        complex_type(np.complex64)
    with pytest.raises(ValueError):
        scalar_type(np.int)
    with pytest.raises(ValueError):
        complex_type(np.int)


def test_c2c_result_type():
    for use_pyfftw, inverse, overwrite, dtype in \
        product(TF, TF, TF, complex_types):
        plan = Plan(shape=shape, dtype_in=dtype, inverse=inverse,
                    overwrite=overwrite, packed=False, use_pyfftw=use_pyfftw)
        assert plan.data_in.shape == shape
        assert plan.data_in.dtype == dtype
        assert plan.data_out.shape == shape
        assert plan.data_out.dtype == dtype
        if overwrite:
            assert plan.data_in is plan.data_out


def test_c2r_result_type():
    for use_pyfftw, overwrite, dtype_in in \
        product(TF, TF, complex_types):
        plan = Plan(shape=shape, dtype_in=dtype_in, inverse=True, packed=True,
                    overwrite=overwrite, use_pyfftw=use_pyfftw)
        assert plan.data_in.shape == packed_shape
        assert plan.data_in.dtype == dtype_in
        assert plan.data_out.shape == shape
        assert plan.data_out.dtype == scalar_type(dtype_in)
        if overwrite:
            assert ((plan.data_out.base is plan.data_in) or
                    (plan.data_out.base is plan.data_in.base))


def test_r2c_result_type():
    for use_pyfftw, overwrite, dtype_in in \
        product(TF, TF, float_types):
        plan = Plan(shape=shape, dtype_in=dtype_in, inverse=False, packed=True,
                    overwrite=overwrite, use_pyfftw=use_pyfftw)
        assert plan.data_in.shape == shape
        assert plan.data_in.dtype == dtype_in
        assert plan.data_out.shape == packed_shape
        assert plan.data_out.dtype == complex_type(dtype_in)
        if overwrite:
            assert ((plan.data_out.base is plan.data_in) or
                    (plan.data_out.base is plan.data_in.base))


def test_nbytes_allocated():
    nx, ny, nz = shape
    for use_pyfftw, overwrite, packed, dtype_in in \
        product(TF, TF, TF, complex_types):
        plan = Plan(shape=shape, dtype_in=dtype_in, inverse=True,
            packed=packed, overwrite=overwrite, use_pyfftw=use_pyfftw)
        item_size = np.dtype(dtype_in).itemsize
        print(dtype_in, item_size, packed, overwrite)
        if not packed:
            nbytes = item_size * nx * ny * nz
            if not overwrite:
                nbytes *= 2
        else:
            if overwrite:
                nbytes = item_size * nx * ny * (nz//2 + 1)
            else:
                nbytes = item_size * nx * ny * (nz + 1)
        assert nbytes == plan.nbytes_allocated
        # Check that a reverse plan does not allocate any new memory when
        # it reuses the original output.
        if packed and not overwrite:
            # Reverse plan cannot reuse original output in this case.
            continue
        plan_r = plan.create_reverse_plan(reuse_output=True, overwrite=True)
        assert plan_r.nbytes_allocated == 0


def test_is_hermitian():
    np.random.seed(seed)
    for use_pyfftw, overwrite, packed, ftype in \
        product(TF, TF, TF, float_types):
        plan = Plan(
            shape=shape, dtype_in=ftype if packed else complex_type(ftype),
            inverse=False, overwrite=overwrite, packed=packed,
            use_pyfftw=use_pyfftw)
        plan.data_in[:] = np.random.normal(size=plan.data_in.shape)
        result = plan.execute()
        assert is_hermitian(result, packed=packed)


def test_symmetrize():
    np.random.seed(seed)
    for use_pyfftw, overwrite, packed, ctype in \
        product(TF, TF, TF, complex_types):
        plan = Plan(
            shape=shape, dtype_in=ctype, inverse=True,
            overwrite=overwrite, packed=packed, use_pyfftw=use_pyfftw)
        real_size = 2 * plan.data_in.size
        real_dtype = scalar_type(ctype)
        plan.data_in.view(real_dtype).reshape(real_size)[:] = (
            np.random.normal(size=real_size))
        symmetrize(plan.data_in, packed=packed)
        assert is_hermitian(plan.data_in, packed=packed)
        result = plan.execute()
        assert np.allclose(result.imag, 0)


def test_c2c_round_trip():
    np.random.seed(seed)
    for use_pyfftw, inverse_first, overwrite_f, overwrite_r, dtype in \
        product(TF, TF, TF, TF, complex_types):
        plan_f = Plan(
            shape=shape, dtype_in=dtype, inverse=inverse_first,
            overwrite=overwrite_f, packed=False, use_pyfftw=use_pyfftw)
        plan_r = Plan(
            shape=shape, dtype_in=dtype, inverse=not inverse_first,
            overwrite=overwrite_r, packed=False, use_pyfftw=use_pyfftw)
        real_size = 2 * plan_f.data_in.size
        real_dtype = scalar_type(dtype)
        plan_f.data_in.view(real_dtype).reshape(real_size)[:] = (
            np.random.normal(size=real_size))
        original = np.copy(plan_f.data_in)
        plan_r.data_in[:] = plan_f.execute()
        result = plan_r.execute()
        assert np.allclose(original, result, atol=1e-6)


def test_c2c_reverse():
    np.random.seed(seed)
    for use_pyfftw, inverse_first, overwrite_f, overwrite_r, reuse, dtype in \
        product(TF, TF, TF, TF, TF, complex_types):
        plan_f = Plan(
            shape=shape, dtype_in=dtype, inverse=inverse_first,
            overwrite=overwrite_f, packed=False, use_pyfftw=use_pyfftw)
        plan_r = plan_f.create_reverse_plan(
            reuse_output=reuse, overwrite=overwrite_r)
        real_size = 2 * plan_f.data_in.size
        real_dtype = scalar_type(dtype)
        plan_f.data_in.view(real_dtype).reshape(real_size)[:] = (
            np.random.normal(size=real_size))
        original = np.copy(plan_f.data_in)
        plan_f.execute()
        if not reuse:
            plan_r.data_in[:] = plan_f.data_out
        result = plan_r.execute()
        assert np.allclose(original, result, atol=1e-6)


def test_c2r_round_trip():
    np.random.seed(seed)
    for use_pyfftw, overwrite_f, overwrite_r, dtype in \
        product(TF, TF, TF, complex_types):
        plan_f = Plan(
            shape=shape, dtype_in=dtype, inverse=True,
            packed=True, overwrite=overwrite_f, use_pyfftw=use_pyfftw)
        plan_r = Plan(
            shape=shape, dtype_in=scalar_type(dtype), inverse=False,
            packed=True, overwrite=overwrite_r, use_pyfftw=use_pyfftw)
        real_size = 2 * plan_f.data_in.size
        real_dtype = scalar_type(dtype)
        plan_f.data_in.view(real_dtype).reshape(real_size)[:] = (
            np.random.normal(size=real_size))
        symmetrize(plan_f.data_in, packed=True)
        original = np.copy(plan_f.data_in)
        plan_r.data_in[:] = plan_f.execute()
        result = plan_r.execute()
        assert np.allclose(original, result, atol=1e-6)


def test_c2r_reverse():
    np.random.seed(seed)
    for use_pyfftw, overwrite_f, overwrite_r, reuse, dtype in \
        product(TF, TF, TF, TF, complex_types):
        if reuse and not overwrite_f and overwrite_r:
            continue
        plan_f = Plan(
            shape=shape, dtype_in=dtype, inverse=True,
            packed=True, overwrite=overwrite_f, use_pyfftw=use_pyfftw)
        plan_r = plan_f.create_reverse_plan(
            reuse_output=reuse, overwrite=overwrite_r)
        real_size = 2 * plan_f.data_in.size
        real_dtype = scalar_type(dtype)
        plan_f.data_in.view(real_dtype).reshape(real_size)[:] = (
            np.random.normal(size=real_size))
        symmetrize(plan_f.data_in, packed=True)
        original = np.copy(plan_f.data_in)
        plan_f.execute()
        if not reuse:
            plan_r.data_in[:] = plan_f.data_out
        result = plan_r.execute()
        assert np.allclose(original, result, atol=1e-6)


def test_r2c_round_trip():
    np.random.seed(seed)
    for use_pyfftw, overwrite_f, overwrite_r, dtype in \
        product(TF, TF, TF, complex_types):
        plan_f = Plan(
            shape=shape, dtype_in=scalar_type(dtype), inverse=False,
            packed=True, overwrite=overwrite_f, use_pyfftw=use_pyfftw)
        plan_r = Plan(
            shape=shape, dtype_in=dtype, inverse=True,
            packed=True, overwrite=overwrite_r, use_pyfftw=use_pyfftw)
        plan_f.data_in[:] = np.random.normal(size=shape)
        original = np.copy(plan_f.data_in)
        plan_r.data_in[:] = plan_f.execute()
        result = plan_r.execute()
        assert np.allclose(original, result, atol=1e-6)


def test_r2c_reverse():
    np.random.seed(seed)
    for use_pyfftw, overwrite_f, overwrite_r, reuse, dtype in \
        product(TF, TF, TF, TF, complex_types):
        plan_f = Plan(
            shape=shape, dtype_in=scalar_type(dtype), inverse=False,
            packed=True, overwrite=overwrite_f, use_pyfftw=use_pyfftw)
        plan_r = plan_f.create_reverse_plan(
            reuse_output=reuse, overwrite=overwrite_r)
        plan_f.data_in[:] = np.random.normal(size=shape)
        original = np.copy(plan_f.data_in)
        plan_f.execute()
        if not reuse:
            plan_r.data_in[:] = plan_f.data_out
        result = plan_r.execute()
        assert np.allclose(original, result, atol=1e-6)


def test_methods_agree():
    """
    Verify that np.fft and pyfftw give the same results (when pyfft
    is available).
    """
    try:
        import pyfftw
        np.random.seed(seed)
        for inverse, overwrite, dtype in product(TF, TF, complex_types):
            for packed in (True, False) if inverse else (False,):
                plan1 = Plan(shape=shape, dtype_in=dtype, inverse=inverse,
                             overwrite=overwrite, packed=packed,
                             use_pyfftw=True)
                plan2 = Plan(shape=shape, dtype_in=dtype, inverse=inverse,
                             overwrite=overwrite, packed=packed,
                             use_pyfftw=False)
                real_size = 2 * plan1.data_in.size
                real_dtype = scalar_type(dtype)
                plan1.data_in.view(real_dtype).reshape(real_size)[:] = (
                    np.random.normal(size=real_size))
                if packed:
                    symmetrize(plan1.data_in, packed=True)
                plan2.data_in[:] = plan1.data_in
                result1 = plan1.execute()
                result2 = plan2.execute()
                assert np.allclose(result1, result2, atol=1e-7)
    except ImportError:
        pass
