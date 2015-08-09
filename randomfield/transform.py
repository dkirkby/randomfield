# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np

import memory


def is_symmetric_full(data, rtol=1e-08, atol=1e-08):
    """
    Test if a 3D field has the symmetries required for a real inverse FFT.
    """
    nx, ny, nz = data.shape
    if nx % 2 or ny % 2 or nz % 2:
        raise ValueError('Array dimensions must all be even.')
    for ix in range(nx):
        jx = (nx - ix) % nx
        for iy in range(ny):
            jy = (ny - iy) % ny
            for iz in range(nz):
                jz = (nz - iz) % nz
                if not np.allclose(
                    data[ix, iy, iz], np.conj(data[jx, jy, jz]), rtol, atol):
                    return False
    return True


def symmetrize_full(data):
    """
    Symmetrize a complex 3D field so that its inverse FFT is real valued.
    """
    nx, ny, nz = data.shape
    if nx % 2 or ny % 2 or nz % 2:
        raise ValueError('Array dimensions must all be even.')


class Plan(object):
    """
    A plan for performing fast Fourier transforms on a single buffer.
    """
    def __init__(self, shape, dtype=np.float32, inverse=True, use_pyfftw=True):
        self.data = memory.allocate(shape, dtype, use_pyfftw=use_pyfftw)
        self.inverse = inverse
        self.fftw_plan = None
        if use_pyfftw:
            try:
                import pyfftw
                self.fftw_plan = pyfftw.FFTW(self.data, self.data,
                    direction=('FFTW_BACKWARD' if inverse else 'FFTW_FORWARD'),
                    axes=(0, 1, 2))
            except ImportError:
                pass

    def execute(self):
        if self.fftw_plan is None:
            # This probably creates a temporary for the RHS, thus doubling
            # the peak memory requirements.  Any way to confirm this?
            if self.inverse:
                self.data[:] = np.fft.ifftn(self.data, axes=(0,1,2))
            else:
                self.data[:] = np.fft.fftn(self.data, axes=(0,1,2))
        else:
            self.fftw_plan.execute()
            self.data /= np.sqrt(self.data.size)
        return self.data
