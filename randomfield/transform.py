# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import print_function, division

import numpy as np

import memory


def expanded_shape(data, packed=False):
    nx, ny, nz = data.shape
    if nx % 2 or ny % 2:
        raise ValueError('First two dimensions of array must be even.')
    if packed:
        if nz % 2 == 0:
            raise ValueError('Last dimension of packed array must be odd.')
        nz = 2 * (nz - 1)
    else:
        if nz % 2:
            raise ValueError('Last dimension of unpacked array must be even.')
    return nx, ny, nz


def is_hermitian(data, packed=False, rtol=1e-08, atol=1e-08):
    """
    Test if a 3D field is Hermitian.
    """
    nx, ny, nz = expanded_shape(data)
    if packed:
        z_range = [0, nz//2 + 1]
    else:
        z_range = range(nz//2 + 1)
    for ix in range(nx//2 + 1):
        jx = (nx - ix) % nx
        for iy in range(ny//2 + 1):
            jy = (ny - iy) % ny
            for iz in z_range:
                jz = (nz - iz) % nz
                if not np.allclose(
                    data[ix, iy, iz], np.conj(data[jx, jy, jz]), rtol, atol):
                    return False
    return True


def symmetrize(data, packed=False):
    """
    Symmetrize a complex 3D field so that its inverse FFT is real valued.
    """
    nx, ny, nz = expanded_shape(data)
    xlo, xhi = slice(1, nx//2), slice(nx-1, nx//2, -1)
    ylo, yhi = slice(1, ny//2), slice(ny-1, ny//2, -1)

    if not packed:
        zlo, zhi = slice(1, nz//2), slice(nz-1, nz//2, -1)
        # Symmetrize in the 3D volume where none of x,y,z is either 0 or n/2.
        data[xhi,yhi,zhi] = np.conj(data[xlo,ylo,zlo])
        data[xlo,yhi,zhi] = np.conj(data[xhi,ylo,zlo])
        data[xhi,ylo,zhi] = np.conj(data[xlo,yhi,zlo])
        data[xhi,yhi,zlo] = np.conj(data[xlo,ylo,zhi])
        # Symmetrize in the 2D planes where x is 0 or nx/2.
        data[[0,nx//2],ylo,zhi] = np.conj(data[[0,nx//2],yhi,zlo])
        data[[0,nx//2],yhi,zhi] = np.conj(data[[0,nx//2],ylo,zlo])
        # Symmetrize in the 2D planes where y is 0 or ny/2.
        data[xlo,[0,ny//2],zhi] = np.conj(data[xhi,[0,ny//2],zlo])
        data[xhi,[0,ny//2],zhi] = np.conj(data[xlo,[0,ny//2],zlo])
        # Symmetrize along the 4 edges where x and y are 0 or n/2 and
        # z is not equal to 0 or n/2.
        data[[0,0,nx//2,nx//2],[0,ny//2,0,ny//2],zhi] = (
            np.conj(data[[0,0,nx//2,nx//2],[0,ny//2,0,ny//2],zlo]))

    # Symmetrize in the 2D planes where z is 0 or nz/2.
    data[xlo,yhi,[0,nz//2]] = np.conj(data[xhi,ylo,[0,nz//2]])
    data[xhi,yhi,[0,nz//2]] = np.conj(data[xlo,ylo,[0,nz//2]])

    # Symmetrize along the 4 edges where x and z are 0 or n/2 and
    # y is not equal to 0 or n/2.
    data[[0,0,nx//2,nx//2],yhi,[0,nz//2,0,nz//2]] = (
        np.conj(data[[0,0,nx//2,nx//2],ylo,[0,nz//2,0,nz//2]]))
    # Symmetrize along the 4 edges where y and z are 0 or n/2 and
    # x is not equal to 0 or n/2.
    data[xhi,[0,0,ny//2,ny//2],[0,nz//2,0,nz//2]] = (
        np.conj(data[xlo,[0,0,ny//2,ny//2],[0,nz//2,0,nz//2]]))
    # Symmetrize the 8 vertices where all of x,y,z are 0 or n/2,
    # so elements must be real.
    data.imag[[0,0,0,0,nx//2,nx//2,nx//2,nx//2],
              [0,0,ny//2,ny//2,0,0,ny//2,ny//2],
              [0,nz//2,0,nz//2,0,nz//2,0,nz//2]] = 0
    # Set the DC component with |k|=0 to zero.
    data.real[0,0,0] = 0


class Plan(object):
    """
    A plan for performing fast Fourier transforms on a single buffer.
    """
    def __init__(self, shape, dtype=np.float32,
                 inverse=True, packed=False, use_pyfftw=True):
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
