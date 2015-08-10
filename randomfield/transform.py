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
    def __init__(self, shape, in_dtype=np.complex64, overwrite=True,
                 inverse=True, packed=False, use_pyfftw=True):
        try:
            nx, ny, nz = shape
        except (TypeError, ValueError):
            raise ValueError('Expected 3D shape.')
        if nx % 2 or ny % 2 or nz % 2:
            raise ValueError('All shape dimensions must be even.')

        # Convert in_dtype to an object in the numpy scalar type hierarchy.
        in_dtype = np.obj2sctype(in_dtype)
        if in_dtype is None:
            raise ValueError('Invalid in_dtype: {0}.'.format(in_dtype))

        # Determine the input and output array type and shape.
        if packed:
            if inverse:
                in_shape = (nx, ny, nz + 2)
                if not issubclass(in_dtype, np.complexfloating):
                    raise ValueError(
                        'Invalid in_dtype for inverse packed transform ' +
                        '(should be complex): {0}.'.format(in_dtype))
                out_dtype = in_dtype().real.dtype
                out_shape = (nx, ny, nz//2 + 1)
            else:
                in_shape = (nx, ny, nz//2 + 1)
                if not issubclass(in_dtype, np.floating):
                    raise ValueError(
                        'Invalid in_dtype for forward packed transform ' +
                        '(should be floating): {0}.'.format(in_dtype))
                out_dtype = (in_dtype() + 1j * in_dtype()).dtype
                out_shape = (nx, ny, nz + 2)
        else:
            if not issubclass(in_dtype, np.complexfloating):
                raise ValueError(
                    'Expected complex in_dtype for transform: {0}.'
                    .format(in_dtype))
            in_shape = out_shape = shape
            out_dtype = in_dtype

        # Allocate the input and output data buffers.
        self.data_in = memory.allocate(
            in_shape, in_dtype, use_pyfftw=use_pyfftw)
        if overwrite:
            if packed:
                self.data_out = self.data_in.view(out_dtype).reshape(out_shape)
            else:
                self.data_out = self.data_in
        else:
            self.data_out = memory.allocate(
                out_shape, out_dtype, use_pyfftw=use_pyfftw)

        # Try to use pyFFTW to configure the transform, if requested.
        self.use_pyfftw = use_pyfftw
        if self.use_pyfftw:
            try:
                import pyfftw
                direction = 'FFTW_BACKWARD' if inverse else 'FFTW_FORWARD'
                self.fftw_plan = pyfftw.FFTW(
                    self.data_in, self.data_out, direction=direction,
                    axes=(0, 1, 2))
                self.fftw_norm = np.float(nx * ny * nz)**0.5
            except ImportError:
                self.use_pyfftw = False

        # Fall back to numpy.fft if we are not using pyFFTW.
        if not self.use_pyfftw:
            if inverse:
                self.transformer = np.fft.irfftn if packed else np.fft.ifftn
            else:
                self.transformer = np.fft.rfftn if packed else np.fft.fftn

    def execute(self):
        if self.use_pyfftw:
            self.fftw_plan.execute()
            # FFTW does not apply any normalization factors so a round trip
            # gains a factor of nx*ny*nz.  The choice of how to split this
            # normalization factor between the forward and inverse transforms
            # is arbitrary, but we adopt the numpy.fft convention.
            self.data_out /= self.fftw_norm
        else:
            # This probably creates a temporary for the RHS, thus doubling
            # the peak memory requirements.  Any way to avoid this?
            self.data_out[:] = self.transformer(self.data_in, axes=(0,1,2))
        return self.data_out
