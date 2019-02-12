import cython
cimport numpy as np

from fastmat.core.types cimport *
from fastmat.Matrix cimport Matrix

cdef extern from "BlockTwoLevelKernel.h":
    cdef cppclass C_BlkTwoLvlToepGPU "BlkTwoLvlToepGPU":
        C_BlkTwoLvlToepGPU(
            int,
            int,
            int,
            int,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float
        )
        void forward(
            np.float32_t*,
            np.float32_t*,
            bint
        )
        void backward(
            np.float32_t*,
            np.float32_t*,
            bint
        )


cdef class CudaBlockTwoLevelToeplitz(Matrix):
    cdef C_BlkTwoLvlToepGPU* _cuda

    cdef public int _nZ1
    cdef public int _nZ2
    cdef public int _nX
    cdef public int _nY
    cdef public float _pulseLength
    cdef public float _dx
    cdef public float _dy
    cdef public float _dz
    cdef public float _centerFreq
    cdef public float _bandWidth
    cdef public float _speedOfSound
    cdef public float _samplingFreq
    cdef public float _foreRunLength
    cdef public float _beamAngle

    cpdef _forwardC(self, np.ndarray, np.ndarray, ftype, ftype)
    cpdef _backwardC(self, np.ndarray, np.ndarray, ftype, ftype)

    cpdef np.ndarray getPulse(self)

    cpdef np.ndarray _getColNorms(self)
    cpdef np.ndarray _getRowNorms(self)
