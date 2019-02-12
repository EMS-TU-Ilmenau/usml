import fastmat as fm
import numpy as np
import pulse_models
from fastmat.core.types cimport *
from fastmat.Matrix cimport Matrix


cdef class CudaBlockTwoLevelToeplitz(Matrix):

    def __init__(self, **options):
        self._nZ1 = options['nZ1']
        self._nZ2 = options['nZ2']
        self._nX = options['nX']
        self._nY = options['nY']
        self._pulseLength = options['pulseLength']
        self._dx = options['dx']
        self._dy = options['dy']
        self._dz = options['dz']
        self._centerFreq = options['centerFreq']
        self._bandWidth = options['bandWidth']
        self._speedOfSound = options['speedOfSound']
        self._samplingFreq = options['samplingFreq']
        self._foreRunLength = options['foreRunLength']
        self._beamAngle = options['beamAngle']

        self._cuda = new C_BlkTwoLvlToepGPU(
            self._nZ1,
            self._nZ2,
            self._nX,
            self._nY,
            self._pulseLength,
            self._dx,
            self._dy,
            self._dz,
            self._centerFreq,
            self._bandWidth,
            self._speedOfSound,
            self._samplingFreq,
            self._foreRunLength,
            self._beamAngle
        )

        numN = self._nZ1 * self._nX * self._nY
        numM = self._nZ2 * self._nX * self._nY

        self._cythonCall = True
        self._forceContiguousInput = True

        # set properties of matrix
        self._initProperties(
            numN, numM, np.float32,
            **options
        )

    cpdef np.ndarray getPulse(self):
        params = {'tPulse': self._pulseLength / self._samplingFreq,
                  'fCarrier': self._centerFreq,
                  'B': self._bandWidth,
                  'fS': self._samplingFreq
                }
        cdef np.ndarray pulse = pulse_models.compute_pulse(params)
        cdef np.ndarray a_scan = np.zeros((self._nZ1))
        a_scan[0:self._pulseLength] = pulse
        return a_scan

    cpdef np.ndarray _getColNorms(self):
        cdef np.ndarray arrNorms = np.zeros((self.numCols, 1)).astype(
            self.dtype
        )

        cdef arrX = np.ones((self.numRows, 1)).astype(
            self.dtype
        )


        self.backwardCuda(
            arrX[:, 0],
            arrNorms[:, 0],
            square = 1
        )

        return np.sqrt(arrNorms)

    cpdef np.ndarray _getRowNorms(self):
        cdef np.ndarray arrNorms = np.zeros((self.numRows, 1)).astype(
            self.dtype
        )

        cdef arrX = np.ones((self.numCols, 1)).astype(
            self.dtype
        )

        self.forwardCuda(
            arrX[:, 0],
            arrNorms[:, 0],
            square = 1
        )

        return np.sqrt(arrNorms)

    cpdef _forwardC(
        self,
        np.ndarray arrX,
        np.ndarray arrRes,
        ftype typeX,
        ftype typeRes
    ):
        if arrX.dtype in [np.complex64, np.complex128]:
            for ii in range(arrRes.shape[1]):
                arrInRe = np.real(arrX[:, ii]).astype(np.float32)
                arrInIm = np.imag(arrX[:, ii]).astype(np.float32)
                arrOutRe = np.zeros(arrRes.shape[0]).astype(np.float32)
                arrOutIm = np.zeros(arrRes.shape[0]).astype(np.float32)
                self.forwardCuda(
                    arrInRe[:],
                    arrOutRe[:],
                    square = 0
                )
                self.forwardCuda(
                    arrInIm[:],
                    arrOutIm[:],
                    square = 0
                )
                arrRes[:, ii] = arrOutRe + 1j * arrOutIm
        elif arrX.dtype in [np.float32, np.float64]:
            for ii in range(arrRes.shape[1]):
                arrX_ = arrX[:, ii].astype(np.float32)
                arrR_ = np.zeros(arrRes.shape[0]).astype(np.float32)
                self.forwardCuda(
                    arrX_,
                    arrR_,
                    square = 0
                )
                arrRes[:, ii] = arrR_[:]
        else:
            raise TypeError("Datatype not implemented")

    cpdef _backwardC(
        self,
        np.ndarray arrX,
        np.ndarray arrRes,
        ftype typeX,
        ftype typeRes
    ):
        if arrX.dtype in [np.complex64, np.complex128]:
            for ii in range(arrRes.shape[1]):
                arrInRe = np.real(arrX[:, ii]).astype(np.float32)
                arrInIm = np.imag(arrX[:, ii]).astype(np.float32)
                arrOutRe = np.zeros(arrRes.shape[0]).astype(np.float32)
                arrOutIm = np.zeros(arrRes.shape[0]).astype(np.float32)
                self.backwardCuda(
                    arrInRe[:],
                    arrOutRe[:],
                    square = 0
                )
                self.backwardCuda(
                    arrInIm[:],
                    arrOutIm[:],
                    square = 0
                )
                arrRes[:, ii] = arrOutRe + 1j * arrOutIm
        elif arrX.dtype in [np.float32, np.float64]:
            for ii in range(arrRes.shape[1]):
                arrX_ = arrX[:, ii].astype(np.float32)
                arrR_ = np.zeros(arrRes.shape[0]).astype(np.float32)
                self.backwardCuda(
                    arrX_,
                    arrR_,
                    square = 0
                )
                arrRes[:, ii] = arrR_[:]
        else:
            raise TypeError("Datatype not implemented")

    def forwardCuda(
        self,
        np.ndarray[ndim=1, dtype=np.float32_t] arrIn,
        np.ndarray[ndim=1, dtype=np.float32_t] arrOut,
        bint square = 0
    ):
        self._cuda.forward(&arrIn[0], &arrOut[0], square)

    def backwardCuda(
        self,
        np.ndarray[ndim=1, dtype=np.float32_t] arrIn,
        np.ndarray[ndim=1, dtype=np.float32_t] arrOut,
        bint square = 0
    ):
        self._cuda.backward(&arrIn[0], &arrOut[0], square)
