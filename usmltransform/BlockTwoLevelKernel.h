#include <cuda.h>
#include <complex.h>
#include <cufft.h>
#include <cufftXt.h>
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <stdbool.h>

class BlkTwoLvlToepGPU{

    // pointer to the GPU memory where the arrays are stored

    // since we copy and apply the FFT to the input blockwise,
    // we need a buffer of the length of each block
    float* arrXbuffer_device;

    // the whole input vector in blockwise 2D frequency domain
    cuComplex * arrXhat_device;

    // defining elements in time domain (per TwoLvlToep-Block)
    float* arrT_device;

    // defining elements in frequency domain (per TwoLvlToep-Block)
    cuComplex* arrThat_device;

    // the whole output vector in blockwise 2D frequency domain
    cuComplex* arrYhat_device;

    // since we do the blockwise ffts and the copying on the fly, we
    // need a buffer to store the result of the inverse fft per block
    float* arrYbuffer_device;

    // number of blocks
    int nZ1;
    int nZ2;

    // size of the first level (NOT number of defining elements!)
    int nX;

    // size of the second level (NOT number of  defining elements!)
    int nY;

    float pulseLength;
    float dx;
    float dy;
    float dz;
    float centerFreq;
    float bandWidth;
    float speedOfSound;
    float samplingFreq;
    float foreRunLength;
    float beamAngle;

    cufftCallbackStoreC h_storeCallbackPtr;
    // plan for the forward transform of each block
    cufftHandle planBlockForward;

    // plan for the backward transform of each block
    cufftHandle planBlockBackward;

    // organization of the cuda threads
    dim3 blockSize;
    dim3 gridSizeTime;
    dim3 gridSizeFreq;

    public:
        // the constructor
        BlkTwoLvlToepGPU(
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
        );

        ~BlkTwoLvlToepGPU();

        void forward(
            float*,
            float*,
            bool
        );

        void backward(
            float*,
            float*,
            bool
        );
};
