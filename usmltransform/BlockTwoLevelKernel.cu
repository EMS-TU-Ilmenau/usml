#include "BlockTwoLevelKernel.h"
#include <cuda.h>
#include <complex.h>
#include <assert.h>
#include <cuComplex.h>
#include <iostream>
#include <stdio.h>
#include <cufft.h>

// // length of the pulse in samples
// __constant__ float pulseLength = 11.0;
//
// // grid size in x
// __constant__ float dx = 0.5e-3;
//
// // grid size in y
// __constant__ float dy = 0.5e-3;
//
// // center frequency of the pulse
// __constant__ float centerFreq = 4e6;
// __constant__ float B = 3e6;
// // __constant__ float centerFreq = 3.4e6;
// // __constant__ float B = 2.2361e6;
//
// // speed of sound
// __constant__ float c0 = 5900.0;
//
// // sampling frequency
// __constant__ float fS = 40e6;
//
// // spacing in z
// __constant__ float dz = (0.5 * 5900.0)/40e6;
//
// // forerun
// __constant__ float foreRunLength = 8e-3;
//
// // opening angle of the transducer in radians
// __constant__ float beamAngle = 1.0;

void __global__ kernelDefElements(
    int ii,
    int jj,
    int nX,
    int nY,
    int nZ1,
    int nZ2,
    bool square,
    float* t,
    float pulseLength,
    float dx,
    float dy,
    float dz,
    float centerFreq,
    float bandWidth,
    float speedOfSound,
    float samplingFreq,
    float foreRunLength,
    float beamAngle
){

    // thread position within the (ii,jj)-Block
    int thrjj = threadIdx.x + blockDim.x*blockIdx.x;
    int thrii = threadIdx.y + blockDim.y*blockIdx.y;

    // depth coordinate of the hyperbola (increasing with columns -> ii)
    float posZhyp = __int2float_rn(ii);

    // depth coordinate within the current image (increasing with rows -> jj)
    float posZimg = __int2float_rn(jj);

    if (thrjj < (2 * nX - 1)){
        if (thrii < (2 * nY - 1)){
            float xidx = -fabs(
                __int2float_rn(thrjj) - (__int2float_rn(nX) - 0.5)
            ) + __int2float_rn(nX) - 0.5;

            float yidx = -fabs(
                __int2float_rn(thrii) - (__int2float_rn(nY) - 0.5)
            ) + __int2float_rn(nY) - 0.5;

            float xyLength = xidx * dx * xidx * dx + yidx * dy * yidx * dy;
            float zLength = posZhyp * dz + foreRunLength;

            float time_bin = nearbyint(
                (
                    sqrt(
                        zLength * zLength
                        + xyLength
                    ) - foreRunLength
                ) / speedOfSound * samplingFreq * 2
            );

            t[
                (2 * nY - 1) * thrjj + thrii
            ] = 0.0;

            if (time_bin < nZ1){
                if (fabs(time_bin - posZimg) <= 0.5 * pulseLength){
                    float n = (time_bin - posZimg)/samplingFreq;
                    t[
                        (2 * nY - 1) * thrjj + thrii
                    ] = exp(
                        -xyLength
                        / (beamAngle * zLength * beamAngle * zLength)
                        - bandWidth * bandWidth * n * n
                    )
                    * cos(6.28318530718 * centerFreq * n)
                    / ((2.0 * float(nX) - 1.0) * (2.0 * float(nY) - 1));
                }
            }
            if (square == 1){
                t[
                    (2 * nY - 1) * thrjj + thrii
                ] = t[
                    (2 * nY - 1) * thrjj + thrii
                ] *
                t[
                    (2 * nY - 1) * thrjj + thrii
                ] *
                ((2.0 * float(nX) - 1.0) * (2.0 * float(nY) - 1))
                ;
            }
            // t[
            //     (2 * nY - 1) * thrjj + thrii
            // ] = float(1000*ii + 100*jj + 10*thrjj + 1*thrii) / ((2 * nX - 1) * (2 * nY - 1));
        }
    }
}

void __global__ kernelMultForward(
    int nX,
    int nY,
    cuComplex * __restrict__ t,
    cuComplex * __restrict__ x,
    cuComplex * __restrict__ y
){
    // thread position within the (ii,jj)-Block
    int thrii = threadIdx.x + blockDim.x * blockIdx.x;
    int thrjj = threadIdx.y + blockDim.y * blockIdx.y;

    // check if we are still within the bounds of a single block
    if (thrjj < nY){
        if (thrii < (2 * nX - 1)){
            y[(2 * nX - 1) * thrjj + thrii] = cuCaddf(
                y[(2 * nX - 1) * thrjj + thrii],
                cuCmulf(
                    x[(2 * nX - 1) * thrjj + thrii],
                    t[(2 * nX - 1) * thrjj + thrii]
                )
            );
        }
    }
}

void __global__ kernelMultBackward(
    int nX,
    int nY,
    cuComplex * __restrict__ t,
    cuComplex * __restrict__ x,
    cuComplex * __restrict__ y
){
    // thread position within the (ii,jj)-Block
    int thrii = threadIdx.x + blockDim.x * blockIdx.x;
    int thrjj = threadIdx.y + blockDim.y * blockIdx.y;

    // check if we are still within the bounds of a single block
    if (thrjj < nY){
        if (thrii < (2 * nX - 1)){
            y[(2 * nX - 1) * thrjj + thrii] = cuCaddf(
                y[(2 * nX - 1) * thrjj + thrii],
                cuCmulf(
                    x[(2 * nX - 1) * thrjj + thrii],
                    cuConjf(t[(2 * nX - 1) * thrjj + thrii])
                )
            );
        }
    }
}

void __global__ printMem2D(
    float* mem,
    int nX,
    int nY
){
    // thread positions
    int thrjj = threadIdx.x + blockDim.x * blockIdx.x;
    int thrii = threadIdx.y + blockDim.y * blockIdx.y;

    // check if we are still within the bounds of a single block
    if (thrjj < (2*nY-1)){
        if (thrii < (2*nX-1)){
            printf("(%d,%d) – %f\n", thrjj, thrii, mem[(2 * nY - 1) * thrjj + thrii]);
        }
    }
}

void __global__ printMem2D(
    cuComplex* mem,
    int nX,
    int nY
){
    // thread positions
    int thrjj = threadIdx.x + blockDim.x * blockIdx.x;
    int thrii = threadIdx.y + blockDim.y * blockIdx.y;

    // check if we are still within the bounds of a single block
    if (thrjj < (2*nY-1)){
        if (thrii < (2*nX-1)){
            printf("(%d,%d) – %f,%f\n", thrjj, thrii, cuCrealf(mem[(2 * nY - 1) * thrjj + thrii]), cuCimagf(mem[(2 * nY - 1) * thrjj + thrii]));
        }
    }
}

void __global__ printMem(
    float* mem,
    int N
){
    int thrjj = threadIdx.x + blockDim.x * blockIdx.x;
    if(thrjj < N){
        printf("%d – %f\n", thrjj, mem[thrjj]);
    }
}

void __global__ printMem(
    cuComplex* mem,
    int N
){
    int thrjj = threadIdx.x + blockDim.x * blockIdx.x;
    if(thrjj < N){
        printf("%d – %f,%f\n", thrjj, cuCrealf(mem[thrjj]), cuCimagf(mem[thrjj]));
    }
}

void printMem(){
    size_t f;
    size_t t;
    cudaMemGetInfo(&f,&t);
    printf("Free: %zd; Total: %zd; \n", f,t);
}

BlkTwoLvlToepGPU::BlkTwoLvlToepGPU (
    int nZ1_,
    int nZ2_,
    int nX_,
    int nY_,
    float pulseLength_,
    float dx_,
    float dy_,
    float dz_,
    float centerFreq_,
    float bandWidth_,
    float speedOfSound_,
    float samplingFreq_,
    float foreRunLength_,
    float beamAngle_
){
    // number of block rows
    nZ1 = nZ1_;
    // number of block columns
    nZ2 = nZ2_;

    // sizes of the levels
    nX = nX_;
    nY = nY_;

    pulseLength = pulseLength_;
    dx = dx_;
    dy = dy_;
    dz = dz_;
    centerFreq = centerFreq_;
    bandWidth = bandWidth_;
    speedOfSound = speedOfSound_;
    samplingFreq = samplingFreq_;
    foreRunLength = foreRunLength_;
    beamAngle = beamAngle_;

    // these plans transform each block forward or backward
    cufftPlan2d(
        &planBlockForward,
        2 * nX - 1,
        2 * nY - 1,
        CUFFT_R2C
    );

    cufftPlan2d(
        &planBlockBackward,
        2 * nX - 1,
        2 * nY - 1,
        CUFFT_C2R
    );


    // allocate memory for defining elements of the blocks
    cudaMalloc(
        (void**) &arrT_device,
        sizeof(float) * (2 * nX - 1) * (2 * nY - 1)
    );

    cudaMalloc(
        (void**) &arrThat_device,
        sizeof(cufftComplex) * (2 * nX - 1) * nY
    );

    // allocate memory for the blocks of the input vector (zero padded)
    cudaMalloc(
        (void**) &arrXbuffer_device,
        sizeof(float) * (2 * nX - 1) * (2 * nY - 1)
    );

    // we only need to write zeros once, since non-zero values are
    // always written to the same places
    cudaMemset(
        arrXbuffer_device,
        0,
        sizeof(float) * (2 * nX - 1) * (2 * nY - 1)
    );

    // allocate memory for the whole input vector
    // in frequency domain. this is why we have (nY) instead of (2*nY-1)
    // since we apply twodimensional real FFTs
    cudaMalloc(
        (void**) &arrXhat_device,
        sizeof(cufftComplex) * nZ2 * (2 * nX - 1) * nY
    );

    // memory for the whole output vector
    // in frequency domain. this is why we have (nY) instead of (2*nY-1)
    // since we apply twodimensional real FFTs
    cudaMalloc(
        (void**) &arrYhat_device,
        sizeof(cufftComplex) * nZ1 * (2 * nX - 1) * nY
    );

    // allocate memory for the blocks of output vector
    // (matching the zero padded vector arrXbuffer_device)
    cudaMalloc(
        (void**) &arrYbuffer_device,
        sizeof(float) * (2 * nX - 1) * (2 * nY - 1)
    );

    blockSize = dim3(32, 32, 1);
    gridSizeTime = dim3(
        int(float(2 * nX - 1) / float(blockSize.x)) + 1,
        int(float(2 * nY - 1) / float(blockSize.y)) + 1,
        1
    );
    gridSizeFreq = dim3(
        int(float(2 * nX - 1) / float(blockSize.x)) + 1,
        int(float(nY) / float(blockSize.y)) + 1,
        1
    );
}

void BlkTwoLvlToepGPU::forward(
    float* arrX,
    float* arrY,
    bool square = 0
) {
    /*
    Calculation Routine for the Forward Projection

    0. Step:
        Initalization

    1. Step:
        Copy the input data arrX, which is a 1D array
        from the host to the device and apply the 2D FFTs blockwise

    2. Step:
        a) Iterate over the rows of the blocks
            b) Iterate over columns of the blocks
                - calc the elements of the current block in
                  parallel -> kernelDefElements
                - wait for all threads to finish
                - calc the 2dfft of the current generating
                  elements inplace
                - do the multiplication simultaneously
                  and in parallel and write in
                  arrY_device -> kernelMult
                - wait for all threads to finish

    3. Step:
        Do blockwise 2D iFFTs and copy these blocks' parts that belong to
        the output iteratively
    */

    // 0. Step:


    cudaMemset(
        arrYhat_device,
        0,
        sizeof(cuComplex) * nZ1 * (2 * nX - 1) * nY
    );
    cudaThreadSynchronize();

    // 1. Step:
    // we apply the zero padding right during copying memory
    // we can do it out of sync since the regions do not
    // overlap
    // if we have finished copying a whole block, we apply
    // the 2D FFT to this block
    for (int kk = 0; kk < nZ2; kk++){
        for (int ii = 0; ii < nX; ii++){
            cudaMemcpyAsync(
                &arrXbuffer_device[
                    ii * (2 * nY - 1)
                ],
                &arrX[
                    kk * nX * nY + ii * nY
                ],
                sizeof(float) * nY,
                cudaMemcpyHostToDevice
            );
        }
        cudaThreadSynchronize();
        cufftExecR2C(
            planBlockForward,
            reinterpret_cast<cufftReal*>(
                arrXbuffer_device
            ),
            reinterpret_cast<cufftComplex*>(
                &arrXhat_device[
                    kk * nY * (2 * nX - 1)
                ]
            )
        );
    }
    cudaThreadSynchronize();


    // 2. Step:
    // iterate through the columns -> ii goes through arrX
    for (int ii = 0; ii < nZ2; ii++){

        // iterate through the rows -> jj goes to arrY
        for (int jj = 0; jj < nZ1; jj++){

            kernelDefElements<<<gridSizeTime,blockSize>>>(
                ii,
                jj,
                nX,
                nY,
                nZ1,
                nZ2,
                square,
                arrT_device,
                pulseLength,
                dx,
                dy,
                dz,
                centerFreq,
                bandWidth,
                speedOfSound,
                samplingFreq,
                foreRunLength,
                beamAngle
            );

            cudaThreadSynchronize();

            cufftExecR2C(
                planBlockForward,
                reinterpret_cast<cufftReal*>(arrT_device),
                reinterpret_cast<cufftComplex*>(arrThat_device)
            );

            cudaThreadSynchronize();

            kernelMultForward<<<gridSizeFreq,blockSize>>>(
                nX,
                nY,
                arrThat_device,
                &arrXhat_device[ii * (2 * nX - 1) * nY],
                &arrYhat_device[jj * (2 * nX - 1) * nY]
            );

            cudaThreadSynchronize();
        }
    }

    // 3. Step:
    // apply the backward transforms and copy the results
    // to the respective places in the output vector
    for (int kk = 0; kk < nZ1; kk++){
        cufftExecC2R(
            planBlockBackward,
            reinterpret_cast<cufftComplex*>(
                &arrYhat_device[
                    kk * nY * (2 * nX - 1)
                ]
            ),
            reinterpret_cast<cufftReal*>(
                arrYbuffer_device
            )
        );

        cudaThreadSynchronize();
        for (int ii = 0; ii < nX; ii++){
            cudaMemcpyAsync(
                &arrY[
                    kk * nX * nY + ii * nY
                ],
                &arrYbuffer_device[
                    ii * (2 * nY - 1)
                ],
                sizeof(float) * nY,
                cudaMemcpyDeviceToHost
            );
        }
    }
    cudaThreadSynchronize();
}


void BlkTwoLvlToepGPU::backward(
    float* arrX,
    float* arrY,
    bool square = 0
) {
    /*
    Calculation Routine for the Backward Projection

    See forward above
    */

    // 0. Step:


    cudaMemset(
        arrYhat_device,
        0,
        sizeof(cuComplex) * nZ2 * (2 * nX - 1) * nY
    );
    cudaThreadSynchronize();

    // 1. Step:
    // we apply the zero padding right during copying memory
    // we can do it out of sync since the regions do not
    // overlap
    // if we have finished copying a whole block, we apply
    // the 2D FFT to this block
    for (int kk = 0; kk < nZ1; kk++){
        for (int ii = 0; ii < nX; ii++){
            cudaMemcpyAsync(
                &arrXbuffer_device[
                    ii * (2 * nY - 1)
                ],
                &arrX[
                    kk * nX * nY + ii * nY
                ],
                sizeof(float) * nY,
                cudaMemcpyHostToDevice
            );
        }
        cudaThreadSynchronize();
        cufftExecR2C(
            planBlockForward,
            reinterpret_cast<cufftReal*>(
                arrXbuffer_device
            ),
            reinterpret_cast<cufftComplex*>(
                &arrXhat_device[
                    kk * nY * (2 * nX - 1)
                ]
            )
        );
    }
    cudaThreadSynchronize();


    // 2. Step:
    // iterate through the columns -> ii goes through arrX
    for (int ii = 0; ii < nZ1; ii++){

        // iterate through the rows -> jj goes to arrY
        for (int jj = 0; jj < nZ2; jj++){

            kernelDefElements<<<gridSizeTime,blockSize>>>(
                jj,
                ii,
                nX,
                nY,
                nZ1,
                nZ2,
                square,
                arrT_device,
                pulseLength,
                dx,
                dy,
                dz,
                centerFreq,
                bandWidth,
                speedOfSound,
                samplingFreq,
                foreRunLength,
                beamAngle
            );

            cudaThreadSynchronize();

            cufftExecR2C(
                planBlockForward,
                reinterpret_cast<cufftReal*>(arrT_device),
                reinterpret_cast<cufftComplex*>(arrThat_device)
            );

            cudaThreadSynchronize();

            kernelMultBackward<<<gridSizeFreq,blockSize>>>(
                nX,
                nY,
                arrThat_device,
                &arrXhat_device[ii * (2 * nX - 1) * nY],
                &arrYhat_device[jj * (2 * nX - 1) * nY]
            );

            cudaThreadSynchronize();
        }
    }

    // 3. Step:
    // apply the backward transforms and copy the results
    // to the respective places in the output vector
    for (int kk = 0; kk < nZ2; kk++){
        cufftExecC2R(
            planBlockBackward,
            reinterpret_cast<cufftComplex*>(
                &arrYhat_device[
                    kk * nY * (2 * nX - 1)
                ]
            ),
            reinterpret_cast<cufftReal*>(
                arrYbuffer_device
            )
        );

        cudaThreadSynchronize();
        for (int ii = 0; ii < nX; ii++){
            cudaMemcpyAsync(
                &arrY[
                    kk * nX * nY + ii * nY
                ],
                &arrYbuffer_device[
                    ii * (2 * nY - 1)
                ],
                sizeof(float) * nY,
                cudaMemcpyDeviceToHost
            );
        }
    }
    cudaThreadSynchronize();
}



BlkTwoLvlToepGPU::~BlkTwoLvlToepGPU() {
    cudaFree(arrXbuffer_device);
    cudaFree(arrT_device);
    cudaFree(arrYbuffer_device);
    cudaFree(arrXhat_device);
    cudaFree(arrThat_device);
    cudaFree(arrYhat_device);
    cufftDestroy(planBlockForward);
    cufftDestroy(planBlockBackward);
}
