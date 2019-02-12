#!/bin/bash

# build the cuda stuff
nvcc -arch=compute_60 -lcuda -lcudart -lcufft -I/usr/include/ -Xcompiler -fPIC -c -o usmltransform/BlockTwoLevelKernel.o usmltransform/BlockTwoLevelKernel.cu

# build the module
python3 setup.py build_ext --inplace
